"""Regressions for export privacy and evidence-backed analytical claims."""

import json
from dataclasses import replace

import pandas as pd
import pytest

from oribi import Evidence, ProfileOptions, Warning, inspect


def test_shareable_exports_omit_sensitive_values_by_default():
    frame = pd.DataFrame({
        "a": ["Alice", "Bob"],
        "b": ["alice@example.org", "bob@example.org"],
        "c": ["HIV", "cancer"],
    })
    context = inspect(frame)
    secrets = ("Alice", "Bob", "alice@example.org", "bob@example.org", "HIV", "cancer")
    for exported in (context.to_json(), json.dumps(context.to_dict()), context.to_markdown(), context.to_context()):
        assert not any(value in exported for value in secrets)
    assert all(value in context.to_json(include_values=True) for value in secrets)
    assert all(value in context.to_markdown(include_values=True) for value in secrets)
    assert context.columns[0].statistics.top_values  # Local snapshot retains facts.


def test_exact_bounds_and_tiny_numeric_summaries_require_opt_in():
    frame = pd.DataFrame({
        "reading": [812345.0, 912345.0],
        "date": pd.to_datetime(["2026-01-03", "2026-02-04"]),
    })
    context = inspect(frame)
    safe = context.to_json()
    assert "812345" not in safe and "912345" not in safe
    assert "2026-01-03" not in safe and "2026-02-04" not in safe
    assert context.to_dict()["columns"][0]["statistics"]["mean"] is None
    opt_in = context.to_json(include_values=True)
    assert "812345" in opt_in and "2026-01-03" in opt_in


def test_markdown_renders_opt_in_samples_as_literal_text():
    sample = "<em>example</em>\n**second line**"
    markdown = inspect(pd.DataFrame({"record_id": [sample, "other"]})).to_markdown(include_values=True)
    assert "<em>" not in markdown
    assert r"- Local value samples: \<em\>example\</em\> \*\*second line\*\* (1)" in markdown

    markdown = inspect(pd.DataFrame({"value": [sample]})).to_markdown(include_values=True)
    assert "<em>" not in markdown
    assert r'- Categories (all 1): "\<em\>example\</em\>\\n\*\*second line\*\*" (1)' in markdown


def test_nested_evidence_strings_cannot_bypass_export_redaction():
    context = inspect(pd.DataFrame({"ordinary": ["a", "b"]}))
    warning = Warning("example", "low", "Example", "No raw values in message", (
        Evidence("nested", "Structured evidence", {"metadata": {"sample": "secret-token", "count": 2}}),
    ))
    context = replace(context, warnings=(warning,))
    assert "secret-token" not in context.to_json()
    assert "secret-token" in context.to_json(include_values=True)
    assert context.to_dict()["warnings"][0]["evidence"][0]["details"]["metadata"]["count"] == 2


def test_near_unique_pair_is_diagnostic_not_observation_grain():
    frame = pd.DataFrame({
        "order_id": [i // 2 for i in range(100)],
        "product_id": [i % 2 for i in range(90)] + [0] * 10,
        "revenue": [float(i) for i in range(100)],
    })
    context = inspect(frame)
    pair = next(key for key in context.candidate_keys if key.columns == ("order_id", "product_id"))
    assert (pair.uniqueness_rate, pair.duplicate_count, pair.missing_count, pair.verified_unique) == (0.95, 5, 0, False)
    assert context.grain.status == "undetermined"
    assert "jointly identify" not in context.to_context()


def test_near_unique_single_column_is_not_rendered_as_identifying_in_markdown():
    context = inspect(pd.DataFrame({"record_key": [f"k{i}" for i in range(99)] + ["k0"]}))
    rendered = context.to_markdown()
    assert "mostly-unique observation-key candidate" in rendered
    assert "record_key** appears to identify observations" not in rendered


def test_missing_key_components_are_not_verified():
    context = inspect(pd.DataFrame({"order_id": [1, 1, 2, 2], "item": [None, "a", None, "b"]}))
    pair = next(key for key in context.candidate_keys if key.kind == "composite")
    assert pair.missing_count == 2
    assert not pair.verified_unique
    assert context.grain.status == "undetermined"


def test_repeated_amount_is_not_an_entity_group():
    context = inspect(pd.DataFrame({"record_id": range(100), "amount": [10.0, 20.0] * 50}))
    assert next(col for col in context.columns if col.name == "amount").semantic.role == "measure"
    assert not any(rel.source_column == "amount" and rel.kind == "one_to_many" for rel in context.relationships)
    assert "amount groups observations" not in context.to_context()


def test_known_mapping_contradiction_is_approximate_only():
    source = [f"s{i}" for i in range(50) for _ in range(2)]
    target = source.copy()
    target[-1] = source[0]
    context = inspect(pd.DataFrame({"source": source, "target": target}))
    assert not any(rel.kind in {"functional_dependency", "one_to_one"} for rel in context.relationships)
    approximate = [rel for rel in context.relationships if rel.kind == "approximate_mapping"]
    assert approximate
    assert all((rel.violation_count, rel.affected_groups, rel.coverage) == (1, 1, 0.98) for rel in approximate)


def test_mapping_with_null_target_is_not_an_exact_dependency():
    source = [f"s{i}" for i in range(50) for _ in range(2)]
    target = source.copy()
    target[-1] = None
    relations = inspect(pd.DataFrame({"source": source, "target": target})).relationships
    assert not any(rel.kind == "functional_dependency" for rel in relations)
    assert any(rel.kind == "approximate_mapping" and rel.missing_pair_count == 1 for rel in relations)


def test_many_to_one_dependency_is_detected_in_the_correct_direction():
    frame = pd.DataFrame({
        "product": [f"p{i // 2}" for i in range(40)],
        "category": [f"c{i // 10}" for i in range(40)],
    })
    relations = inspect(frame).relationships
    assert any(rel.kind == "functional_dependency" and rel.source_column == "product" and rel.target_column == "category" for rel in relations)
    assert not any(rel.kind == "functional_dependency" and rel.source_column == "category" and rel.target_column == "product" for rel in relations)


@pytest.mark.parametrize("seed", range(5))
def test_randomized_exact_mapping_never_has_observed_contradictions(seed):
    import random

    rng = random.Random(seed)
    source = [f"g{i % 15}" for i in range(150)]
    target = [f"v{int(value[1:])}" for value in source]
    for _ in range(seed):
        target[rng.randrange(len(target))] = "conflict"
    frame = pd.DataFrame({"source": source, "target": target})
    relations = inspect(frame).relationships
    for relation in relations:
        if relation.kind == "functional_dependency":
            pair_counts = frame[[relation.source_column, relation.target_column]].groupby(relation.source_column)[relation.target_column].nunique()
            assert pair_counts.max() == 1
            assert relation.violation_count == 0


@pytest.mark.parametrize("width", [10, 40, 100, 200])
def test_wide_context_retains_schema_and_bounds_relationships(width):
    frame = pd.DataFrame({f"category_{i}": [f"v{(row + i) % 10}" for row in range(100)] for i in range(width)})
    context = inspect(frame)
    output = context.to_context()
    assert len(context.relationships) <= 40
    assert len(output) <= 8000
    assert "COLUMNS" in output
    assert all(f"- category_{i}:" in output for i in range(width))
    if width >= 40:
        assert "[Context truncated.]" in output


def test_long_temporal_gap_has_bounded_evidence():
    context = inspect(pd.DataFrame({"date": pd.to_datetime([
        "1900-01-01", "1900-01-02", "1900-01-03", "2000-01-01",
    ])}), options=ProfileOptions(max_gap_examples=2))
    evidence = context.columns[0].warnings[0].evidence[0].details
    assert evidence["missing_intervals"] > 36_000
    assert len(evidence["expected_timestamps"]) == 2
    assert evidence["examples_omitted"] == evidence["missing_intervals"] - 2
    assert len(context.to_json()) < 20_000


def test_gap_example_limit_can_disable_samples():
    context = inspect(pd.DataFrame({"date": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-10",
    ])}), options=ProfileOptions(max_gap_examples=0))
    assert context.columns[0].warnings[0].evidence[0].details["expected_timestamps"] == ()


def test_low_diversity_id_is_not_high_confidence_entity_identifier():
    context = inspect(pd.DataFrame({"status_id": [1, 2] * 50, "amount": [float(i) for i in range(100)]}))
    status = next(col for col in context.columns if col.name == "status_id")
    assert status.semantic.role != "identifier"
    assert not any(rec.operation == "COUNT DISTINCT" for rec in status.recommendations)


def test_named_and_multiindex_are_exported_as_index_metadata():
    named = inspect(pd.DataFrame({"amount": [1.0, 2.0]}, index=pd.Index([10, 11], name="order_id")))
    assert named.profile.index.names == ("order_id",)
    assert named.profile.index.is_unique
    assert "reset_index()" in named.to_context()
    multi = pd.MultiIndex.from_tuples([("A", 1), ("A", 1), ("B", 2)], names=["group", "period"])
    result = inspect(pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=multi))
    assert result.profile.index.level_count == 2
    assert result.profile.index.duplicate_count == 1
    assert result.to_dict()["profile"]["index"]["names"] == ["group", "period"]
    default = inspect(pd.DataFrame({"value": [1.0, 2.0]}))
    assert default.profile.index is None


@pytest.mark.parametrize(
    ("values", "unique", "duplicates", "missing", "is_unique"),
    [
        ([10, 11, None], 2, 0, 1, True),
        ([10, 10, None], 1, 1, 1, False),
        ([None, None], 0, 1, 2, False),
        ([10, 11, 12], 3, 0, 0, True),
        (["a", None, "a", None], 1, 2, 2, False),
    ],
)
def test_single_index_missing_values_are_not_duplicates(values, unique, duplicates, missing, is_unique):
    index = pd.Index(values, name="record_id")
    profile = inspect(pd.DataFrame({"value": range(len(values))}, index=index)).profile.index
    assert profile.unique_count == unique
    assert profile.duplicate_count == duplicates
    assert profile.duplicate_count == int(index.duplicated().sum())
    assert profile.missing_count == missing
    assert profile.is_unique is is_unique
    assert profile.is_unique == index.is_unique


@pytest.mark.parametrize(
    ("values", "unique", "duplicates", "missing", "is_unique"),
    [
        ([{"store": "a"}, {"store": "a"}, {"store": "b"}, None], 2, 1, 1, False),
        ([{"a": 1, "b": 2}, {"b": 2, "a": 1}, {"a": 2, "b": 1}, None], 2, 1, 1, False),
        ([{"store": "a"}, {"store": "b"}, None], 2, 0, 1, True),
        ([{"store": "a"}, {"store": "a"}, None, None], 1, 2, 2, False),
        ([["a"], ["a"], ["b"], None], 2, 1, 1, False),
    ],
    ids=["dictionaries", "dictionary-key-order", "unique", "repeated-missing", "lists"],
)
def test_unhashable_single_index_labels_are_profiled(values, unique, duplicates, missing, is_unique):
    index = pd.Index(values, dtype=object, name="record_id")
    context = inspect(pd.DataFrame({"quantity": range(len(values))}, index=index))
    profile = context.profile.index

    assert profile.names == ("record_id",)
    assert profile.dtypes == ("object",)
    assert profile.level_count == 1
    assert profile.unique_count == unique
    assert profile.duplicate_count == duplicates
    assert profile.missing_count == missing
    assert profile.is_unique is is_unique
    assert context.to_dict()["profile"]["index"]["unique_count"] == unique


def test_multiindex_missing_values_are_counted_like_pandas():
    multi = pd.MultiIndex.from_tuples([("A", None), ("A", None), ("B", 1)], names=["group", "period"])
    profile = inspect(pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=multi)).profile.index
    assert profile.duplicate_count == 1
    assert profile.missing_count == 2
    assert profile.is_unique is False


def test_multiindex_with_repeated_names_is_profiled():
    index = pd.MultiIndex.from_tuples(
        [("us", 1), ("us", 1), ("ca", 2), (None, 3)], names=["key", "key"]
    )
    result = inspect(pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}, index=index))
    profile = result.profile.index

    assert profile.names == ("key", "key")
    assert profile.level_count == 2
    assert profile.unique_count == 3
    assert profile.missing_count == 1
    assert profile.duplicate_count == 1
    assert profile.is_unique is False
    assert result.to_dict()["profile"]["index"]["names"] == ["key", "key"]


@pytest.mark.parametrize(
    ("index", "label"),
    [
        (pd.Index([10, 11], name=42), "42"),
        (pd.MultiIndex.from_tuples([("A", 1), ("B", 2)], names=[0, None]), "0, level 1"),
    ],
)
@pytest.mark.parametrize("renderer", ["to_markdown", "to_context"])
def test_renderers_display_non_string_index_names(index, label, renderer):
    context = inspect(pd.DataFrame({"amount": [1.0, 2.0]}, index=index))
    assert f"({label})" in getattr(context, renderer)()
    assert context.profile.index.names == tuple(index.names)


def test_unique_duration_column_is_not_an_identifier_or_grain():
    context = inspect(pd.DataFrame({"elapsed": pd.to_timedelta(range(1, 11), unit="h")}))
    column = context.columns[0]
    assert column.semantic.role != "identifier"
    assert context.grain.status == "undetermined"
    assert context.grain.columns == ()
    assert not any(candidate.verified_unique for candidate in context.candidate_keys)


def test_duration_column_does_not_displace_integer_identifier_or_datetime_behavior():
    frame = pd.DataFrame(
        {
            "order_id": range(1, 11),
            "elapsed": pd.to_timedelta(range(1, 11), unit="h"),
            "placed_at": pd.date_range("2024-01-01", periods=10, freq="D"),
        }
    )
    context = inspect(frame)
    roles = {column.name: column.semantic.role for column in context.columns}
    assert roles["order_id"] == "identifier"
    assert roles["placed_at"] == "temporal"
    assert roles["elapsed"] != "identifier"
    assert context.grain.columns == ("order_id",)


def test_to_context_categorical_values_respect_max_chars_without_dropping_sections():
    frame = pd.DataFrame({
        "id": range(1, 6),
        "category": [
            "alpha_long_category_name",
            "beta_long_category_name",
            "gamma_long_category_name",
            "delta_long_category_name",
            "alpha_long_category_name",
        ],
        "z_measure": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    context = inspect(frame)

    full_ctx = context.to_context(max_chars=8000, include_values=True)
    assert "CARDINALITY" in full_ctx
    assert "Categories (all 4):" in full_ctx
    measure_section = full_ctx.split("\nz_measure\n", 1)[1]
    assert "Aggregation basis:" in measure_section

    without_categories = "\n".join(
        line for line in full_ctx.splitlines() if not line.startswith("Categories (")
    )
    tight_budget = len(without_categories)

    tight_ctx = context.to_context(max_chars=tight_budget, include_values=True)
    assert len(tight_ctx) <= tight_budget
    assert "CARDINALITY" in tight_ctx
    assert "Categories (all 4):" not in tight_ctx
    assert "[Context truncated.]" not in tight_ctx
    assert tight_ctx == without_categories
