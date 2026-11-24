import json
from dataclasses import replace
from numbers import Real

import numpy as np
import pandas as pd
import pytest

from oribi import ProfileOptions, inspect
from oribi.models.evidence import CategoricalContext
from oribi.serialization.context import _format_aggregations, _format_range_value
from oribi.serialization.dict import _normalize


def test_serialization_outputs(sales_frame):
    context = inspect(sales_frame)
    assert json.loads(context.to_json()) == context.to_dict()
    assert "## Columns" in context.to_markdown()
    compact = context.to_context(max_chars=500)
    assert len(compact) <= 500
    assert "DATASET" in compact


@pytest.mark.parametrize(
    ("values", "observed_count", "ratio"),
    [
        ([True, True, False, False], 4, 0.5),
        ([True, True, False, None, None], 3, 2 / 3),
    ],
    ids=["repeated-boolean", "masked-rows"],
)
def test_compact_cardinality_describes_distinct_value_ratio(values, observed_count, ratio):
    context = inspect(pd.DataFrame({"active": pd.array(values, dtype="boolean")}))
    card = context.columns[0].cardinality
    assert card is not None
    assert card.unique_count == 2
    assert card.observed_count == observed_count
    assert card.uniqueness_ratio == pytest.approx(ratio)

    prefix = f"- active: 2 unique / {len(values)} rows"
    line = next(line for line in context.to_context().splitlines() if line.startswith(prefix))
    assert line.startswith(
        f"{prefix} ({card.uniqueness_ratio:.0%} distinct values / non-missing rows)"
        f"; {observed_count} non-missing"
    )


def test_compact_cardinality_preserves_no_observed_values():
    context = inspect(pd.DataFrame({"active": pd.array([None, None], dtype="boolean")}))
    card = context.columns[0].cardinality
    assert card is not None
    assert card.observed_count == 0
    assert card.uniqueness_ratio is None
    assert "- active: 0 unique / 2 rows; no observed values" in context.to_context().splitlines()


def test_context_renderers_use_scannable_hierarchy_and_compact_columns(sales_frame):
    context = inspect(sales_frame)
    compact = context.to_context()
    assert compact.index("DATASET\n") < compact.index("GRAIN") < compact.index("COLUMNS")
    assert "Missing " in compact
    assert "Unique " in compact
    assert "Aggregations:" in compact
    assert "→" in context.to_context(include_values=True)
    assert "Evidence:" not in compact
    assert "confidence" in compact
    assert "DATASET SUMMARY" not in compact
    assert "- order_date: temporal dimension" in compact
    assert " · " in compact
    assert "Missing 25.0%" not in compact

    markdown = context.to_markdown()
    assert markdown.index("## Dataset summary") < markdown.index("## Grain") < markdown.index("## Columns")
    assert "Recommended aggregations:" in markdown
    assert "<details>" in markdown
    assert "<summary>Inference evidence</summary>" in markdown


def test_compact_context_formats_column_warnings_concisely():
    import pandas as pd

    context = inspect(pd.DataFrame({"region": ["East", "West", "East", None]}))
    rendered = context.to_context()
    assert "⚠ region · 25% missing" in rendered
    assert "[medium]" not in rendered


def test_text_renderers_surface_warning_and_aggregation_evidence(sales_frame):
    context = inspect(sales_frame)
    before = context.to_dict()
    warning = next(w for w in context.warnings if w.code == "negative_measure_values")
    warning_basis = warning.evidence[0].statement
    revenue = next(column for column in context.columns if column.name == "revenue")
    recommendation_basis = revenue.recommendations[0].evidence[0].statement

    markdown = context.to_markdown()
    assert f"Basis: {warning_basis}" in markdown
    assert markdown.count(warning_basis) == 1

    compact = context.to_context()
    assert compact.count(warning_basis) == 1
    revenue_section = compact.split("\nrevenue\n", 1)[1]
    assert "Aggregations: SUM · MEAN" in revenue_section
    assert "Aggregation basis: " + recommendation_basis in revenue_section
    assert revenue_section.count(recommendation_basis) == 1

    truncated = context.to_context(max_chars=700)
    assert len(truncated) <= 700
    assert "[Context truncated.]" in truncated
    assert warning.message in truncated
    assert context.to_dict() == before


def test_context_preserves_aggregations_when_only_evidence_exceeds_budget():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [1.0, 2.0, 3.0]}))
    full = context.to_context()
    report = "\n".join(line for line in full.splitlines() if not line.startswith("Aggregation basis:"))

    constrained = context.to_context(max_chars=len(report))

    assert "Aggregations: SUM · MEAN" in constrained
    assert constrained == report


def test_truncated_context_keeps_qualified_operations_and_warnings():
    import pandas as pd

    context = inspect(pd.DataFrame({
        "account_id": ["a", "a", "b", None],
        **{f"measure_{i:02}": [10.0, 20.0, 40.0, 90.0] for i in range(17)},
    }))
    rendered = context.to_context()
    assert len(rendered) <= 8000
    assert rendered.endswith("[Context truncated.]")
    assert rendered.count("SUM only for additive quantities") == 17
    assert "account_id: Aggregations: COUNT DISTINCT | Inappropriate aggregations: SUM · MEAN" in rendered
    for code in ("high_missingness", "non_unique_identifier"):
        warning = next(w for w in context.warnings if w.code == code)
        assert warning.message in rendered

    guidance = next(line for line in rendered.splitlines() if "Aggregations: COUNT DISTINCT" in line)
    for budget in (100, 501, 1500, 2500):
        compact = context.to_context(max_chars=budget)
        assert len(compact) <= budget
        assert compact.endswith("[Context truncated.]")
        if "Aggregations: COUNT DISTINCT" in compact:
            assert guidance in compact.splitlines()
        if "Aggregations: SUM · MEAN" in compact:
            assert all("SUM only for additive quantities" in line for line in compact.splitlines()
                       if "Aggregations: SUM · MEAN" in line)


def test_aggregation_formatter_preserves_names_as_complete_values():
    assert _format_aggregations(["COUNT DISTINCT"]) == "COUNT DISTINCT"
    assert _format_aggregations(["SUM", "MEAN"]) == "SUM · MEAN"
    assert _format_aggregations([]) == "—"
    assert _format_aggregations(["SUM, MEAN", "COUNT DISTINCT"]) == "SUM, MEAN · COUNT DISTINCT"
    assert _format_aggregations("COUNT DISTINCT") == "COUNT DISTINCT"


def test_range_formatter_removes_unnecessary_time_and_decimal_parts():
    import pandas as pd

    assert _format_range_value(pd.Timestamp("2026-01-01 00:00:00")) == "2026-01-01"
    assert _format_range_value(pd.Timestamp("2026-01-01 12:30:00")) == "2026-01-01 12:30:00"
    assert _format_range_value(85.0) == "85"
    assert _format_range_value(85.5) == "85.5"


def test_compact_context_formats_date_and_whole_number_ranges():
    import pandas as pd

    context = inspect(pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "amount": [85.0, 210.0],
    }))
    rendered = context.to_context(include_values=True)
    assert "Range 2026-01-01 → 2026-01-02" in rendered
    assert "Range 85 → 210" in rendered
    safe_rendered = context.to_context()
    assert "Range 2026-01-01" not in safe_rendered
    assert "Range 85" not in safe_rendered


def test_result_mappings_cannot_be_mutated_through_the_snapshot(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    try:
        revenue.statistics.quantiles["0.5"] = -100
    except TypeError:
        pass
    else:
        raise AssertionError("quantiles should be read-only")
    assert context.to_dict()["columns"][2]["statistics"]["quantiles"]["0.5"] == 6.25


@pytest.mark.parametrize(
    "new_value,mutate",
    [
        pytest.param(lambda: ["a"], lambda value: value.append("later"), id="list"),
        pytest.param(lambda: {"label": "a"}, lambda value: value.update(label="later"), id="dictionary"),
        pytest.param(
            lambda: {"labels": ["a"]},
            lambda value: value["labels"].append("later"),
            id="nested-list",
        ),
    ],
)
def test_inspect_snapshots_mutable_top_values(new_value, mutate):
    import pandas as pd

    frame = pd.DataFrame({"value": [new_value(), new_value()], "region": ["East", "East"]})
    context = inspect(frame)
    expected = new_value()
    statistics = context.columns[0].statistics
    assert statistics.top_values == ((expected, 2),)
    before = context.to_dict(include_values=True)
    assert before["columns"][0]["statistics"]["top_values"] == [[expected, 2]]
    json_before = context.to_json(include_values=True)
    markdown_before = context.to_markdown(include_values=True)
    compact_before = context.to_context(include_values=True)

    mutate(frame.at[0, "value"])

    assert statistics.top_values == ((expected, 2),)
    assert context.columns[1].statistics.top_values == (("East", 2),)
    assert context.to_dict(include_values=True) == before
    assert context.to_json(include_values=True) == json_before
    assert context.to_markdown(include_values=True) == markdown_before
    assert context.to_context(include_values=True) == compact_before


def test_snapshot_detaches_mutable_values_inside_tuple_samples():
    from oribi.profiling.columns import _snapshot_top_value

    inner = [10]
    snapshot = _snapshot_top_value((inner,), {})
    inner.append(20)
    # A tuple is immutable but can carry a mutable list; the snapshot must hold
    # a detached copy, so the later edit does not reach into it.
    assert snapshot == ([10],)

    # A list reused inside the same tuple keeps its shared identity in the copy.
    shared = [0]
    reused = _snapshot_top_value((shared, shared), {})
    assert reused[0] == [0]
    assert reused[0] is reused[1]


def test_inspect_preserves_cycles_and_shared_items_in_a_snapshot():
    import pandas as pd

    shared = ["a"]
    value = {"first": shared, "second": shared}
    value["cycle"] = value
    context = inspect(pd.DataFrame({"value": [value]}))
    snapshot, count = context.columns[0].statistics.top_values[0]

    assert count == 1
    assert snapshot is not value
    assert snapshot["cycle"] is snapshot
    assert snapshot["first"] is snapshot["second"]
    assert snapshot["first"] is not shared


def test_opt_in_exports_raise_clear_error_for_circular_list_sample():
    import pandas as pd

    sample = []
    sample.append(sample)
    context = inspect(pd.DataFrame({"items": [sample]}))

    with pytest.raises(ValueError, match="circular reference") as dict_error:
        context.to_dict(include_values=True)
    assert "top_values" in str(dict_error.value)
    assert "include_values" in str(dict_error.value)
    with pytest.raises(ValueError, match="circular reference"):
        context.to_json(include_values=True)


def test_opt_in_exports_raise_clear_error_for_circular_dict_sample():
    import pandas as pd

    sample = {"label": "x"}
    sample["self"] = sample
    context = inspect(pd.DataFrame({"value": [sample]}))

    with pytest.raises(ValueError, match="circular reference"):
        context.to_dict(include_values=True)
    with pytest.raises(ValueError, match="circular reference"):
        context.to_json(include_values=True)


def test_opt_in_export_serializes_sample_reusing_the_same_child_twice():
    import pandas as pd

    shared = ["a"]
    sample = {"first": shared, "second": shared}
    context = inspect(pd.DataFrame({"value": [sample]}))

    exported = context.to_dict(include_values=True)
    assert exported["columns"][0]["statistics"]["top_values"] == [
        [{"first": ["a"], "second": ["a"]}, 1]
    ]
    assert json.loads(context.to_json(include_values=True)) == exported


def test_default_exports_succeed_for_circular_samples_and_omit_values():
    import pandas as pd

    sample = []
    sample.append(sample)
    context = inspect(pd.DataFrame({"items": [sample]}))

    exported = context.to_dict()
    assert exported["columns"][0]["statistics"]["top_values"] == []
    assert json.loads(context.to_json()) == exported


def test_context_limit_keeps_an_explicit_marker_for_long_labels():
    import pandas as pd

    context = inspect(pd.DataFrame({"x" * 1_000: [1, 2, 3]}))
    rendered = context.to_context(max_chars=100)
    assert len(rendered) <= 100
    assert "[Context truncated.]" in rendered


def test_markdown_prints_each_warning_message_once():
    import pandas as pd

    frame = pd.DataFrame({"amount": [1.0, -2.0, 3.0, 3.0]}).iloc[[0, 1, 2, 3, 3]]
    context = inspect(frame)
    markdown = context.to_markdown()
    column_warning = next(w for w in context.warnings if w.code == "negative_measure_values")
    dataset_warning = next(w for w in context.warnings if w.code == "duplicate_rows")
    assert markdown.count(column_warning.message) == 1
    assert markdown.count("negative values were observed") == 1
    assert markdown.count(dataset_warning.message) == 1
    column_section = markdown[markdown.index("### amount"):markdown.index("## Analytical warnings")]
    assert f"Warning [{column_warning.severity}]: {column_warning.title}" in column_section
    assert column_warning.message not in column_section
    warnings_section = markdown[markdown.index("## Analytical warnings"):]
    assert column_warning.message in warnings_section
    assert column_warning.implication in warnings_section
    assert dataset_warning.title in warnings_section


def test_markdown_warning_change_leaves_structured_output_alone():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [1.0, -2.0, 3.0]}))
    column = next(c for c in context.columns if c.name == "amount")
    assert [w.code for w in column.warnings] == ["negative_measure_values"]
    assert "negative_measure_values" in [w.code for w in context.warnings]
    exported = context.to_dict()
    assert any(w["code"] == "negative_measure_values" for w in exported["warnings"])
    assert json.loads(context.to_json()) == exported


def test_markdown_renders_column_names_as_literal_text():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "a*b": [-1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0],
            "`code`": list("abababab"),
            "line\nbreak": range(8),
            "# head": [1] * 8,
            "order_id": list("aabbccdd"),
        }
    )
    markdown = inspect(frame).to_markdown()

    # Markdown characters are escaped and a newline never splits a line.
    assert r"### a\*b" in markdown
    assert r"### \`code\`" in markdown
    assert r"### \# head" in markdown
    assert "### line break" in markdown
    assert r"- **a\*b**: numeric measure" in markdown
    assert r"Negative values in a\*b" in markdown
    assert "a*b" not in markdown.replace(r"a\*b", "")
    assert "line\nbreak" not in markdown
    # Underscores inside a word are left readable.
    assert "### order_id" in markdown
    assert r"order\_id" not in markdown

def test_compact_context_keeps_multiline_column_labels_on_one_line():
    name = "Delivery\nWarehouse"
    frame = pd.DataFrame({name: range(8), "region": list("aabbccdd")})
    context = inspect(frame)
    for budget in (8000, 200):
        report = context.to_context(max_chars=budget)
        assert "Delivery\nWarehouse" not in report
        assert "Delivery Warehouse" in report
        assert "region" in report
        assert len(report) <= budget
    assert context.get_column(name).name == name
    exported = context.to_dict()
    column_names = [column["name"] for column in exported["columns"]]
    assert name in column_names
    assert json.loads(context.to_json()) == exported


@pytest.mark.parametrize("tz", [None, "UTC"])
def test_compact_context_preserves_nanosecond_ranges_near_midnight(tz):
    stamps = pd.date_range("2026-01-01 00:00:00.000000001", periods=2, freq="ns", tz=tz)
    context = inspect(pd.DataFrame({"stamp": stamps}))
    offset = "+00:00" if tz else ""
    bounds = (
        f"Range 2026-01-01 00:00:00.000000001{offset} → "
        f"2026-01-01 00:00:00.000000002{offset}"
    )

    assert bounds in context.to_context(include_values=True)
    assert "Range " not in context.to_context()


def test_compact_context_preserves_timezone_offset_on_midnight_ranges():
    """Timezone-aware midnight bounds keep time and offset; naive still date-only."""
    from datetime import datetime, timedelta, timezone

    import pandas as pd

    offset = timezone(timedelta(hours=5, minutes=30))  # +05:30
    aware = [
        datetime(2026, 1, 1, 0, 0, tzinfo=offset),
        datetime(2026, 1, 2, 0, 0, tzinfo=offset),
        datetime(2026, 1, 3, 0, 0, tzinfo=offset),
    ]
    naive = [
        datetime(2026, 1, 1, 0, 0),
        datetime(2026, 1, 2, 0, 0),
        datetime(2026, 1, 3, 0, 0),
    ]

    aware_df = pd.DataFrame({"event": aware})
    naive_df = pd.DataFrame({"event": naive})

    aware_rendered = inspect(aware_df).to_context(include_values=True)
    naive_rendered = inspect(naive_df).to_context(include_values=True)

    # Aware midnight range must retain times and offsets
    assert "Range 2026-01-01 00:00:00+05:30 → 2026-01-03 00:00:00+05:30" in aware_rendered
    # Naive midnight still uses date-only
    assert "Range 2026-01-01 → 2026-01-03" in naive_rendered
    # Default compact output omits exact range
    assert "Range 2026-01-01" not in inspect(aware_df).to_context()
    assert "Range 2026-01-01" not in inspect(naive_df).to_context()


def test_categorical_serialization_respects_include_values():
    frame = pd.DataFrame({
        "tier": ["free", "pro", "free", "pro"],
        "plan": ["a", "b", "c", "a"],
    })
    context = inspect(frame, options=ProfileOptions(top_k=2))

    default_dict = context.to_dict()
    cols_default = {c["name"]: c for c in default_dict["columns"]}
    assert cols_default["tier"]["categories"] == {"values": [], "is_complete": True}
    assert cols_default["plan"]["categories"] == {"values": [], "is_complete": False}

    default_md = context.to_markdown()
    assert "Categories (" not in default_md
    assert "Top categories (" not in default_md
    assert "free" not in default_md

    default_ctx = context.to_context()
    assert "Categories (" not in default_ctx
    assert "Top categories (" not in default_ctx
    assert "free" not in default_ctx

    opt_in_dict = context.to_dict(include_values=True)
    cols_opt_in = {c["name"]: c for c in opt_in_dict["columns"]}
    assert cols_opt_in["tier"]["categories"] == {
        "values": [["free", 2], ["pro", 2]],
        "is_complete": True,
    }

    opt_in_md = context.to_markdown(include_values=True)
    assert '- Categories (all 2): "free" (2), "pro" (2)' in opt_in_md
    assert "- Top categories (top 2 of 3):" in opt_in_md

    opt_in_ctx = context.to_context(include_values=True)
    assert 'Categories (all 2): "free" (2), "pro" (2)' in opt_in_ctx
    assert "Top categories (top 2 of 3):" in opt_in_ctx


def test_redaction_applies_to_categorical_context_subclasses():
    class CustomCategories(CategoricalContext):
        pass

    context = inspect(pd.DataFrame({"tier": ["hidden-plan-a", "hidden-plan-b"] * 2}))
    column = context.columns[0]
    custom = CustomCategories(column.categories.values, column.categories.is_complete)
    context = replace(context, columns=(replace(column, categories=custom),))

    assert context.to_dict()["columns"][0]["categories"]["values"] == []
    assert "hidden-plan-a" not in context.to_json()
    assert "hidden-plan-a" in context.to_json(include_values=True)


def test_text_category_values_distinguish_types_and_render_once():
    labels = pd.Categorical([1, "1", " a, b ", "north\nwest"] * 2)
    context = inspect(pd.DataFrame({"category": labels}))
    for rendered in (context.to_markdown(include_values=True), context.to_context(include_values=True)):
        categories = next(line for line in rendered.splitlines() if "Categories (all 4):" in line)
        assert ': 1 (2), " a, b " (2), "1" (2)' in categories
        assert "north" in categories and "west" in categories
        assert "Local value samples:" not in rendered


def test_compact_context_omits_oversized_local_samples_before_truncating():
    ids = [f"id-{i}-" + "x" * 3000 for i in range(10)]
    context = inspect(pd.DataFrame({"customer_id": ids}))

    default_ctx = context.to_context(max_chars=1600)
    opt_in_ctx = context.to_context(max_chars=1600, include_values=True)

    assert len(default_ctx) <= 1600
    assert len(opt_in_ctx) <= 1600
    assert "[Context truncated.]" not in default_ctx
    assert "[Context truncated.]" not in opt_in_ctx
    assert "Local value samples:" not in default_ctx
    assert "Local value samples:" not in opt_in_ctx
    assert opt_in_ctx == default_ctx
    assert "RELATIONSHIPS" in opt_in_ctx
    assert "CARDINALITY" in opt_in_ctx

    over_budget = context.to_context(max_chars=250, include_values=True)
    assert len(over_budget) <= 250
    assert over_budget.endswith("[Context truncated.]")


def test_compact_context_includes_short_local_samples_when_budget_allows():
    ids = [f"id-{i}" for i in range(5)]
    context = inspect(pd.DataFrame({"customer_id": ids}))

    default_ctx = context.to_context(max_chars=1600)
    assert "Local value samples:" not in default_ctx

    with_samples = context.to_context(max_chars=1600, include_values=True)
    assert "[Context truncated.]" not in with_samples
    assert "Local value samples: " in with_samples
    assert "id-0 (1)" in with_samples

    exact_fit = context.to_context(max_chars=len(with_samples), include_values=True)
    assert exact_fit == with_samples
    assert len(exact_fit) <= len(with_samples)

    one_char_short = context.to_context(max_chars=len(with_samples) - 1, include_values=True)
    assert len(one_char_short) <= len(with_samples) - 1
    assert "[Context truncated.]" not in one_char_short
    assert "Local value samples:" not in one_char_short
    assert one_char_short == default_ctx


def test_compact_context_omits_oversized_sample_while_keeping_short_sample():
    context = inspect(
        pd.DataFrame(
            {
                "long_id": [f"long-{i}-" + "x" * 2000 for i in range(5)],
                "short_id": [f"s-{i}" for i in range(5)],
            }
        )
    )

    rendered = context.to_context(max_chars=2400, include_values=True)
    assert len(rendered) <= 2400
    assert "[Context truncated.]" not in rendered
    assert "long-0-" not in rendered
    assert "Local value samples: s-0 (1)" in rendered


@pytest.mark.parametrize(
    "invalid_max_chars",
    [
        "100",
        "200",
        True,
        False,
        np.bool_(True),
        100.5,
        100.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        None,
    ],
)
def test_to_context_rejects_non_integer_max_chars(invalid_max_chars):
    context = inspect(pd.DataFrame({"region": ["east", "west", "east"]}))
    with pytest.raises(ValueError, match="max_chars must be an integer"):
        context.to_context(max_chars=invalid_max_chars)


@pytest.mark.parametrize("too_small", [99, 0, -1, np.int64(99)])
def test_to_context_rejects_max_chars_below_minimum(too_small):
    context = inspect(pd.DataFrame({"region": ["east", "west", "east"]}))
    with pytest.raises(ValueError, match="max_chars must be at least 100"):
        context.to_context(max_chars=too_small)


def test_to_context_accepts_minimum_and_numpy_integer_budgets():
    context = inspect(pd.DataFrame({"region": ["east", "west", "east"]}))

    at_minimum = context.to_context(max_chars=100)
    assert len(at_minimum) <= 100
    assert "DATASET" in at_minimum

    for numpy_budget, python_budget in (
        (np.int64(100), 100),
        (np.uint8(100), 100),
        (np.int32(500), 500),
        (np.int64(8000), 8000),
    ):
        assert context.to_context(max_chars=numpy_budget) == context.to_context(max_chars=python_budget)

def test_markdown_renders_empty_column_names_explicitly():
    context = inspect(pd.DataFrame({"": [1, 2, 3]}))
    markdown = context.to_markdown()

    # Empty names should have a readable Markdown-only representation.
    assert "### (empty)" in markdown
    assert "- **(empty)**: categorical dimension" in markdown

    # Original column names must remain unchanged.
    assert context.columns[0].name == ""
    assert context.get_column("") is not None

    # Structured exports must preserve the original empty name.
    exported = context.to_dict()
    assert exported["columns"][0]["name"] == ""
    assert json.loads(context.to_json())["columns"][0]["name"] == ""


def test_markdown_renders_empty_column_names_in_grain():
    frame = pd.DataFrame({
        "": ["A", "B", "C"],
        "region": ["East", "West", "North"],
    })
    context = inspect(frame)

    estimated_grain = replace(
        context.grain,
        status="estimated",
        columns=("",),
        confidence=0.9,
    )
    context = replace(context, grain=estimated_grain)

    markdown = context.to_markdown()

    assert "**(empty)** · candidate observation grain" in markdown


def test_markdown_renders_empty_column_names_in_relationships():
    frame = pd.DataFrame({
        "": ["A", "A", "B", "B"],
        "region": ["East", "East", "West", "West"],
        "amount": [10, 20, 30, 40],
    })
    context = inspect(frame)

    markdown = context.to_markdown()

    assert "- **(empty)** groups observations" in markdown
    assert "- **(empty) → region: one_to_one**" in markdown


def _frame_with_two_temporal_gaps(extra=None):
    import pandas as pd

    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    observed = dates[[i for i in range(12) if i not in (3, 7)]]
    data = {"observed_at": observed}
    if extra:
        data.update(extra)
    return pd.DataFrame(data)


def test_compact_context_keeps_separate_temporal_gaps_visible():
    context = inspect(_frame_with_two_temporal_gaps())

    assert len([w for w in context.warnings if w.code == "potential_temporal_gap"]) == 2
    assert context.to_context().count("has a possible temporal gap") == 2


def test_compact_context_still_shows_an_ordinary_warning_once():
    amount = [-1.0, 2, 3, -4, 5, 6, 7, 8, 9, 10]
    context = inspect(_frame_with_two_temporal_gaps({"amount": amount}))
    warning = next(w for w in context.warnings if w.code == "negative_measure_values")

    assert context.to_context().count(warning.message) == 1


def test_compact_context_does_not_expose_temporal_gap_dates():
    rendered = inspect(_frame_with_two_temporal_gaps()).to_context()

    assert "2026-01-04" not in rendered
    assert "2026-01-08" not in rendered


def test_truncated_context_keeps_separate_temporal_gaps_visible():
    context = inspect(_frame_with_two_temporal_gaps())

    rendered = context.to_context(max_chars=700)

    assert len(rendered) <= 700
    assert rendered.count("has a possible temporal gap") == 2


def test_summary_lists_each_separate_temporal_gap():
    summary = inspect(_frame_with_two_temporal_gaps()).summary()

    assert "2 analytical warning(s) detected." in summary
    assert summary.count("has a possible temporal gap") == 2


def _strict_json_loads(payload: str):
    def reject_constant(constant: str):
        raise AssertionError(f"Nonstandard JSON constant emitted: {constant}")

    return json.loads(payload, parse_constant=reject_constant)


def test_structured_exports_handle_numpy_longdouble_index_name():
    index = pd.Index(["a", "b", "c"], name=np.longdouble("1.25"))
    context = inspect(pd.DataFrame({"amount": [1, 2, 3]}, index=index))

    exported = context.to_dict()
    encoded = context.to_json()

    assert exported["profile"]["index"]["names"] == [1.25]
    assert type(exported["profile"]["index"]["names"][0]) is float
    assert _strict_json_loads(encoded) == exported


@pytest.mark.parametrize(
    "raw_value",
    [
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("-inf"),
    ],
)
def test_structured_exports_normalize_non_finite_longdouble_to_none(raw_value):
    index = pd.Index(["a", "b", "c"], name=raw_value)
    context = inspect(pd.DataFrame({"amount": [1, 2, 3]}, index=index))

    exported = context.to_dict()
    encoded = context.to_json()

    assert exported["profile"]["index"]["names"] == [None]
    assert _strict_json_loads(encoded) == exported


def test_normalize_maps_out_of_range_extended_floats_to_none():
    class OverflowingExtendedFloat(Real):
        def item(self):
            return self

        def __float__(self):
            raise OverflowError("extended float outside float64 range")

        def __abs__(self): return self
        def __add__(self, other): return NotImplemented
        def __ceil__(self): return 0
        def __eq__(self, other): return self is other
        def __floor__(self): return 0
        def __floordiv__(self, other): return NotImplemented
        def __le__(self, other): return NotImplemented
        def __lt__(self, other): return NotImplemented
        def __mod__(self, other): return NotImplemented
        def __mul__(self, other): return NotImplemented
        def __neg__(self): return self
        def __pos__(self): return self
        def __pow__(self, other): return NotImplemented
        def __radd__(self, other): return NotImplemented
        def __rfloordiv__(self, other): return NotImplemented
        def __rmod__(self, other): return NotImplemented
        def __rmul__(self, other): return NotImplemented
        def __round__(self, ndigits=None): return 0
        def __rpow__(self, other): return NotImplemented
        def __rtruediv__(self, other): return NotImplemented
        def __truediv__(self, other): return NotImplemented
        def __trunc__(self): return 0

    assert _normalize(OverflowingExtendedFloat()) is None
    if np.finfo(np.longdouble).maxexp > 1024:
        assert _normalize(np.longdouble("1e400")) is None
        assert _normalize(np.longdouble("-1e400")) is None


def test_normalize_preserves_standard_numpy_scalars_and_handles_clongdouble():
    assert _normalize(np.int64(7)) == 7
    assert type(_normalize(np.int64(7))) is int
    assert _normalize(np.uint8(3)) == 3
    assert type(_normalize(np.uint8(3))) is int
    assert _normalize(np.float32(1.5)) == 1.5
    assert type(_normalize(np.float32(1.5))) is float
    assert _normalize(np.float64(2.5)) == 2.5
    assert type(_normalize(np.float64(2.5))) is float
    assert _normalize(np.float64("nan")) is None
    assert _normalize(np.float64("inf")) is None
    assert _normalize(np.bool_(True)) is True
    assert type(_normalize(np.bool_(True))) is bool
    assert _normalize(np.clongdouble(1 + 2j)) == str(np.clongdouble(1 + 2j))