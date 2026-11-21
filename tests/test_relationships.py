from unittest.mock import patch

import pandas as pd

from oribi import CardinalityContext, ProfileOptions, inspect
from oribi.profiling import relationships


def _column(context, name):
    return next(column for column in context.columns if column.name == name)


def test_column_cardinality_reports_row_and_value_frequencies():
    context = inspect(pd.DataFrame({"customer_id": ["a", "a", "a", "b", "b", "c"]}))
    card = _column(context, "customer_id").cardinality
    assert isinstance(card, CardinalityContext)
    assert (card.row_count, card.observed_count, card.unique_count) == (6, 6, 3)
    assert card.uniqueness_ratio == 0.5
    assert card.average_records_per_value == 2
    assert (card.minimum_records_per_value, card.maximum_records_per_value) == (1, 3)


def test_one_to_many_uses_observed_group_and_unique_entity_values():
    frame = pd.DataFrame({
        "customer_id": [f"c{i // 4}" for i in range(100)],
        "order_id": [f"o{i}" for i in range(100)],
    })
    context = inspect(frame)
    relation = next(
        rel for rel in context.relationships
        if rel.kind == "one_to_many" and rel.source_column == "customer_id"
    )
    assert relation.target_column == "order_id"
    assert relation.coverage == 1
    assert relation.confidence >= 0.9
    assert relation.evidence[0].details["average_entities_per_group"] == 4
    assert relation.evidence[0].details["minimum_entities_per_group"] == 4
    assert relation.evidence[0].details["maximum_entities_per_group"] == 4


def test_relationship_scan_runs_below_and_at_limit_but_is_skipped_above():
    frame = pd.DataFrame({
        "customer_id": ["c0", "c0", "c1", "c1", "c2"],
        "order_id": [f"o{i}" for i in range(5)],
    })

    below = inspect(frame.iloc[:3], options=ProfileOptions(max_exact_rows=4))
    at_limit = inspect(frame.iloc[:4], options=ProfileOptions(max_exact_rows=4))
    above = inspect(frame, options=ProfileOptions(max_exact_rows=4))

    assert any(relation.kind == "one_to_many" for relation in below.relationships)
    assert any(relation.kind == "one_to_many" for relation in at_limit.relationships)
    assert above.relationships == ()
    assert _column(above, "order_id").cardinality.unique_count == 5


def test_context_explains_grouping_key_and_sql_relevant_cardinality():
    frame = pd.DataFrame({
        "customer_id": [f"c{i // 4}" for i in range(100)],
        "order_id": [f"o{i}" for i in range(100)],
        "revenue": [float(i + 1) for i in range(100)],
        "order_date": pd.date_range("2026-01-01", periods=100),
    })
    rendered = inspect(frame).to_context()
    assert "RELATIONSHIPS" in rendered
    assert "customer_id groups observations" in rendered
    assert "maps observations to order_id" in rendered
    assert "about 4.0 related observations per group" in rendered
    assert "CARDINALITY" in rendered
    assert "customer_id: 25 unique / 100 rows" in rendered
    assert "COLUMN ROLES" in rendered
    assert "order_date: temporal dimension" in rendered
    assert "revenue is a quantitative measure at order_id" in rendered


def test_cardinality_excludes_missing_values_but_keeps_total_rows():
    context = inspect(pd.DataFrame({"account_id": ["a", "a", None, "b", None]}))
    card = _column(context, "account_id").cardinality
    assert (card.row_count, card.observed_count, card.unique_count) == (5, 3, 2)
    assert (card.missing_count, card.missing_rate) == (2, 0.4)
    assert card.uniqueness_ratio == 2 / 3
    assert card.average_records_per_value == 1.5
    assert (card.minimum_records_per_value, card.maximum_records_per_value) == (1, 2)


def test_empty_and_single_row_cardinality_are_defined():
    empty = inspect(pd.DataFrame({"id": pd.Series(dtype="int64")}))
    empty_card = _column(empty, "id").cardinality
    assert empty_card.row_count == 0
    assert empty_card.observed_count == empty_card.unique_count == 0
    assert empty_card.uniqueness_ratio is None
    assert empty_card.average_records_per_value is None

    single = inspect(pd.DataFrame({"id": [42]}))
    card = _column(single, "id").cardinality
    assert card.uniqueness_ratio == 1
    assert card.average_records_per_value == 1
    assert (card.minimum_records_per_value, card.maximum_records_per_value) == (1, 1)


def test_duplicate_numeric_ids_are_grouped_by_value():
    context = inspect(pd.DataFrame({"account_id": pd.Series([10, 10, 10, 20], dtype="int64")}))
    card = _column(context, "account_id").cardinality
    assert card.unique_count == 2
    assert card.uniqueness_ratio == 0.5
    assert (card.minimum_records_per_value, card.maximum_records_per_value) == (1, 3)


def test_string_identifiers_can_form_observed_groups():
    frame = pd.DataFrame({
        "team_id": [f"t{i // 5}" for i in range(50)],
        "member_id": [f"m{i}" for i in range(50)],
    })
    context = inspect(frame)
    assert any(
        rel.kind == "one_to_many" and rel.source_column == "team_id" and rel.target_column == "member_id"
        for rel in context.relationships
    )


def test_bijective_value_mapping_can_show_one_to_one_with_uncertainty():
    frame = pd.DataFrame({"K1": [1, 1, 2, 2], "K2": ["a", "a", "b", "b"]})
    context = inspect(frame)
    relation = next(rel for rel in context.relationships if rel.kind == "one_to_one")
    assert {relation.source_column, relation.target_column} == {"K1", "K2"}
    assert relation.confidence >= 0.75
    assert relation.semantic_confidence == relation.confidence
    assert relation.confidence_components["mapping_coverage"] == 1.0
    assert len([rel for rel in context.relationships if rel.kind == "one_to_one"]) == 1


def test_missing_entity_value_cannot_establish_exact_one_to_many():
    frame = pd.DataFrame({
        "customer_id": [f"c{i // 4}" for i in range(100)],
        "order_id": [None if i == 99 else f"o{i}" for i in range(100)],
    })
    context = inspect(frame)
    key = next(key for key in context.candidate_keys if key.columns == ("order_id",))
    assert not key.verified_unique
    assert not any(rel.kind == "one_to_many" and rel.target_column == "order_id" for rel in context.relationships)
    assert _column(context, "order_id").cardinality.observed_count == 99


def test_complete_entity_key_establishes_exact_one_to_many():
    frame = pd.DataFrame({
        "customer_id": [f"c{i // 4}" for i in range(100)],
        "order_id": [f"o{i}" for i in range(100)],
    })
    context = inspect(frame)
    key = next(key for key in context.candidate_keys if key.columns == ("order_id",))
    assert key.verified_unique
    assert any(rel.kind == "one_to_many" and rel.source_column == "customer_id"
               and rel.target_column == "order_id" for rel in context.relationships)


def test_unique_id_names_do_not_create_foreign_key_claims_without_group_evidence():
    frame = pd.DataFrame({
        "customer_id": [f"c{i}" for i in range(30)],
        "order_id": [f"o{i}" for i in range(30)],
    })
    context = inspect(frame)
    assert not [rel for rel in context.relationships if rel.kind in {"one_to_many", "one_to_one"}]
    assert "groups multiple" not in context.to_context()


def test_identifier_name_hint_requires_observed_value_variation():
    context = inspect(pd.DataFrame({"field_id": ["same"] * 20}))
    column = _column(context, "field_id")
    assert column.semantic.role != "identifier"
    assert not any(rel.kind == "identifies_observations" for rel in column.relationships)


def test_small_sample_mapping_keeps_strong_observed_evidence():
    frame = pd.DataFrame({
        "customer_id": ["a", "a", "b", "b"],
        "order_id": ["o1", "o2", "o3", "o4"],
    })
    relation = next(rel for rel in inspect(frame).relationships if rel.kind == "one_to_many")
    assert relation.confidence >= 0.8
    assert relation.confidence_components["mapping_coverage"] == 1.0
    assert relation.confidence_components["generalization"] < 1.0


def test_cardinality_and_relationships_are_serialized_and_rendered_in_markdown():
    frame = pd.DataFrame({
        "customer_id": [f"c{i // 4}" for i in range(100)],
        "order_id": [f"o{i}" for i in range(100)],
    })
    context = inspect(frame)
    data = context.to_dict()
    customer = next(column for column in data["columns"] if column["name"] == "customer_id")
    assert customer["cardinality"]["unique_count"] == 25
    assert any(rel["kind"] == "one_to_many" for rel in data["relationships"])
    markdown = context.to_markdown()
    assert "## Relationships" in markdown
    assert "## Columns" in markdown
    assert "Cardinality: 25 unique across 100 observed values" in markdown


def test_arbitrary_columns_infer_event_grain_and_keep_roles_separate():
    frame = pd.DataFrame({
        "A": [1, 1, 2, 2],
        "B": ["X", "Y", "X", "X"],
        "C": [10, 15, 20, 18],
        "D": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"]),
    })
    context = inspect(frame)
    assert context.grain.columns == ("A", "D")
    assert len(context.candidate_keys) > 1
    assert {relation.kind for relation in _column(context, "A").relationships} >= {
        "jointly_identifies_observations", "groups_observations",
    }
    assert {relation.kind for relation in _column(context, "D").relationships} >= {
        "jointly_identifies_observations", "temporal_attribute",
    }
    rendered = context.to_context()
    assert "COLUMN ROLES" in rendered
    assert "RELATIONSHIPS" in rendered
    assert "A + D jointly identify an observation" in rendered
    assert "D is a temporal attribute associated with observations" in rendered
    assert "A + D jointly identify" in rendered


def test_entity_level_data_with_arbitrary_names_has_unique_observation_key():
    context = inspect(pd.DataFrame({"K": [f"v{i}" for i in range(100)], "V": [i % 5 for i in range(100)]}))
    assert context.grain.columns == ("K",)
    assert any(rel.kind == "identifies_observations" for rel in _column(context, "K").relationships)


def test_unique_temporal_column_can_describe_time_observation_grain():
    frame = pd.DataFrame({
        "T": pd.date_range("2025-01-01", periods=40, freq="h"),
        "M": [i % 7 for i in range(40)],
    })
    context = inspect(frame)
    assert "T" in context.grain.columns
    assert any(rel.kind == "temporal_attribute" for rel in _column(context, "T").relationships)

    temporal_only = inspect(pd.DataFrame({"T": pd.date_range("2025-01-01", periods=40, freq="h")}))
    assert temporal_only.grain.columns == ("T",)
    assert temporal_only.grain.evidence[0].code == "unique_temporal_observations"


def test_repeated_entity_and_time_jointly_form_a_composite_grain():
    frame = pd.DataFrame({
        "E": [1, 1, 1, 2, 2, 2],
        "T": pd.to_datetime([
            "2025-01-01", "2025-01-02", "2025-01-03",
            "2025-01-01", "2025-01-02", "2025-01-03",
        ]),
        "M": [2.1, 2.4, 2.6, 3.1, 3.2, 3.5],
    })
    context = inspect(frame)
    assert context.grain.columns == ("E", "T")
    assert any(rel.kind == "jointly_identifies_observations" for rel in _column(context, "E").relationships)
    assert any(rel.kind == "jointly_identifies_observations" for rel in _column(context, "T").relationships)
    assert context.to_context().count("jointly identify an observation") == 1


def test_multiple_temporal_columns_are_each_related_to_observations():
    frame = pd.DataFrame({
        "T1": pd.date_range("2025-01-01", periods=8),
        "T2": pd.date_range("2025-02-01", periods=8),
        "G": ["a", "b"] * 4,
    })
    context = inspect(frame)
    for name in ("T1", "T2"):
        assert any(rel.kind == "temporal_attribute" for rel in _column(context, name).relationships)


def test_arbitrary_reused_categorical_column_is_a_grouping_dimension():
    context = inspect(pd.DataFrame({"Q": ["north", "south", "north", "north", "south"]}))
    assert [rel.kind for rel in _column(context, "Q").relationships].count("groups_observations") == 1


def test_unique_determinants_do_not_create_trivial_functional_dependencies():
    frame = pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "T": pd.date_range("2025-01-01", periods=4),
        "M": [2.0, 4.0, 6.0, 8.0],
    })
    context = inspect(frame)
    assert not [
        rel for rel in context.relationships
        if rel.kind == "functional_dependency" and rel.source_column in {"record_id", "T"}
    ]


def test_multiple_unique_columns_do_not_create_trivial_one_to_one_links():
    frame = pd.DataFrame({"U1": [1, 2, 3, 4], "U2": [10, 20, 30, 40]})
    context = inspect(frame)
    assert not [rel for rel in context.relationships if rel.kind == "one_to_one"]
    assert not [rel for rel in context.relationships if rel.kind == "functional_dependency"]


def test_small_sample_shows_uniqueness_evidence_separately_from_confidence():
    context = inspect(pd.DataFrame({"record_key": ["a", "b", "c", "d"]}))
    relation = next(rel for rel in _column(context, "record_key").relationships if rel.kind == "identifies_observations")
    assert relation.evidence_strength == 1.0
    assert relation.confidence >= 0.85
    assert relation.semantic_confidence == relation.confidence
    assert relation.confidence_components["uniqueness"] == 1.0
    assert relation.confidence_components["coverage"] == 1.0
    rendered = context.to_context()
    assert "4/4 values distinct" in rendered
    assert "100% uniqueness evidence" in rendered
    assert f"{relation.confidence:.0%} confidence" in rendered


def test_identifier_confidence_tracks_uniqueness_and_sample_support_monotonically():
    full = inspect(pd.DataFrame({"K": [f"k{i}" for i in range(10)]}))
    mostly = inspect(pd.DataFrame({"K": [f"k{i}" for i in range(8)] + ["k0", "k1"]}))
    larger = inspect(pd.DataFrame({"K": [f"k{i}" for i in range(40)]}))
    sparse = inspect(pd.DataFrame({"K": [f"k{i}" for i in range(5)] + [None] * 5}))

    def key_confidence(context):
        return next(
            relation.confidence for relation in _column(context, "K").relationships
            if relation.kind == "identifies_observations"
        )

    assert key_confidence(full) > key_confidence(mostly)
    assert key_confidence(larger) > key_confidence(full)
    assert key_confidence(full) > key_confidence(sparse)
    assert key_confidence(full) >= 0.85


def test_temporal_confidence_uses_dtype_and_coverage_not_uniqueness():
    complete = inspect(pd.DataFrame({"T": pd.date_range("2026-01-01", periods=10)}))
    repeated = inspect(pd.DataFrame({"T": pd.date_range("2026-01-01", periods=5).repeat(2)}))
    partial = inspect(pd.DataFrame({"T": [*pd.date_range("2026-01-01", periods=5), *([pd.NaT] * 5)]}))

    def temporal_confidence(context):
        return next(
            relation.confidence for relation in _column(context, "T").relationships
            if relation.kind == "temporal_attribute"
        )

    assert temporal_confidence(complete) == temporal_confidence(repeated)
    assert temporal_confidence(complete) > temporal_confidence(partial)
    assert temporal_confidence(complete) >= 0.9
    assert temporal_confidence(partial) >= 0.75


def test_measure_confidence_tracks_coverage_but_not_uniqueness():
    unique = inspect(pd.DataFrame({"M": [float(i) for i in range(10)]}))
    repeated = inspect(pd.DataFrame({"M": [float(i % 3) for i in range(10)]}))
    half_missing = inspect(pd.DataFrame({"M": [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, None, None]}))

    def measure_relation(context):
        return next(
            relation for relation in _column(context, "M").relationships
            if relation.kind == "measure_at_grain"
        )

    assert measure_relation(unique).confidence == measure_relation(repeated).confidence
    assert measure_relation(unique).confidence > measure_relation(half_missing).confidence
    assert measure_relation(unique).confidence >= 0.85


def test_grouping_confidence_tracks_reuse_and_missingness_gradually():
    strongly_repeated = inspect(pd.DataFrame({"G": ["a", "a", "a", "b", "b", "b", "c", "c", "c", "c"]}))
    mostly_unique = inspect(pd.DataFrame({"G": [f"g{i}" for i in range(9)] + ["g0"]}))
    partial = inspect(pd.DataFrame({"G": ["a", "a", "b", "b", "c", None, None, None, None, None]}))

    def group_confidence(context):
        return next(
            relation.confidence for relation in _column(context, "G").relationships
            if relation.kind == "groups_observations"
        )

    assert group_confidence(strongly_repeated) > group_confidence(mostly_unique)
    assert group_confidence(strongly_repeated) > group_confidence(partial)


def test_confidence_components_explain_score_deterministically():
    context = inspect(pd.DataFrame({"T": pd.date_range("2026-01-01", periods=4)}))
    relation = next(
        relation for relation in _column(context, "T").relationships
        if relation.kind == "temporal_attribute"
    )
    components = relation.confidence_components
    assert components["datetime_type"] == 1.0
    assert components["datetime_type_weight"] == 0.55
    contribution_sum = sum(value for key, value in components.items() if key.endswith("_contribution"))
    assert round(contribution_sum, 2) == relation.semantic_confidence
    serialized_temporal = next(
        item for item in context.to_dict()["columns"][0]["relationships"]
        if item["kind"] == "temporal_attribute"
    )
    assert serialized_temporal["confidence_components"]["datetime_type"] == 1.0


def test_measure_is_not_attached_to_weak_small_sample_grain():
    context = inspect(pd.DataFrame({
        "K": ["a", "b", "c", "d"],
        "M": [1.0, 2.0, 3.0, 4.0],
    }))
    measure_relation = next(
        relation for relation in _column(context, "M").relationships
        if relation.kind == "measure_at_grain"
    )
    assert measure_relation.related_columns == ()
    rendered = context.to_context()
    assert "M is a quantitative measure at the observed record grain" in rendered
    assert "M is a quantitative measure at K" not in rendered


def test_supported_measure_grain_is_retained_for_larger_dataset():
    context = inspect(pd.DataFrame({
        "K": [f"k{i}" for i in range(100)],
        "M": [float(i % 13) for i in range(100)],
    }))
    measure_relation = next(
        relation for relation in _column(context, "M").relationships
        if relation.kind == "measure_at_grain"
    )
    assert measure_relation.related_columns == ("K",)
    assert "M is a quantitative measure at K" in context.to_context()


def test_grouping_dimension_has_one_relationship_description():
    context = inspect(pd.DataFrame({"Q": ["north", "south", "north", None]}))
    relationship_section = context.to_context().split("RELATIONSHIPS", 1)[1].split("CARDINALITY", 1)[0]
    assert relationship_section.count("Q groups observations") == 1
    assert "provides a reusable grouping dimension" not in relationship_section


def test_meaningful_functional_dependency_requires_reused_determinant_values():
    frame = pd.DataFrame({
        "A": ["x", "x", "y", "y", "z", "z"],
        "B": ["x1", "x1", "y1", "y1", "z1", "z1"],
        "M": [1, 2, 3, 4, 5, 6],
    })
    context = inspect(frame)
    dependency = next(
        rel for rel in context.relationships
        if rel.kind == "functional_dependency" and rel.source_column == "A" and rel.target_column == "B"
    )
    assert dependency.evidence[0].details["source_distinct"] == 3
    assert dependency.evidence[0].details["coverage"] == 1.0


def test_hierarchical_values_produce_only_observed_functional_dependencies():
    frame = pd.DataFrame({
        "L1": ["a", "a", "a", "b", "b", "b"],
        "L2": ["a1", "a1", "a1", "b1", "b1", "b1"],
        "L3": ["x", "y", "z", "w", "w", "v"],
    })
    context = inspect(frame)
    assert any(rel.source_column == "L1" and rel.target_column == "L2" and rel.kind == "functional_dependency" for rel in context.relationships)
    assert not any(rel.source_column == "L2" and rel.target_column == "L1" and rel.kind == "functional_dependency" for rel in context.relationships)


def test_repeated_and_mostly_unique_columns_have_distinct_relationships():
    frame = pd.DataFrame({
        "G": ["a"] * 50 + [f"g{i}" for i in range(50)],
        "K": [f"k{i}" for i in range(99)] + ["k0"],
    })
    context = inspect(frame)
    assert _column(context, "G").cardinality.uniqueness_ratio == 0.51
    assert _column(context, "K").cardinality.uniqueness_ratio == 0.99
    assert any(rel.kind == "identifies_observations" for rel in _column(context, "K").relationships)
    assert "mostly-unique observation-key candidate" in context.to_context()


def test_arbitrary_single_row_empty_and_unstructured_frames_remain_safe():
    one = inspect(pd.DataFrame({"A": [1], "B": [None]}))
    empty = inspect(pd.DataFrame({"A": pd.Series(dtype="object"), "B": pd.Series(dtype="float64")}))
    unstructured = inspect(pd.DataFrame({"A": [1, 1, 1], "B": ["x", "x", "x"]}))
    assert one.profile.row_count == 1
    assert empty.grain.status == "undetermined"
    assert unstructured.grain.status == "undetermined"


def test_mixed_type_values_and_wide_frames_do_not_break_relationship_analysis():
    mixed = inspect(pd.DataFrame({"A": [1, "1", None, 2], "B": [True, False, True, False]}))
    wide = inspect(pd.DataFrame({f"C{i}": [i % 3 for _ in range(20)] for i in range(30)}))
    assert mixed.profile.column_count == 2
    assert wide.profile.column_count == 30
    assert all(column.cardinality is not None for column in wide.columns)


def test_long_numeric_group_values_can_relate_to_unique_observations():
    frame = pd.DataFrame({
        "G": [index // 100 for index in range(5_000)],
        "E": [f"v{index}" for index in range(5_000)],
        "M": [float(index % 17) for index in range(5_000)],
    })
    context = inspect(frame)
    relation = next(
        relation for relation in context.relationships
        if relation.kind == "one_to_many" and relation.source_column == "G"
    )
    assert relation.target_column == "E"
    assert relation.evidence[0].details["average_entities_per_group"] == 100


def test_zero_relationship_column_limit_skips_pair_scan_preparation():
    frame = pd.DataFrame({
        "group": ["a", "a", "b", "b"],
        "item_id": [1, 2, 3, 4],
    })

    with patch.object(relationships, "_nunique", wraps=relationships._nunique) as count:
        zero_limit = inspect(frame, options=ProfileOptions(max_relationship_columns=0))

    assert zero_limit.relationships == ()
    assert count.call_count == 0
    assert _column(zero_limit, "group").cardinality.unique_count == 2
    assert _column(zero_limit, "item_id").cardinality.unique_count == 4
    assert any(
        rel.kind == "groups_observations"
        for rel in _column(zero_limit, "group").relationships
    )
    assert any(
        rel.kind == "identifies_observations"
        for rel in _column(zero_limit, "item_id").relationships
    )

    positive_limit = inspect(frame, options=ProfileOptions(max_relationship_columns=2))
    assert any(
        rel.kind == "one_to_many" and rel.source_column == "group" and rel.target_column == "item_id"
        for rel in positive_limit.relationships
    )
