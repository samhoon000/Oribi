import pandas as pd
import pytest

from oribi import inspect


def test_recommendations_and_warnings(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    assert {r.operation for r in revenue.recommendations} >= {"SUM", "MEAN", "MEDIAN"}
    assert any(w.code == "negative_measure_values" for w in revenue.warnings)
    assert all(w.code and w.evidence is not None for w in context.warnings)


def test_detects_deterministic_value_mapping():
    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"], "amount": [1, 2, 3, 4]})
    context = inspect(frame)
    assert any(r.kind == "functional_dependency" for r in context.relationships)


def test_missing_unique_identifiers_warn_for_missingness_only():
    column = inspect(pd.DataFrame({"account_id": [1, 2, 3, 4, None]})).columns[0]
    assert column.semantic.role == "identifier"
    assert any(w.code == "high_missingness" for w in column.warnings)
    assert not any(w.code == "non_unique_identifier" for w in column.warnings)


def test_null_only_identifiers_do_not_warn_about_duplicates():
    column = inspect(pd.DataFrame({"account_id": [None, None, None]})).columns[0]
    assert any(w.code == "high_missingness" for w in column.warnings)
    assert not any(w.code == "non_unique_identifier" for w in column.warnings)


def test_observed_duplicate_identifiers_still_warn():
    column = inspect(pd.DataFrame({"account_id": [1, 2, 2, 3, None]})).columns[0]
    assert column.semantic.role == "identifier"
    assert any(w.code == "non_unique_identifier" for w in column.warnings)


def test_rate_abbreviations_are_non_additive():
    for name in ("CTR", "ROI", "CAGR"):
        column = inspect(pd.DataFrame({name: [0.02, 0.03, 0.01, 0.04]})).columns[0]
        by_operation = {r.operation: r.disposition for r in column.recommendations}
        assert by_operation["SUM"] == "not_applicable"
        assert by_operation["MEAN"] == "caution"
        assert any(e.code == "name_non_additive_hint" for e in (ev for r in column.recommendations for ev in r.evidence))


def test_rate_abbreviations_are_non_additive_only_as_whole_tokens():
    column = inspect(pd.DataFrame({"ctr_value": [0.02, 0.03, 0.01, 0.04]})).columns[0]
    by_operation = {r.operation: r.disposition for r in column.recommendations}
    assert by_operation["SUM"] == "not_applicable"

    column = inspect(pd.DataFrame({"contract": [0.02, 0.03, 0.01, 0.04]})).columns[0]
    by_operation = {r.operation: r.disposition for r in column.recommendations}
    assert by_operation["SUM"] == "recommended"


def test_additive_control_retains_sum_recommendation():
    column = inspect(pd.DataFrame({"cost": [2.5, 3.7, 4.5, 5.5]})).columns[0]
    by_operation = {r.operation: r.disposition for r in column.recommendations}
    assert by_operation["SUM"] == "recommended"
    assert by_operation["MEAN"] == "recommended"


def test_rate_abbreviations_omit_derived_totals():
    frame = pd.DataFrame({
        "CTR": [0.02, 0.03, 0.01, 0.04],
        "ROI": [0.15, 0.20, 0.10, 0.02],
        "CAGR": [0.03, 0.20, 0.30, 0.10],
        "cost": [2.5, 3.7, 4.5, 5.5],
    })
    metric_names = {metric.name for metric in inspect(frame).metrics}
    assert "total_cost" in metric_names
    assert not {"total_ctr", "total_roi", "total_cagr"} & metric_names


def test_rate_abbreviation_embedded_in_word_stays_additive():
    column = inspect(pd.DataFrame({"production_ctrvalue": [0.02, 0.03, 0.01, 0.04]})).columns[0]
    by_operation = {r.operation: r.disposition for r in column.recommendations}
    assert by_operation["SUM"] == "recommended"
    assert by_operation["MEAN"] == "recommended"


def test_identifier_recommends_count_distinct_and_flags_inappropriate_operations():
    context = inspect(pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"]}))
    col = context.columns[0]
    assert col.semantic.role == "identifier"

    cd_rec = next((r for r in col.recommendations if r.operation == "COUNT DISTINCT"), None)
    assert cd_rec is not None
    assert cd_rec.disposition == "recommended"
    assert "entity" in cd_rec.rationale.lower() or "entities" in cd_rec.rationale.lower()

    sum_rec = next((r for r in col.recommendations if r.operation == "SUM"), None)
    assert sum_rec is not None
    assert sum_rec.disposition == "not_applicable"
    assert "inappropriate" in sum_rec.rationale.lower()

    mean_rec = next((r for r in col.recommendations if r.operation == "MEAN"), None)
    assert mean_rec is not None
    assert mean_rec.disposition == "not_applicable"
    assert "inappropriate" in mean_rec.rationale.lower()


def test_repeated_identifiers_explain_row_counts_vs_entity_counts():
    context = inspect(pd.DataFrame({"order_id": [101, 101, 102, 102, 103]}))
    col = context.columns[0]
    assert col.semantic.role == "identifier"

    cd_rec = next(r for r in col.recommendations if r.operation == "COUNT DISTINCT")
    assert cd_rec.disposition == "recommended"
    assert "row count" in cd_rec.rationale.lower()
    assert "entity count" in cd_rec.rationale.lower() or "entities" in cd_rec.rationale.lower()

    warning = next(w for w in col.warnings if w.code == "non_unique_identifier")
    combined = f"{warning.implication} {warning.recommendation}".lower()
    assert "row count" in combined
    assert "entity count" in combined or "entities" in combined
    assert "count distinct" in combined


def test_duplicate_rows_warning_explains_measurement_difference():
    context = inspect(pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]}))
    warning = next(w for w in context.warnings if w.code == "duplicate_rows")
    combined = f"{warning.implication} {warning.recommendation}".lower()
    assert "row count" in combined
    assert "entity count" in combined
    # Does not claim that duplicate rows always represent invalid data
    assert "may be valid" in combined or "expected" in combined or "grain" in combined


def test_unique_numeric_measures_retain_existing_recommendations_and_starting_points():
    context = inspect(pd.DataFrame({"revenue": [10.5, 20.0, 35.5, 80.0]}))
    col = context.columns[0]
    assert col.semantic.role == "measure"
    assert {r.operation for r in col.recommendations} >= {"SUM", "MEAN", "MEDIAN", "MIN", "MAX"}

    rec_ops = {r.operation for r in col.recommendations if r.disposition == "recommended"}
    assert rec_ops == {"SUM", "MEAN"}

    sum_rec = next(r for r in col.recommendations if r.operation == "SUM")
    assert "starting point" in sum_rec.rationale.lower() or "additive" in sum_rec.rationale.lower()

    mean_rec = next(r for r in col.recommendations if r.operation == "MEAN")
    assert "starting point" in mean_rec.rationale.lower() or "typical" in mean_rec.rationale.lower()


def test_negative_and_skewed_measures_retain_cautions():
    context = inspect(pd.DataFrame({"metric": [1.0, 1.0, 1.0, 1.0, 100.0, -5.0]}))
    col = context.columns[0]
    assert col.semantic.role == "measure"

    skew_warning = next(w for w in col.warnings if w.code == "high_skew")
    assert "median" in skew_warning.recommendation.lower()

    neg_warning = next(w for w in col.warnings if w.code == "negative_measure_values")
    assert "domain" in neg_warning.recommendation.lower() or "meaning" in neg_warning.recommendation.lower()


@pytest.mark.parametrize(("sign", "tail"), [(1, "small values"), (-1, "large values")])
def test_high_skew_warning_describes_the_observed_tail(sign, tail):
    values = [sign * value for value in [-100.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]]
    context = inspect(pd.DataFrame({"amount": values}))

    warning = next(w for w in context.warnings if w.code == "high_skew")
    assert tail in warning.implication
    assert warning.severity == "low"
    assert "median" in warning.recommendation.lower()


def test_symmetric_measure_has_no_high_skew_warning():
    context = inspect(pd.DataFrame({"amount": [10.5, 20.5, 30.5, 40.5, 50.5]}))
    assert context.columns[0].semantic.role == "measure"
    assert all(w.code != "high_skew" for w in context.warnings)


def test_categorical_and_dimension_columns_do_not_receive_numeric_aggregations():
    frame = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "A", "C"]),
        "region": ["East", "West", "East", "West"],
    })
    context = inspect(frame)
    for col in context.columns:
        numeric_ops = {"SUM", "MEAN", "MEDIAN", "MIN", "MAX"}
        recommended_ops = {r.operation for r in col.recommendations if r.disposition in {"recommended", "caution", "useful", "valid"}}
        assert not (recommended_ops & numeric_ops)
        assert any("group" in r.operation.lower() or "group" in r.rationale.lower() for r in col.recommendations)


def test_unknown_or_ambiguous_columns_avoid_overconfident_recommendations():
    # A column named like a measure or identifier but with ambiguous / unstructured values
    frame = pd.DataFrame({"ambiguous_metric": ["alpha", "beta", "gamma"]})
    context = inspect(frame)
    col = context.columns[0]
    assert col.semantic.role == "unknown"
    # Semantic role is uncertain, so confident recommendations must be avoided
    assert not any(r.disposition == "recommended" for r in col.recommendations)
    assert not any(r.operation in {"SUM", "MEAN"} for r in col.recommendations)


def test_aggregation_guidance_in_markdown_and_context(sales_frame):
    context = inspect(sales_frame)
    markdown = context.to_markdown()
    compact = context.to_context()

    # Identifiers show recommended COUNT DISTINCT and inappropriate SUM, MEAN
    assert "- Recommended aggregations: COUNT DISTINCT" in markdown
    assert "- Inappropriate aggregations: SUM, MEAN" in markdown
    assert "Aggregations: COUNT DISTINCT" in compact
    assert "Inappropriate aggregations: SUM · MEAN" in compact

    # Measures show recommended SUM, MEAN, useful MEDIAN, and no inappropriate line
    revenue_section = compact.split("\nrevenue\n", 1)[1]
    assert "Aggregations: SUM · MEAN" in revenue_section
    assert "Useful operations: MEDIAN" in revenue_section
    assert "- Useful operations: MEDIAN" in markdown
    assert "Inappropriate aggregations" not in revenue_section

    # Dimensions show recommended GROUP BY
    region_section = compact.split("\nregion\n", 1)[1]
    assert "Aggregations: GROUP BY" in region_section

    # Warnings explain row count vs entity count
    assert "Row counts measure record occurrences rather than unique entities" in markdown


def test_useful_operations_rendered_in_markdown_and_context():
    frame = pd.DataFrame({
        "category": pd.Categorical(["A", "B", "A", "C"]),
    })
    context = inspect(frame)
    col = context.columns[0]
    assert col.semantic.role == "categorical"
    assert any(r.operation == "COUNT DISTINCT" and r.disposition == "useful" for r in col.recommendations)

    markdown = context.to_markdown()
    assert "- Recommended aggregations: GROUP BY" in markdown
    assert "- Useful operations: COUNT DISTINCT" in markdown

    compact = context.to_context()
    cat_section = compact.split("\ncategory\n", 1)[1]
    assert "Aggregations: GROUP BY" in cat_section
    assert "Useful operations: COUNT DISTINCT" in cat_section


def test_aggregation_guidance_preserves_privacy():
    frame = pd.DataFrame({
        "patient_id": [1, 1, 2, 2],
        "diagnosis": pd.Categorical(["HIV", "cancer", "HIV", "cancer"]),
        "charge": [100.0, 250.0, -10.0, 400.0],
    })
    context = inspect(frame)
    secrets = ("HIV", "cancer")

    for exported in (context.to_json(), context.to_markdown(), context.to_context()):
        for secret in secrets:
            assert secret not in exported


def test_grouping_advice_distinguishes_high_cardinality_text_and_categories():
    labels = [f"segment{i:03d}" for i in range(120)] * 3
    frame = pd.DataFrame({"text_segment": labels, "category_segment": pd.Categorical(labels)})
    context = inspect(frame)
    text, category = context.columns
    assert text.semantic.role == "dimension"
    assert any(w.code == "high_cardinality_text" for w in text.warnings)
    assert not any(rec.operation == "GROUP BY" for rec in text.recommendations)
    assert any(w.code == "high_cardinality_category" for w in category.warnings)
    assert next(rec for rec in category.recommendations if rec.operation == "GROUP BY").disposition == "caution"
    for rendered in (context.to_markdown(), context.to_context()):
        assert rendered.count("GROUP BY") == 1
        assert "group only if many small groups are useful" in rendered


@pytest.mark.parametrize("name", ["conversion_rate", "price_per_unit", "profit ratio", "gross_margin (%)"])
def test_non_additive_measures_render_their_restrictions(name):
    context = inspect(pd.DataFrame({name: [0.15, 0.20, 0.25, 0.40]}))
    ops = {rec.operation: rec for rec in context.columns[0].recommendations}
    assert ops["SUM"].disposition == "not_applicable"
    assert ops["MEAN"].disposition == "caution"
    for rendered in (context.to_markdown(), context.to_context(), context.to_context(max_chars=420)):
        assert "Inappropriate aggregations: SUM" in rendered
        assert "check weighting and denominators before averaging" in rendered


@pytest.mark.parametrize(
    "name", ["unit_price", "unit_cost", "unitPrice", "unitCost", "retail_unit_price_usd", "price_per_unit"]
)
def test_per_unit_measures_get_non_additive_guidance(name):
    context = inspect(pd.DataFrame({name: [1.25, 2.75, 3.1, 4.5]}))
    ops = {rec.operation: rec.disposition for rec in context.get_column(name).recommendations}
    assert ops["SUM"] == "not_applicable"
    assert ops["MEAN"] == "caution"
    assert not any(metric.kind == "measure_total" and name in metric.columns for metric in context.metrics)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("units_sold", [12, 7, 15, 20, 7, 17, 14, 11]),
        ("cost_by_unit", [1.25, 2.75, 3.1, 4.5]),
        ("price_unit", [1.25, 2.75, 3.1, 4.5]),
    ],
)
def test_nonmatching_unit_columns_stay_additive(name, values):
    context = inspect(pd.DataFrame({name: values}))
    ops = {rec.operation: rec.disposition for rec in context.get_column(name).recommendations}
    assert ops["SUM"] == "recommended"
    assert ops["MEAN"] == "recommended"
    assert any(metric.kind == "measure_total" and metric.columns == (name,) for metric in context.metrics)


def test_pipeline_renders_composite_grain_and_measure_conditions():
    rows = 36
    frame = pd.DataFrame({
        "account_id": [i // 6 for i in range(rows)],
        "day": list(pd.date_range("2025-01-01", periods=6)) * (rows // 6),
        "revenue": pd.Series([10 + 13 * i for i in range(rows - 1)] + [None], dtype="Int64"),
        "amount_band_v2": [1 + (i // 4) % 3 for i in range(rows)],
    })
    context = inspect(frame)
    assert set(context.grain.columns) == {"account_id", "day"}
    assert "Likely grain:" in context.summary()
    revenue, band = context.columns[2:]
    assert revenue.semantic.role == "measure"
    relationship = next(rel for rel in revenue.relationships if rel.kind == "measure_at_grain")
    assert set(relationship.related_columns) == {"account_id", "day"}
    assert band.semantic.role == "dimension"
    assert {rec.operation for rec in band.recommendations} == {"GROUP BY"}
    for rendered in (context.to_markdown(), context.to_context()):
        assert "SUM only for additive quantities" in rendered
        assert "MEAN for roughly symmetric values" in rendered
        assert "Aggregations: GROUP BY" in rendered or "Recommended aggregations: GROUP BY" in rendered


@pytest.mark.parametrize("values", [[], [None], [2.0]])
def test_report_pipeline_accepts_empty_and_single_row_numeric_frames(values):
    context = inspect(pd.DataFrame({"conversion_rate": pd.Series(values, dtype="Float64")}))
    assert context.profile.row_count == len(values)
    assert context.summary()
    if not values or values[0] is None:
        assert not context.columns[0].recommendations
    for rendered in (context.to_markdown(), context.to_context()):
        assert "Aggregations: SUM" not in rendered and "Recommended aggregations: SUM" not in rendered
