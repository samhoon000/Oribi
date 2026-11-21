import pytest

from oribi import inspect


def test_roles_have_evidence(sales_frame):
    context = inspect(sales_frame)
    roles = {c.name: c.semantic for c in context.columns}
    assert roles["order_id"].role == "identifier"
    assert roles["revenue"].role == "measure"
    assert roles["order_date"].role == "temporal"
    assert roles["order_id"].evidence


def test_numeric_low_cardinality_can_be_dimension():
    context = inspect(__import__("pandas").DataFrame({"rating": [1, 2, 1, 3, 2]}))
    assert context.columns[0].semantic.role == "dimension"


def test_unique_numeric_business_measure_stays_a_measure():
    import pandas as pd

    context = inspect(pd.DataFrame({"revenue": [10.0, 20.0, 35.0, 80.0]}))
    column = context.columns[0]
    assert column.semantic.role == "measure"
    assert {r.operation for r in column.recommendations} >= {"SUM", "MEAN", "MEDIAN"}


def test_unique_integer_revenue_and_amount_stay_measures():
    import pandas as pd

    for name in ("revenue", "amount"):
        column = inspect(pd.DataFrame({name: [10, 20, 35, 80]})).columns[0]
        assert column.semantic.role == "measure"
        assert "name_measure_hint" in {e.code for e in column.semantic.evidence}


def test_numeric_identifier_name_outweighs_numeric_measure_type():
    import pandas as pd

    context = inspect(pd.DataFrame({"customer_id": [101, 102, 103, 104]}))
    assert context.columns[0].semantic.role == "identifier"


def test_reusable_code_is_eligible_for_relationship_detection():
    import pandas as pd

    frame = pd.DataFrame({"store_code": ["a", "a", "b", "b"], "store_name": ["East", "East", "West", "West"]})
    context = inspect(frame)
    assert context.columns[0].semantic.role == "identifier"
    assert any(r.source_column == "store_code" and r.target_column == "store_name" for r in context.relationships)


def test_ambiguous_unique_numeric_column_remains_unknown():
    import pandas as pd

    context = inspect(pd.DataFrame({"score": [11, 17, 23, 41]}))
    assert context.columns[0].semantic.role == "unknown"


def test_short_numeric_measure_name_is_enough_for_measure_inference():
    import pandas as pd

    context = inspect(pd.DataFrame({"amount": [2.0, 4.5, 8.0]}))
    assert context.columns[0].semantic.role == "measure"


def test_short_unique_integer_revenue_and_amount_are_measures():
    import pandas as pd

    for name in ("revenue", "amount", "sales", "net_profit", "category_revenue", "bucket_amount", "revenue_by_category"):
        context = inspect(pd.DataFrame({name: [10, 20, 35]}))
        column = context.columns[0]
        assert column.semantic.role == "measure"
        evidence_codes = {e.code for e in column.semantic.evidence}
        assert "name_measure_hint" in evidence_codes
        assert "low_cardinality_numeric" not in evidence_codes
        assert {r.operation for r in column.recommendations} >= {"SUM", "MEAN", "MEDIAN"}
        assert any(rel.kind == "measure_at_grain" for rel in column.relationships)

    nullable = inspect(pd.DataFrame({"revenue": pd.Series([10, 20, 35, None], dtype="Int64")})).columns[0]
    assert nullable.semantic.role == "measure"


@pytest.mark.parametrize("name", ["amount_band", "amount_band_v2", "revenue_bucket_version_3"])
def test_repeated_integer_amount_band_keeps_dimension_role_and_relationships(name):
    import pandas as pd

    frame = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        name: [1, 1, 2, 2, 3, 3],
        "band_label": ["low", "low", "mid", "mid", "high", "high"],
    })
    context = inspect(frame)
    band = next(c for c in context.columns if c.name == name)
    assert band.semantic.role == "dimension"
    evidence_codes = {e.code for e in band.semantic.evidence}
    assert "low_cardinality_numeric" in evidence_codes
    assert "name_measure_hint" not in evidence_codes
    assert any(rel.kind == "groups_observations" for rel in band.relationships)
    assert any(
        rel.kind == "one_to_many" and rel.source_column == name and rel.target_column == "order_id"
        for rel in context.relationships
    )
    source, target = sorted((name, "band_label"))
    assert any(rel.kind == "functional_dependency" and rel.source_column == source and rel.target_column == target
               for rel in context.relationships)


@pytest.mark.parametrize(("name", "dtype"), [
    ("amount_band_v2", "Int64"), ("amount_percentile", "Int64"),
    ("profit_rank_v3", "Float64"), ("sales_category_code_v1", "Int64"),
])
def test_numeric_labels_and_ranks_do_not_acquire_measure_guidance(name, dtype):
    import pandas as pd

    # A percentile rank is an ordinal value, even when it includes "amount".
    context = inspect(pd.DataFrame({name: pd.Series([10, 20, 35, None], dtype=dtype)}))
    column = context.columns[0]
    assert column.semantic.role in {"dimension", "identifier", "unknown"}
    assert "name_measure_hint" not in {e.code for e in column.semantic.evidence}
    assert not any(rel.kind == "measure_at_grain" for rel in column.relationships)
    assert not any(rec.operation == "SUM" and rec.disposition != "not_applicable" for rec in column.recommendations)
    for rendered in (context.to_markdown(), context.to_context()):
        assert not any("SUM" in line for line in rendered.splitlines()
                       if line.startswith(("- Recommended aggregations:", "Aggregations:")))


def test_repeated_integer_measure_name_does_not_erase_grouping_evidence():
    import pandas as pd

    context = inspect(pd.DataFrame({"order_id": range(6), "amount": [1, 1, 2, 2, 3, 3]}))
    column = context.columns[1]
    assert column.semantic.role == "dimension"
    assert any(rel.kind == "groups_observations" for rel in column.relationships)


def test_identifier_precedence_and_misleading_substring_controls():
    import pandas as pd

    for name in ("revenue_id", "amount_key", "amount_code"):
        column = inspect(pd.DataFrame({name: [101, 102, 103, 104]})).columns[0]
        assert column.semantic.role == "identifier"
        assert "name_measure_hint" not in {e.code for e in column.semantic.evidence}

    for name in ("amount_code", "store_code"):
        column = inspect(pd.DataFrame({name: [101, 102, 103]})).columns[0]
        assert column.semantic.role == "unknown"

    low_diversity_id = inspect(pd.DataFrame({"revenue_id": [1, 2] * 50})).columns[0]
    assert low_diversity_id.semantic.role == "dimension"

    for name, values in (
        ("paramount", [10, 20, 35, 80]),
        ("amounted", [10, 20, 35, 80]),
        ("nonrevenue", [10, 20, 35, 80]),
        ("amount_band", [1, 2, 3, 4]),
    ):
        column = inspect(pd.DataFrame({name: values})).columns[0]
        assert column.semantic.role == "unknown"

    short_repeated = inspect(pd.DataFrame({"rating": [1, 1, 2]})).columns[0]
    assert short_repeated.semantic.role == "dimension"

    short_unnamed = inspect(pd.DataFrame({"rating": [1, 2, 3]})).columns[0]
    assert short_unnamed.semantic.role == "dimension"


def test_identifier_guardrails_discount_float_negative_and_constant_values():
    import pandas as pd

    frame = pd.DataFrame({
        "invoice_no": list(range(101, 105)),
        "amount_id": [12.5, 24.75, 38.25, 51.0],
        "negative_id": [-101, -102, -103, -104],
        "constant_id": list(range(42, 43)) * 4,
    })
    context = inspect(frame)
    by_name = {col.name: col for col in context.columns}

    assert by_name["invoice_no"].semantic.role == "identifier"

    assert by_name["amount_id"].semantic.role != "identifier"
    assert "non_identifier_numeric_domain" in {e.code for e in by_name["amount_id"].semantic.evidence}

    assert by_name["negative_id"].semantic.role == "unknown"
    assert "non_identifier_numeric_domain" in {e.code for e in by_name["negative_id"].semantic.evidence}

    assert by_name["constant_id"].semantic.role != "identifier"
    assert "constant_numeric_values" in {e.code for e in by_name["constant_id"].semantic.evidence}
    assert not any(r.operation == "COUNT DISTINCT" and r.disposition == "recommended" for r in by_name["constant_id"].recommendations)


def test_conflicting_and_borderline_numeric_signals_remain_conservative():
    import pandas as pd

    frame = pd.DataFrame({
        "borderline_seq": list(range(1, 10)) + list(range(1, 2)),
        "constant_float": [3.14] * 10,
        "signed_code": [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1],
    })
    context = inspect(frame)
    by_name = {col.name: col for col in context.columns}

    borderline = by_name["borderline_seq"]
    assert borderline.semantic.role == "unknown"
    assert "high_integer_uniqueness" in {e.code for e in borderline.semantic.evidence}

    const_float = by_name["constant_float"]
    assert const_float.semantic.role == "unknown"
    assert "constant_numeric_values" in {e.code for e in const_float.semantic.evidence}

    assert by_name["signed_code"].semantic.role == "dimension"
    assert not any(r.operation in {"SUM", "MEAN"} and r.disposition == "recommended" for r in by_name["signed_code"].recommendations)


def test_camelcase_and_acronym_identifier_names():
    import pandas as pd

    from oribi.reasoning.semantic import column_name_words

    values = [1, 1, 2, 2, 3, 3, 4]
    for name in ("customerId", "customerID", "customer_id"):
        column = inspect(pd.DataFrame({name: values})).get_column(name)
        assert column.semantic.role == "identifier"
        assert "name_identifier_hint" in {e.code for e in column.semantic.evidence}
        assert any(r.operation == "COUNT DISTINCT" and r.disposition == "recommended" for r in column.recommendations)
        assert any(r.operation == "SUM" and r.disposition == "not_applicable" for r in column.recommendations)
        assert any(r.operation == "MEAN" and r.disposition == "not_applicable" for r in column.recommendations)

    paid_column = inspect(pd.DataFrame({"paid": values})).get_column("paid")
    assert "name_identifier_hint" not in {e.code for e in paid_column.semantic.evidence}
    assert paid_column.semantic.role != "identifier"

    assert column_name_words("customerId") == ["customer", "id"]
    assert column_name_words("customerID") == ["customer", "id"]
    assert column_name_words("customer_id") == ["customer", "id"]
    assert column_name_words("paid") == ["paid"]
    assert column_name_words("amount_band_v2") == ["amount", "band"]
    assert column_name_words("revenue_bucket_version_3") == ["revenue", "bucket"]
    assert column_name_words("profit_rank_v3") == ["profit", "rank"]
    assert column_name_words("sales_category_code_v1") == ["sales", "category", "code"]
