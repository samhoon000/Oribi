import json
import sqlite3
from decimal import Decimal

import pandas as pd
import pytest

from oribi import MetricDefinition, ProfileOptions, RelationshipContext, inspect
from oribi.reasoning.metrics import _sql_col, infer_metrics


def _metric(context, name):
    return next(metric for metric in context.metrics if metric.name == name)


def test_orders_dataset_infers_common_analytical_metrics():
    frame = pd.DataFrame(
        {
            "order_id": [101, 102, 103, 104, 105, 106],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c4"],
            "revenue": [25.0, 40.0, 30.0, 50.0, 15.0, 60.0],
            "order_date": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-05",
                    "2026-01-06",
                ]
            ),
        }
    )
    context = inspect(frame)
    by_name = {metric.name: metric for metric in context.metrics}

    assert {
        "order_count",
        "customer_count",
        "total_revenue",
        "average_order_value",
        "revenue_per_customer",
        "orders_per_customer",
    } <= set(by_name)

    order_count = by_name["order_count"]
    assert isinstance(order_count, MetricDefinition)
    assert order_count.kind == "entity_count"
    assert order_count.sql == "COUNT(DISTINCT order_id)"
    assert order_count.columns == ("order_id",)
    assert 0.5 < order_count.confidence <= 0.98
    assert order_count.evidence

    aov = by_name["average_order_value"]
    assert aov.kind == "measure_per_entity"
    assert aov.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"
    assert aov.columns == ("revenue", "order_id")
    assert 0.5 < aov.confidence <= 0.98
    assert any(e.code == "additive_measure_and_entity" for e in aov.evidence)

    rpc = by_name["revenue_per_customer"]
    assert rpc.kind == "measure_per_entity"
    assert rpc.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert rpc.columns == ("revenue", "customer_id")
    assert any(e.code == "repeated_entity_observations" for e in rpc.evidence)

    opc = by_name["orders_per_customer"]
    assert opc.kind == "entity_ratio"
    assert opc.sql == "1.0 * COUNT(DISTINCT order_id) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert opc.columns == ("order_id", "customer_id")
    assert any(e.code == "one_to_many_entity_hierarchy" for e in opc.evidence)


def test_order_items_grain_uses_distinct_order_count_for_average_order_value():
    frame = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 102, 103, 103],
            "product_id": [1, 2, 1, 3, 2, 3],
            "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            "revenue": [20.0, 30.0, 15.0, 25.0, 40.0, 10.0],
            "quantity": [1, 2, 3, 4, 5, 2],
        }
    )
    context = inspect(frame)
    aov = _metric(context, "average_order_value")
    assert aov.sql == "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"
    assert any(e.code == "repeated_entity_observations" for e in aov.evidence)

    rpu = _metric(context, "revenue_per_unit")
    assert rpu.kind == "measure_ratio"
    assert rpu.sql == "SUM(revenue) / NULLIF(SUM(quantity), 0)"
    assert rpu.columns == ("revenue", "quantity")


def test_integer_measure_division_avoids_integer_truncation_in_sql():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "revenue": pd.Series([10, 20, 35, 80], dtype="int64"),
            "profit": pd.Series([2, 5, 10, 25], dtype="int64"),
        }
    )
    context = inspect(frame)
    aov = _metric(context, "average_order_value")
    assert aov.sql == "1.0 * SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"

    margin = _metric(context, "profit_margin")
    assert margin.kind == "measure_ratio"
    assert margin.sql == "1.0 * SUM(profit) / NULLIF(SUM(revenue), 0)"
    assert margin.columns == ("profit", "revenue")


@pytest.mark.parametrize(
    "name", ["conversion_rate", "price_per_unit", "profit_ratio", "gross_margin (%)"]
)
def test_non_additive_measures_do_not_generate_sum_based_metrics(name):
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            name: [0.15, 0.20, 0.25, 0.40],
        }
    )
    context = inspect(frame)
    assert [m.name for m in context.metrics] == ["order_count"]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"order_id": pd.Series(dtype="int64"), "revenue": pd.Series(dtype="float64")}),
        pd.DataFrame({"order_id": [None, None, None], "revenue": [None, None, None]}),
        pd.DataFrame({"order_id": [1], "revenue": [25.0]}),
        pd.DataFrame({"order_id": ["same"] * 6, "revenue": [10.0] * 6}),
    ],
)
def test_empty_null_single_row_and_constant_frames_emit_no_metrics(frame):
    assert inspect(frame).metrics == ()


def test_conservative_guardrails_skip_low_diversity_ids_dimensions_and_row_keys():
    frame = pd.DataFrame(
        {
            "row_id": list(range(40)),
            "status_id": [1, 2] * 20,
            "region": ["east", "west"] * 20,
            "revenue": [10.0 + float(i) for i in range(40)],
        }
    )
    context = inspect(frame)
    names = {m.name for m in context.metrics}
    assert "row_count" in names
    assert "total_revenue" in names
    assert "status_count" not in names
    assert "revenue_per_row" not in names
    assert "revenue_per_region" not in names
    assert "revenue_per_status" not in names


def test_arbitrary_unique_key_without_name_hint_skips_speculative_per_entity_metric():
    frame = pd.DataFrame(
        {
            "K": [f"k{i}" for i in range(20)],
            "M": [float(i % 7) + 1.0 for i in range(20)],
        }
    )
    context = inspect(frame)
    names = {m.name for m in context.metrics}
    assert "m_per_k" not in names


def test_metric_caveats_reflect_measure_warnings_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 4],
            "customer_id": ["c1", "c1", "c2", "c3", "c3"],
            "revenue": [10.0, -5.0, 20.0, 1000.0, 1000.0],
        }
    )
    context = inspect(frame)
    order_count = _metric(context, "order_count")
    assert not any("duplicate rows" in caveat.lower() for caveat in order_count.caveats)

    aov = _metric(context, "average_order_value")
    combined_caveats = " ".join(aov.caveats).lower()
    assert "negative values" in combined_caveats
    assert "duplicate rows" in combined_caveats


def test_camelcase_and_special_column_names_quote_sql_and_render_safely():
    frame = pd.DataFrame(
        {
            "orderId": [1, 2, 3, 4],
            "customerId": ["c1", "c1", "c2", "c3"],
            "gross revenue ($)": [10.0, 20.0, 30.0, 40.0],
            "totalSales": [12.0, 24.0, 36.0, 48.0],
        }
    )
    context = inspect(frame)
    by_name = {m.name: m for m in context.metrics}
    assert "order_count" in by_name
    assert "customer_count" in by_name
    assert "total_sales" in by_name
    assert "total_total_sales" not in by_name
    assert by_name["total_gross_revenue"].sql == 'SUM("gross revenue ($)")'
    assert (
        by_name["average_order_value"].sql
        == 'SUM("gross revenue ($)") / NULLIF(COUNT(DISTINCT orderId), 0)'
    )


def test_metrics_serialize_privately_and_render_in_markdown_and_compact_context():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "patient_id": ["p1", "p1", "p2", "p3"],
            "diagnosis": pd.Categorical(["HIV", "cancer", "HIV", "cancer"]),
            "charge": [100.0, 250.0, 150.0, 400.0],
        }
    )
    context = inspect(frame)
    payload = context.to_dict()
    assert payload["schema_version"] == "2"
    assert isinstance(payload["metrics"], list)
    assert any(item["name"] == "order_count" for item in payload["metrics"])

    encoded = context.to_json()
    assert json.loads(encoded) == payload
    assert "HIV" not in encoded
    assert "cancer" not in encoded

    markdown = context.to_markdown()
    assert "## Analytical metrics" in markdown
    assert "COUNT(DISTINCT order_id)" in markdown
    assert "SUM(charge) / NULLIF(COUNT(DISTINCT order_id), 0)" in markdown
    assert "HIV" not in markdown

    compact = context.to_context()
    assert "METRICS" in compact
    assert compact.index("CARDINALITY") < compact.index("METRICS") < compact.index("COLUMNS")
    assert "average_order_value" in compact
    assert "HIV" not in compact

    tight = context.to_context(max_chars=250)
    assert len(tight) <= 250
    assert tight.endswith("[Context truncated.]")


def test_metrics_are_deterministic_across_column_order_and_bounded_on_wide_frames():
    frame = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c4"],
            "revenue": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )
    reordered = frame[["revenue", "customer_id", "order_id"]]
    assert inspect(frame).metrics == inspect(reordered).metrics

    wide_data = {
        "order_id": list(range(30)),
        "customer_id": [f"c{i // 2}" for i in range(30)],
        "revenue": [float(i + 10) for i in range(30)],
        **{f"entity_{i}_id": [f"e{j}_{i}" for j in range(30)] for i in range(12)},
        **{f"amount_{i}": [float(j + i + 1) for j in range(30)] for i in range(12)},
    }
    wide_context = inspect(pd.DataFrame(wide_data))
    assert len(wide_context.metrics) <= 20
    assert any(m.name == "average_order_value" for m in wide_context.metrics)


def _eval_sql(frame: pd.DataFrame, sql: str) -> float:
    query = sql if sql.lstrip().upper().startswith(("WITH ", "SELECT ")) else f"SELECT {sql} FROM dataset"
    with sqlite3.connect(":memory:") as conn:
        frame.to_sql("dataset", conn, index=False, if_exists="replace")
        row = conn.execute(query).fetchone()
        assert row is not None and row[0] is not None
        return float(row[0])


def test_decimal_measure_ratio_avoids_integer_division():
    frame = pd.DataFrame({
        "order_id": [1, 2],
        "revenue": pd.Series([Decimal("1.00"), Decimal("2.00")], dtype=object),
    })
    metric = _metric(inspect(frame), "average_order_value")
    # SQLite can store whole values as integers in a NUMERIC column.
    assert _eval_sql(frame.assign(revenue=[1, 2]), metric.sql) == 1.5


@pytest.mark.parametrize(("revenue", "profit"), [(100.0, None), (None, 40.0)])
def test_metrics_withhold_sums_with_an_incomplete_determinant(revenue, profit):
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 1, 2, 2, None], dtype="Int64"),
        "revenue": [20.0, 20.0, 30.0, 30.0, revenue],
        "profit": [2.0, 2.0, 9.0, 9.0, profit],
    })
    context = inspect(frame)
    ambiguous = "revenue" if revenue is not None else "profit"
    covered = "profit" if ambiguous == "revenue" else "revenue"

    assert all(ambiguous not in metric.columns for metric in context.metrics)
    assert _eval_sql(frame, _metric(context, f"total_{covered}").sql) == (
        11.0 if covered == "profit" else 50.0
    )


@pytest.mark.parametrize("include_empty_row", [False, True])
def test_deduplicated_metrics_only_require_keys_for_observed_measures(include_empty_row):
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 1, 2, 2, None], dtype="Int64"),
        "revenue": [20.0, 20.0, 30.0, 30.0, None],
        "profit": [2.0, 2.0, 9.0, 9.0, None],
    })
    if not include_empty_row:
        frame = frame.iloc[:-1]
    context = inspect(frame)

    assert _eval_sql(frame, _metric(context, "total_revenue").sql) == 50.0
    assert _eval_sql(frame, _metric(context, "average_order_value").sql) == 25.0
    assert _eval_sql(frame, _metric(context, "profit_margin").sql) == pytest.approx(11.0 / 50.0)


def test_complete_determinant_can_replace_incomplete_determinant():
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 1, 2, 2, None], dtype="Int64"),
        "invoice_id": [11, 11, 12, 12, 13],
        "revenue": [20.0, 20.0, 30.0, 30.0, 100.0],
    })
    total = _metric(inspect(frame), "total_revenue")

    assert "GROUP BY invoice_id" in total.sql
    assert _eval_sql(frame, total.sql) == 150.0


def test_missing_entity_key_does_not_withhold_line_item_sum():
    frame = pd.DataFrame({
        "order_id": pd.Series([1, 1, 2, 2, None], dtype="Int64"),
        "revenue": [20.0, 21.0, 30.0, 31.0, 100.0],
    })
    total = _metric(inspect(frame), "total_revenue")

    assert _eval_sql(frame, total.sql) == 202.0


def test_repeated_order_totals_across_line_items_deduplicate_aov_total_and_ratios():
    frame = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 102, 103, 103],
            "product_id": [1, 2, 1, 3, 2, 3],
            "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
            "revenue": [50.0, 50.0, 40.0, 40.0, 60.0, 60.0],
            "quantity": [1, 2, 3, 4, 5, 2],
        }
    )
    context = inspect(frame)

    aov = _metric(context, "average_order_value")
    assert aov.sql != "SUM(revenue) / NULLIF(COUNT(DISTINCT order_id), 0)"
    assert "GROUP BY order_id" in aov.sql
    assert any(e.code == "measure_constant_per_entity" for e in aov.evidence)
    assert any("double-counting" in c.lower() for c in aov.caveats)
    assert _eval_sql(frame, aov.sql) == pytest.approx(50.0)

    total_rev = _metric(context, "total_revenue")
    assert total_rev.sql != "SUM(revenue)"
    assert _eval_sql(frame, total_rev.sql) == pytest.approx(150.0)

    rpc = _metric(context, "revenue_per_customer")
    assert _eval_sql(frame, rpc.sql) == pytest.approx(75.0)

    rpu = _metric(context, "revenue_per_unit")
    assert _eval_sql(frame, rpu.sql) == pytest.approx(150.0 / 17.0)

    total_qty = _metric(context, "total_quantity")
    assert total_qty.sql == "SUM(quantity)"
    assert _eval_sql(frame, total_qty.sql) == pytest.approx(17.0)

    opc = _metric(context, "orders_per_customer")
    assert opc.sql == "1.0 * COUNT(DISTINCT order_id) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert _eval_sql(frame, opc.sql) == pytest.approx(1.5)


@pytest.mark.parametrize(
    "options", [ProfileOptions(max_relationship_columns=1), None], ids=["explicit", "default"],
)
def test_unhashable_metadata_does_not_displace_metric_grain(options):
    # An incomplete pair keeps the exact mapping scan from supplying the same grain fact.
    rows = pd.DataFrame({
        "entity_id": ["e1", "e1", "e2", "e2", "e3", "e3", "e3"],
        "amount": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, None],
    })
    metadata = pd.Series([[1], "[1]", [2], "[2]", [3], "[3]", "extra"], dtype=object)
    limit = (options or ProfileOptions()).max_relationship_columns
    metadata_columns = {f"a_metadata_{i}": metadata for i in range(limit)}
    context = inspect(rows.assign(**metadata_columns), options=options)

    total = _metric(context, "total_amount")
    assert _eval_sql(rows, total.sql) == pytest.approx(60.0), total.sql
    assert all(not metadata_columns.keys() & metric.columns for metric in context.metrics)


def test_repeated_integer_order_totals_and_coarser_customer_grain_measures():
    frame = pd.DataFrame(
        {
            "customer_id": ["c1", "c1", "c1", "c1", "c2", "c2", "c2", "c2"],
            "order_id": [101, 101, 102, 102, 103, 103, 104, 104],
            "revenue": pd.Series([50, 50, 40, 40, 60, 60, 70, 70], dtype="int64"),
            "customer_ltv": [100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0, 200.0],
        }
    )
    context = inspect(frame)

    aov = _metric(context, "average_order_value")
    assert "1.0 *" in aov.sql
    assert _eval_sql(frame, aov.sql) == pytest.approx(55.0)

    total_ltv = _metric(context, "total_customer_ltv")
    assert "GROUP BY customer_id" in total_ltv.sql
    assert _eval_sql(frame, total_ltv.sql) == pytest.approx(300.0)

    ltv_per_order = _metric(context, "customer_ltv_per_order")
    assert "GROUP BY customer_id" in ltv_per_order.sql
    assert any("customer_id grain rather than order_id grain" in c for c in ltv_per_order.caveats)
    assert _eval_sql(frame, ltv_per_order.sql) == pytest.approx(75.0)


def test_entity_ratio_with_multi_parent_child_uses_per_parent_deduplicated_sql():
    # 60 orders across 10 customers (3 line items per order); order_id=0 belongs to both c0 and c1.
    customer_ids = [f"c{i % 10}" for i in range(60) for _ in range(3)]
    order_ids = [i for i in range(60) for _ in range(3)]
    customer_ids[2] = "c1"
    frame = pd.DataFrame({"customer_id": customer_ids, "order_id": order_ids})

    context = inspect(frame)
    opc = _metric(context, "orders_per_customer")
    assert opc.sql != "1.0 * COUNT(DISTINCT order_id) / NULLIF(COUNT(DISTINCT customer_id), 0)"
    assert "GROUP BY customer_id" in opc.sql
    assert any(e.code == "multi_parent_entity_relationship" for e in opc.evidence)
    assert any("multiple customer_id parents" in c for c in opc.caveats)
    # 60 distinct orders across 10 customers, with order 0 shared between c0 and c1 -> 61 / 10 = 6.1.
    assert _eval_sql(frame, opc.sql) == pytest.approx(6.1)


def test_many_to_many_entities_do_not_emit_misleading_entity_ratio():
    frame = pd.DataFrame(
        {
            "parent_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "child_id": ["c1", "c2", "c1", "c3", "c4", "c5"],
        }
    )
    context = inspect(frame)
    assert not any(m.kind == "entity_ratio" for m in context.metrics)


def test_infer_metrics_guards_one_to_many_when_child_belongs_to_multiple_parents_or_missing_parent():
    base_frame = pd.DataFrame(
        {
            "parent_id": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "child_id": ["c1", "c2", "c1", "c3", "c4", "c5"],
        }
    )
    base_ctx = inspect(base_frame)

    multi_parent_rel = (
        RelationshipContext(
            source_column="parent_id",
            target_column="child_id",
            kind="one_to_many",
            coverage=0.80,
            confidence=0.82,
            violation_count=1,
            affected_groups=1,
        ),
    )
    multi_metrics = infer_metrics(
        base_ctx.columns,
        relationships=multi_parent_rel,
        row_count=len(base_frame),
    )
    ratio = next(m for m in multi_metrics if m.name == "childs_per_parent")
    assert ratio.sql != "1.0 * COUNT(DISTINCT child_id) / NULLIF(COUNT(DISTINCT parent_id), 0)"
    assert "GROUP BY parent_id" in ratio.sql
    assert any(e.code == "multi_parent_entity_relationship" for e in ratio.evidence)
    # p1: {c1, c2}=2, p2: {c1, c3}=2, p3: {c4, c5}=2 -> average = 2.0 (not 5/3 = 1.67).
    assert _eval_sql(base_frame, ratio.sql) == pytest.approx(2.0)

    missing_parent_frame = pd.DataFrame(
        {
            "parent_id": ["p1", "p1", "p2", "p2", None],
            "child_id": ["c1", "c2", "c3", "c4", "c5"],
        }
    )
    missing_ctx = inspect(missing_parent_frame)
    missing_rel = (
        RelationshipContext(
            source_column="parent_id",
            target_column="child_id",
            kind="one_to_many",
            coverage=0.80,
            confidence=0.85,
            missing_pair_count=1,
        ),
    )
    missing_metrics = infer_metrics(
        missing_ctx.columns,
        relationships=missing_rel,
        row_count=len(missing_parent_frame),
    )
    missing_ratio = next(m for m in missing_metrics if m.name == "childs_per_parent")
    assert "CASE WHEN parent_id IS NOT NULL THEN child_id END" in missing_ratio.sql
    assert any("missing parent_id" in c.lower() for c in missing_ratio.caveats)
    assert _eval_sql(missing_parent_frame, missing_ratio.sql) == pytest.approx(2.0)


def test_sql_keywords_used_as_metric_columns_are_quoted():
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4],
            "select": [11.5, 25.5, 40.5, 51.5],
        }
    )
    metric = next(m for m in inspect(df).metrics if m.name == "total_select")
    assert metric.sql == 'SUM("select")'
    assert _eval_sql(df, metric.sql) == pytest.approx(129.0)


def test_sql_keywords_case_insensitive_quoting():
    for col_name in ("SELECT", "Select", "where", "GROUP"):
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                col_name: [10.0, 20.0, 30.0, 40.0],
            }
        )
        ctx = inspect(df)
        metric = next(m for m in ctx.metrics if m.columns == (col_name,))
        assert metric.sql == f'SUM("{col_name}")'
        assert _eval_sql(df, metric.sql) == pytest.approx(100.0)


def test_sql_col_formatting_with_keywords_and_identifiers():
    assert _sql_col("order_id") == "order_id"
    assert _sql_col("revenue") == "revenue"
    assert _sql_col("select") == '"select"'
    assert _sql_col("SELECT") == '"SELECT"'
    assert _sql_col("Select") == '"Select"'
    assert _sql_col("where") == '"where"'
    assert _sql_col("group") == '"group"'
    assert _sql_col("order") == '"order"'
    assert _sql_col("col with space") == '"col with space"'
    assert _sql_col('col"quote') == '"col""quote"'


def _repeated_order_revenue_frame():
    return pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2],
            "line_id": [1, 2, 3, 4],
            "revenue": [20.0, 20.0, 30.0, 30.0],
        }
    )


def test_compact_context_never_shows_dedup_sql_without_its_caveat():
    context = inspect(_repeated_order_revenue_frame())

    for budget in range(100, 5001, 25):
        report = context.to_context(max_chars=budget)
        assert len(report) <= budget
        if "- total_revenue" in report:
            assert "coincidental" in report, budget


def test_compact_context_keeps_both_sql_and_caveat_with_room():
    report = inspect(_repeated_order_revenue_frame()).to_context(max_chars=5000)

    assert "- total_revenue" in report
    assert "coincidental" in report
    assert "[Context truncated.]" not in report


def test_metric_caveats_stay_in_structured_output():
    payload = inspect(_repeated_order_revenue_frame()).to_dict()
    total = next(m for m in payload["metrics"] if m["name"] == "total_revenue")

    assert any("coincidental" in caveat for caveat in total["caveats"])
