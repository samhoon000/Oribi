import numpy as np
import pandas as pd
import pytest

from oribi import ProfileOptions, inspect


def test_large_integer_bounds_keep_exact_precision():
    values = [2**53 + 1, 2**53 + 3]
    context = inspect(pd.DataFrame({"value": values}))
    statistics = context.to_dict(include_values=True)["columns"][0]["statistics"]
    assert statistics["minimum"] == values[0]
    assert statistics["maximum"] == values[1]


@pytest.mark.parametrize("values,mean", [([1, 2, 3, pd.NA], 2.0), ([1.5, 2.5, pd.NA], 2.0), ([1, 3.0, pd.NA], 2.0)])
def test_numeric_object_values_with_missing_data_are_profiled(values, mean):
    column = inspect(pd.DataFrame({"amount": pd.Series(values, dtype=object)})).columns[0]
    assert column.logical_type == "numeric"
    assert column.statistics.missing_count == 1
    assert column.statistics.mean == mean
    assert column.statistics.minimum == min(v for v in values if v is not pd.NA)


@pytest.mark.parametrize("values", [
    [True, False, pd.NA], [1, True, pd.NA], ["1", "2", pd.NA],
    [1, "2", pd.NA], [1.0, "x", pd.NA], [1 + 2j, pd.NA],
    [pd.NA, None], list(range(101)) + ["late text", pd.NA],
])
def test_object_controls_do_not_enter_numeric_statistics(values):
    column = inspect(pd.DataFrame({"amount": pd.Series(values, dtype=object)})).columns[0]
    assert column.logical_type != "numeric"
    assert column.statistics.mean is None


def test_numeric_and_temporal_profile(sales_frame):
    context = inspect(sales_frame)
    revenue = next(c for c in context.columns if c.name == "revenue")
    dates = next(c for c in context.columns if c.name == "order_date")
    assert revenue.statistics.minimum == -2
    assert revenue.statistics.negative_count == 1
    assert revenue.statistics.zero_count == 1
    assert dates.logical_type == "temporal"
    assert dates.statistics.minimum.year == 2024


def test_missing_duplicates_and_empty():
    frame = pd.DataFrame({"x": [1, 1, None], "label": ["a", "a", "b"]})
    context = inspect(frame)
    assert context.profile.duplicate_row_count == 1
    assert context.columns[0].statistics.missing_count == 1
    empty = inspect(pd.DataFrame({"x": pd.Series(dtype="float64")}))
    assert empty.profile.row_count == 0
    assert empty.columns[0].statistics.distinct_count == 0


def test_all_null_and_mixed_columns():
    context = inspect(pd.DataFrame({"empty": [None, None], "mixed": ["a", 1]}))
    assert context.columns[0].logical_type == "unknown"
    assert context.columns[0].statistics.missing_rate == 1
    assert context.columns[1].logical_type == "unknown"


def _temporal_warnings(values):
    context = inspect(pd.DataFrame({"event_time": pd.to_datetime(values)}))
    column = next(c for c in context.columns if c.name == "event_time")
    return context, column, [w for w in column.warnings if w.code == "potential_temporal_gap"]


def test_regular_daily_series_with_gap_has_cadence_and_interval_evidence():
    context, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
    assert len(warnings) == 1
    evidence = warnings[0].evidence[0]
    assert evidence.details["cadence"] == "1 day"
    assert evidence.details["expected_timestamps"] == ("2026-01-03T00:00:00",)
    assert evidence.details["observed_before"] == "2026-01-02T00:00:00"
    assert evidence.details["observed_after"] == "2026-01-04T00:00:00"
    assert warnings[0] in context.warnings


def test_multiple_missing_intervals_are_reported():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-06", "2026-01-07"])
    assert warnings[0].evidence[0].details["missing_intervals"] == 3


def test_weekly_cadence_is_supported():
    _, _, warnings = _temporal_warnings(["2026-01-05", "2026-01-12", "2026-01-26", "2026-02-02"])
    assert warnings[0].evidence[0].details["cadence"] == "7 days"


def test_temporal_gap_warning_preserves_nanosecond_cadence():
    _, _, warnings = _temporal_warnings(pd.to_datetime([0, 5, 10, 20, 25], unit="ns"))
    (warning,) = warnings
    (evidence,) = warning.evidence
    cadence = "0 days 00:00:00.000000005"

    assert evidence.details["cadence"] == cadence
    assert f"{cadence} cadence" in warning.message
    assert evidence.details["missing_intervals"] == 1
    assert evidence.details["expected_timestamps"] == ("1970-01-01T00:00:00.000000015",)


def test_irregular_series_does_not_generate_gap_warning():
    _, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-08", "2026-01-09", "2026-01-17", "2026-01-25"])
    assert warnings == []


def test_duplicates_do_not_distort_cadence_and_gap():
    _, column, warnings = _temporal_warnings(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-05"])
    assert len(warnings) == 1
    assert warnings[0].evidence[0].details["expected_timestamps"] == ("2026-01-04T00:00:00",)
    assert column.statistics.minimum == pd.Timestamp("2026-01-01")
    assert column.statistics.maximum == pd.Timestamp("2026-01-05")


def test_short_and_ambiguous_series_do_not_generate_gap_warning():
    _, _, short = _temporal_warnings(["2026-01-01", "2026-01-03", "2026-01-05"])
    _, _, ambiguous = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-07"])
    assert short == []
    assert ambiguous == []


def test_overflowing_timestamp_interval_skips_gap_detection_and_preserves_bounds():
    stamps = np.array(
        ["1680-01-01", "1680-01-02", "2250-01-01", "2250-01-02"],
        dtype="datetime64[ns]",
    )
    context = inspect(pd.DataFrame({"stamp": stamps}))
    column = context.columns[0]

    assert context.profile.row_count == 4
    assert column.logical_type == "temporal"
    assert column.statistics.minimum == pd.Timestamp("1680-01-01")
    assert column.statistics.maximum == pd.Timestamp("2250-01-02")
    assert not any(w.code == "potential_temporal_gap" for w in column.warnings)
    assert not any(w.code == "potential_temporal_gap" for w in context.warnings)


def test_timezone_aware_and_nullable_dtypes_profile():
    frame = pd.DataFrame({
        "at": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
        "count": pd.Series([1, None, 3, 4], dtype="Int64"),
        "active": pd.Series([True, None, False, True], dtype="boolean"),
        "category": pd.Series(["a", "b", "a", "c"], dtype="category"),
    })
    context = inspect(frame)
    columns = {column.name: column for column in context.columns}
    assert columns["at"].logical_type == "temporal"
    assert columns["at"].statistics.minimum.tz is not None
    assert columns["count"].statistics.missing_count == 1
    assert columns["active"].logical_type == "boolean"
    assert columns["category"].logical_type == "categorical"


def test_high_cardinality_categorical_column_has_warning():
    frame = pd.DataFrame({"group": pd.Series([f"g{i}" for i in range(110)], dtype="category")})
    context = inspect(frame)
    group = context.columns[0]
    assert group.logical_type == "categorical"
    assert any(w.code == "high_cardinality_category" for w in group.warnings)


def test_unused_categories_are_excluded_from_top_values():
    frame = pd.DataFrame({"category": pd.Series(["a"], dtype=pd.CategoricalDtype(categories=["a", "b", "c"]))})
    column = next(c for c in inspect(frame).columns if c.name == "category")
    assert column.statistics.top_values == (("a", 1),)
    assert column.statistics.top_frequencies == (1,)


@pytest.mark.parametrize("values", [
    ["beta", "alpha", "beta", "alpha"],
    ["alpha", "beta", "alpha", "beta"],
])
def test_top_value_ties_are_independent_of_row_order(values):
    column = inspect(pd.DataFrame({"category": values}), options=ProfileOptions(top_k=1)).columns[0]
    assert column.statistics.top_values == (("alpha", 2),)
    assert column.statistics.top_frequencies == (2,)


def test_top_k_tied_categories_keep_limit_and_counts():
    frame = pd.DataFrame({
        "category": pd.Series(["beta", "alpha", "beta", "alpha"], dtype="category"),
    })
    column = inspect(frame, options=ProfileOptions(top_k=2)).columns[0]
    assert column.statistics.top_values == (("alpha", 2), ("beta", 2))
    assert column.statistics.top_frequencies == (2, 2)


@pytest.mark.parametrize("error", [RuntimeError, ValueError])
@pytest.mark.parametrize("extra_values", [0, 40])
def test_unrepresentable_values_do_not_break_top_k(error, extra_values):
    class Opaque:
        def __repr__(self):
            raise error("representation unavailable")

    opaque = Opaque()
    values = ["common", "common", opaque, *range(extra_values), None]
    column = inspect(pd.DataFrame({"category": values}), options=ProfileOptions(top_k=1)).columns[0]
    assert column.statistics.top_values == (("common", 2),)
    assert column.statistics.top_frequencies == (2,)

    selected = inspect(pd.DataFrame({"category": [opaque, opaque, "common"]})).columns[0]
    value, count = selected.statistics.top_values[0]
    assert value is opaque
    assert count == 2


def test_opaque_representation_collisions_use_count_order_after_supported_values():
    class Opaque:
        def __repr__(self):
            return "same representation"

    a, b = Opaque(), Opaque()
    for first, second in ((a, b), (b, a)):
        frame = pd.DataFrame({"category": pd.Categorical(
            [a, b, a, b, "text", "text"], categories=[first, second, "text"],
        )})
        stats = inspect(frame, options=ProfileOptions(top_k=2)).columns[0].statistics
        assert stats.top_values[0] == ("text", 2)
        assert stats.top_values[1][0] is first
        assert stats.top_frequencies == (2, 2)


def test_supported_mixed_values_have_stable_ties():
    left, right = frozenset({"a", "x"}), frozenset({"b", "x"})
    values = [right, "1", 1, left, right, "1", 1, left, None]
    expected = ((left, 2), (right, 2), (1, 2), ("1", 2))
    for ordering in (values, values[::-1]):
        stats = inspect(pd.DataFrame({"mixed": pd.Series(ordering, dtype=object)})).columns[0].statistics
        assert stats.top_values == expected
        assert stats.missing_count == 1


@pytest.mark.parametrize(("dtype", "values"), [
    ("Int64", [1, 0]), ("UInt64", [1, 0]), ("Float64", [1.0, 0.0]), ("boolean", [True, False]),
])
def test_nullable_scalar_ties_have_stable_order(dtype, values):
    for ordering in (values, values[::-1]):
        frame = pd.DataFrame({"value": pd.Series(ordering * 2 + [None], dtype=dtype)})
        stats = inspect(frame, options=ProfileOptions(top_k=1)).columns[0].statistics
        assert stats.top_values == ((values[1], 2),)


def test_interval_category_ties_have_stable_order():
    left, right = pd.Interval(0, 1), pd.Interval(1, 2)
    for categories in ([right, left], [left, right]):
        frame = pd.DataFrame({"band": pd.Categorical(categories * 2, categories=categories)})
        stats = inspect(frame, options=ProfileOptions(top_k=1)).columns[0].statistics
        assert stats.top_values == ((left, 2),)


@pytest.mark.parametrize("leaders", [[], ["winner"] * 3])
def test_bounded_top_k_handles_clear_leaders_and_large_categorical_ties(leaders):
    labels = [f"label_{i:03d}" for i in range(500)]
    values = labels[::-1] + leaders + [None]
    categories = ["unused", "winner", *labels]
    expected = (("winner", 3),) if leaders else ()
    expected += tuple((label, 1) for label in labels[:4 - len(expected)])
    for rows, order in ((values, categories), (values[::-1], categories[::-1])):
        frame = pd.DataFrame({"category": pd.Categorical(rows, categories=order)})
        stats = inspect(frame, options=ProfileOptions(top_k=4)).columns[0].statistics
        assert stats.top_values == expected
        assert stats.top_frequencies == tuple(count for _, count in expected)


def test_top_k_ignores_many_unused_categories_at_zero_frequency_cutoff():
    frame = pd.DataFrame({"category": pd.Categorical(
        ["second", None, "first"], categories=[*(str(i) for i in range(80)), "second", "first"],
    )})
    stats = inspect(frame).columns[0].statistics
    assert stats.top_values == (("first", 1), ("second", 1))
    assert stats.top_frequencies == (1, 1)


def test_temporal_gap_warning_is_serialized():
    context, _, warnings = _temporal_warnings(["2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"])
    assert warnings[0] in context.warnings
    serialized = context.to_dict()
    assert any(warning["code"] == "potential_temporal_gap" for warning in serialized["warnings"])


def test_many_temporal_gaps_are_bounded_and_omissions_reported():
    dates = pd.date_range("2026-01-01", periods=400, freq="D")
    frame = pd.DataFrame({"event_time": dates[[i for i in range(400) if i % 4 != 3]]})
    context = inspect(frame, options=ProfileOptions(max_gap_warnings=3))
    warnings = [w for w in context.columns[0].warnings if w.code == "potential_temporal_gap"]
    assert len(warnings) == 3
    assert warnings[-1].evidence[0].details["gaps_omitted"] == 96
    assert "96 additional gap(s) omitted" in warnings[-1].message
    assert len([w for w in context.warnings if w.code == "potential_temporal_gap"]) == 3


def test_small_number_of_temporal_gaps_keeps_detailed_output():
    dates = pd.date_range("2026-01-01", periods=12, freq="D")
    frame = pd.DataFrame({"event_time": dates[[i for i in range(12) if i not in (3, 7)]]})
    warnings = [w for w in inspect(frame).columns[0].warnings if w.code == "potential_temporal_gap"]
    assert len(warnings) == 2
    assert all("gaps_omitted" not in w.evidence[0].details for w in warnings)
    assert all(len(w.evidence[0].details["expected_timestamps"]) == 1 for w in warnings)


def _column(values, name="value"):
    return inspect(pd.DataFrame({name: values})).columns[0]


def test_high_cardinality_text_columns_warn():
    notes = _column([f"note {i} about the order" for i in range(300)], "notes")
    tags = _column([f"tag{i % 150}" for i in range(300)], "tags")
    for column, distinct in ((notes, 300), (tags, 150)):
        assert column.logical_type == "text"
        warning = next(w for w in column.warnings if w.code == "high_cardinality_text")
        assert warning.evidence[0].details["distinct_count"] == distinct
        assert warning.evidence[0].details["non_missing_count"] == 300
        assert warning.evidence[0].details["uniqueness_ratio"] == distinct / 300


def test_high_cardinality_text_ignores_missing_values_in_evidence():
    column = _column([f"comment {i}" for i in range(120)] + [None] * 30, "comment")
    warning = next(w for w in column.warnings if w.code == "high_cardinality_text")
    assert warning.evidence[0].details["non_missing_count"] == 120
    assert warning.evidence[0].details["uniqueness_ratio"] == 1.0


def test_low_cardinality_text_and_identifiers_do_not_warn():
    city = _column([f"city{i % 5}" for i in range(300)], "city")
    assert city.semantic.role == "dimension"
    assert not any(w.code == "high_cardinality_text" for w in city.warnings)
    at_threshold = _column([f"v{i}" for i in range(100)] * 2)
    assert not any(w.code == "high_cardinality_text" for w in at_threshold.warnings)
    order_id = _column([f"ORD-{i:05d}" for i in range(300)], "order_id")
    assert order_id.semantic.role == "identifier"
    assert not any(w.code == "high_cardinality_text" for w in order_id.warnings)


def test_categorical_high_cardinality_warning_is_unchanged():
    group = _column(pd.Series([f"g{i}" for i in range(110)], dtype="category"), "group")
    codes = [w.code for w in group.warnings]
    assert "high_cardinality_category" in codes
    assert "high_cardinality_text" not in codes


def test_sparse_float_column_profiles_and_preserves_dtype():
    amount = pd.arrays.SparseArray([0.0, 1.0, 2.0, 3.0])
    sparse_column = inspect(pd.DataFrame({"amount": amount})).columns[0]
    dense_column = inspect(pd.DataFrame({"amount": [0.0, 1.0, 2.0, 3.0]})).columns[0]

    assert sparse_column.dtype == str(amount.dtype)
    assert sparse_column.logical_type == "numeric"
    assert sparse_column.statistics.mean == 1.5
    assert sparse_column.statistics.median == 1.5
    assert sparse_column.statistics == dense_column.statistics
    assert sparse_column.semantic == dense_column.semantic


@pytest.mark.parametrize("values", [
    pd.arrays.SparseArray([0.0, 1.0, 2.0, 3.0], fill_value=0.0),
    pd.arrays.SparseArray([0.0, None, 2.0, 3.0]),
    pd.arrays.SparseArray([0.0, float("nan"), 2.0, 3.0], fill_value=0.0),
    pd.arrays.SparseArray([0, 1, 2, 3]),
    pd.arrays.SparseArray([1, -2, 1, 4, 1], fill_value=1),
    pd.arrays.SparseArray([0.0, float("inf"), 2.0, 3.0], fill_value=0.0),
    pd.Series([0.0, 1.0, 2.0, 3.0], dtype="float64"),
    pd.Series([0, 1, 2, 3], dtype="int64"),
])
def test_sparse_and_dense_numeric_columns_match_dense_statistics(values):
    series = pd.Series(values)
    dense_series = series.sparse.to_dense() if isinstance(series.dtype, pd.SparseDtype) else series
    actual = inspect(pd.DataFrame({"amount": series})).columns[0]
    expected = inspect(pd.DataFrame({"amount": dense_series})).columns[0]

    assert actual.dtype == str(series.dtype)
    assert actual.logical_type == "numeric"
    assert actual.statistics == expected.statistics
    assert actual.semantic == expected.semantic
