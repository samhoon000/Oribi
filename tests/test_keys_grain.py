import pandas as pd
import pytest

from oribi import ProfileOptions, inspect


def test_composite_key_and_grain(sales_frame):
    context = inspect(sales_frame)
    assert context.grain is not None
    assert context.grain.status == "estimated"
    assert len(context.grain.columns) >= 1
    assert context.candidate_keys


def test_uncertain_grain_is_allowed():
    context = inspect(pd.DataFrame({"group": ["a", "a", "a"], "value": [1, 1, 1]}))
    assert context.grain.status == "undetermined"


def _keys(frame, options=None):
    return inspect(frame, options=options).candidate_keys


def test_all_qualifying_single_column_keys_are_retained():
    frame = pd.DataFrame({
        "line_id": [1, 2, 3, 4],
        "external_id": ["a", "b", "c", "d"],
        "group": ["x", "x", "y", "y"],
    })
    keys = _keys(frame)
    assert {key.columns for key in keys if key.kind == "single"} == {
        ("line_id",), ("external_id",),
    }


def test_single_column_candidate_retention_has_no_old_top_ten_cutoff():
    frame = pd.DataFrame({f"record_{index}_id": range(12) for index in range(12)})
    single_keys = [key for key in _keys(frame) if key.kind == "single"]
    assert len(single_keys) == 12


def test_single_and_composite_candidates_are_both_retained():
    frame = pd.DataFrame({
        "line_id": [1, 2, 3, 4],
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": ["A", "B", "A", "B"],
    })
    context = inspect(frame)
    assert {key.columns for key in context.candidate_keys} == {
        ("line_id",), ("order_id", "product_id"),
    }
    assert context.grain.columns == ("line_id",)


def test_composite_candidate_is_detected_without_single_key():
    frame = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": ["A", "B", "A", "B"],
    })
    context = inspect(frame)
    assert [key.columns for key in context.candidate_keys] == [("order_id", "product_id")]
    assert context.grain.columns == ("order_id", "product_id")


def test_repeated_elapsed_durations_cannot_form_a_composite_key_with_a_flag():
    frame = pd.DataFrame({
        "elapsed": pd.to_timedelta([0, 0, 1, 1], unit="s"),
        "flag": [True, False, True, False],
    })
    context = inspect(frame)

    assert context.candidate_keys == ()
    assert context.grain.status == "undetermined"
    assert context.grain.columns == ()
    statistics = context.get_column("elapsed").statistics
    assert statistics.distinct_count == 2
    assert statistics.missing_count == 0
    assert dict(statistics.top_values) == {pd.Timedelta(0): 2, pd.Timedelta(seconds=1): 2}


def test_elapsed_durations_do_not_displace_a_datetime_composite_key():
    frame = pd.DataFrame({
        "elapsed": pd.to_timedelta(range(4), unit="s"),
        "device_id": ["A", "B", "A", "B"],
        "measured_at": pd.date_range("2025-01-01", periods=2, tz="UTC").repeat(2),
    })
    context = inspect(frame, options=ProfileOptions(max_key_columns=2))

    assert [(key.columns, key.verified_unique) for key in context.candidate_keys] == [
        (("device_id", "measured_at"), True),
    ]
    assert context.grain.status == "estimated"
    assert context.grain.columns == ("device_id", "measured_at")


@pytest.fixture
def near_unique_order_frame():
    return pd.DataFrame({
        "order_id": [*range(99), 0],
        "warehouse": ["west", "east"] * 49 + ["west", "north"],
    })


@pytest.mark.parametrize("columns", [["order_id", "warehouse"], ["warehouse", "order_id"]],
                         ids=["forward", "reversed"])
def test_near_unique_single_can_form_verified_grain_at_row_cap(near_unique_order_frame, columns):
    context = inspect(near_unique_order_frame[columns], options=ProfileOptions(max_exact_rows=100))
    assert [(key.columns, key.verified_unique) for key in context.candidate_keys] == [
        (("order_id",), False),
        (("order_id", "warehouse"), True),
    ]
    assert context.grain.status == "estimated"
    assert context.grain.columns == ("order_id", "warehouse")


def test_pair_with_near_unique_single_is_skipped_above_row_cap(near_unique_order_frame):
    context = inspect(near_unique_order_frame, options=ProfileOptions(max_exact_rows=99))
    assert [(key.columns, key.verified_unique) for key in context.candidate_keys] == [
        (("order_id",), False),
    ]
    assert context.grain.status == "undetermined"
    assert context.grain.evidence[0].code == "key_search_bounded"


def test_bounded_pair_search_prioritizes_near_unique_over_redundant_columns(near_unique_order_frame):
    frame = near_unique_order_frame.assign(
        aisle=["front", "back"] * 50,
        line_id=range(100),
        zone=["red", "blue"] * 49 + ["red", "green"],
    )
    # Prune line_id before the cap, rank warehouse ahead of aisle by distinctness,
    # and break the warehouse/zone tie by name rather than input order.
    context = inspect(frame.iloc[:, ::-1], options=ProfileOptions(max_key_columns=2))
    assert [(key.columns, key.verified_unique) for key in context.candidate_keys] == [
        (("line_id",), True),
        (("order_id",), False),
        (("order_id", "warehouse"), True),
    ]
    assert context.grain.columns == ("line_id",)


def test_candidate_set_and_ranking_ignore_dataframe_column_order():
    frame = pd.DataFrame({
        "line_id": [1, 2, 3, 4],
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": ["A", "B", "A", "B"],
    })
    first = inspect(frame)
    second = inspect(frame[["product_id", "line_id", "order_id"]])
    first_ranked = [(key.columns, key.kind, key.uniqueness_rate) for key in first.candidate_keys]
    second_ranked = [(key.columns, key.kind, key.uniqueness_rate) for key in second.candidate_keys]
    assert first_ranked == second_ranked
    assert first.grain == second.grain


def test_duplicate_composite_values_are_not_a_candidate():
    frame = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": ["A", "A", "B", "B"],
    })
    assert not [key for key in _keys(frame) if key.kind == "composite"]


@pytest.mark.parametrize("case", ["incomplete", "duplicate", "mixed", "unique"])
def test_composite_key_counts_missing_and_collisions_separately(case):
    from itertools import product

    frame = pd.DataFrame(product("ABCD", "0123456789"), columns=["batch", "slot"])
    if case == "incomplete":
        frame.loc[[0, 1], "batch"] = None
        frame.loc[1, "slot"] = "0"
    elif case == "duplicate":
        frame.loc[1, "slot"] = "0"
    elif case == "mixed":
        frame.loc[0, "batch"] = None
        frame.loc[2, "slot"] = "1"

    context = inspect(frame)
    key = next(key for key in context.candidate_keys if key.columns == ("batch", "slot"))
    expected = {"incomplete": (0.95, 0, 2, False), "duplicate": (0.975, 1, 0, False),
                "mixed": (0.95, 1, 1, False), "unique": (1.0, 0, 0, True)}[case]
    assert (key.uniqueness_rate, key.duplicate_count, key.missing_count, key.verified_unique) == expected
    details = key.evidence[0].details
    assert (details["uniqueness_rate"], details["duplicate_count"], details["missing_count"],
            details["verified_unique"]) == expected
    assert details["distinct_pairs"] == 40 * expected[0]
    assert context.grain.status == ("estimated" if case == "unique" else "undetermined")


def test_composite_search_is_bounded_by_max_key_columns():
    frame = pd.DataFrame({
        "a_id": ["a1", "a1", "a2", "a2"],
        "b_id": ["b1", "b2", "b1", "b2"],
        "c_id": ["c1", "c1", "c2", "c2"],
        "d_id": ["d1", "d2", "d1", "d2"],
    })
    composites = [key.columns for key in _keys(frame, ProfileOptions(max_key_columns=2)) if key.kind == "composite"]
    assert composites == [("a_id", "b_id")]


def test_composite_search_runs_below_and_at_limit_but_is_skipped_above():
    frame = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1002, 1003],
        "product_id": ["A", "B", "A", "B", "A"],
    })

    below = inspect(frame.iloc[:3], options=ProfileOptions(max_exact_rows=4))
    at_limit = inspect(frame.iloc[:4], options=ProfileOptions(max_exact_rows=4))
    above = inspect(frame, options=ProfileOptions(max_exact_rows=4))

    assert [key.columns for key in below.candidate_keys] == [("order_id", "product_id")]
    assert [key.columns for key in at_limit.candidate_keys] == [("order_id", "product_id")]
    assert above.candidate_keys == ()
    assert above.grain.evidence[0].code == "key_search_bounded"


def test_empty_and_tiny_frames_are_safe():
    empty = inspect(pd.DataFrame({"order_id": pd.Series(dtype="int64"), "product_id": pd.Series(dtype="object")}))
    tiny = inspect(pd.DataFrame({"order_id": [1], "product_id": ["A"]}))
    assert empty.candidate_keys == ()
    assert empty.grain.status == "undetermined"
    assert tiny.candidate_keys


def test_tiny_unique_column_does_not_receive_near_certain_grain_confidence():
    context = inspect(pd.DataFrame({"record_id": [1, 2, 3]}))
    assert context.grain.status == "estimated"
    assert context.grain.confidence < 0.7


def test_stringified_dict_column_cannot_verify_key_or_establish_grain():
    frame = pd.DataFrame({
        "payload": [
            {"a": 1, "b": 2},
            {"b": 2, "a": 1},
            {"a": 2, "b": 2},
            {"b": 3, "a": 1},
        ]
    })
    context = inspect(frame)
    assert context.profile.duplicate_row_count == 1
    assert not any(key.columns == ("payload",) and key.verified_unique for key in context.candidate_keys)
    assert context.grain.status == "undetermined"
    assert context.grain.columns == ()


def test_unique_scalar_key_still_verifies_alongside_unhashable_dict_column():
    frame = pd.DataFrame({
        "record_id": [1, 2, 3, 4],
        "payload": [
            {"a": 1, "b": 2},
            {"b": 2, "a": 1},
            {"a": 2, "b": 2},
            {"b": 3, "a": 1},
        ],
    })
    context = inspect(frame)
    assert [(key.columns, key.verified_unique) for key in context.candidate_keys] == [
        (("record_id",), True)
    ]
    assert context.grain.status == "estimated"
    assert context.grain.columns == ("record_id",)


def test_independently_unique_native_timestamps_leave_grain_undetermined():
    dates = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame({"opened_at": dates, "closed_at": dates + pd.Timedelta(hours=4)})
    forward = inspect(frame)
    reverse = inspect(frame[["closed_at", "opened_at"]])

    assert forward.grain == reverse.grain
    assert forward.grain.status == "undetermined"
    assert forward.grain.columns == ()
    assert forward.grain.confidence == 0.0
    evidence, = forward.grain.evidence
    assert evidence.code == "ambiguous_temporal_grain"
    assert tuple(evidence.details["columns"]) == ("closed_at", "opened_at")
    assert evidence.statement in forward.to_markdown()
    assert evidence.code in forward.to_json()
    assert "Unknown" in forward.to_context()
    assert forward.candidate_keys[0].verified_unique
    assert inspect(frame.assign(region=list("abcdefghij"))).grain == forward.grain


def test_datetime_id_name_does_not_establish_entity_grain():
    frame = pd.DataFrame({
        "settlement_id": pd.date_range("2025-01-01", periods=4),
        "posted_at": pd.date_range("2025-02-01", periods=4),
    })
    context = inspect(frame)
    assert [col.semantic.role for col in context.columns] == ["temporal", "temporal"]
    assert not any(len(key.columns) == 1 and key.verified_unique for key in context.candidate_keys)
    assert context.grain.status == "undetermined"
    assert context.grain.evidence[0].code == "ambiguous_temporal_grain"

    keyed = inspect(frame.assign(order_id=range(4)))
    assert keyed.columns[-1].semantic.role == "identifier"
    assert keyed.grain.columns == ("order_id",)


def test_single_unique_native_timestamp_supports_grain():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    grain = inspect(pd.DataFrame({"opened_at": dates})).grain
    assert grain.status == "estimated"
    assert grain.columns == ("opened_at",)
    assert grain.evidence[0].code == "unique_temporal_observations"


def test_identifier_resolves_ambiguity_between_unique_timestamps():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame({
        "opened_at": dates,
        "closed_at": dates + pd.Timedelta(hours=4),
        "record_id": range(10),
    })
    context = inspect(frame)
    assert context.grain.columns == ("record_id",)
    assert any(key.columns == ("record_id",) and key.verified_unique for key in context.candidate_keys)


def test_joint_entity_time_grain_resolves_ambiguity_between_unique_timestamps():
    dates = pd.date_range("2025-01-01", periods=10, freq="h")
    frame = pd.DataFrame({
        "opened_at": dates,
        "closed_at": dates + pd.Timedelta(hours=4),
        "measured_at": dates.repeat(2)[:10],
        "device_id": ["A", "B"] * 5,
    })
    context = inspect(frame)
    assert context.grain.status == "estimated"
    assert context.grain.columns == ("device_id", "measured_at")


def test_repeated_and_missing_timestamps_do_not_count_as_individually_unique():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    frame = pd.DataFrame({
        "opened_at": dates,
        "repeated_at": dates.repeat(2)[:10],
        "missing_at": pd.Series(dates).mask(pd.RangeIndex(10) == 0),
    })
    for other in ("repeated_at", "missing_at"):
        grain = inspect(frame[["opened_at", other]]).grain
        assert grain.status == "estimated"
        assert all(evidence.code != "ambiguous_temporal_grain" for evidence in grain.evidence)

def test_unhashable_column_does_not_displace_verified_composite_key():
    frame = pd.DataFrame({
        "order_id": ["a", "a", "b", "b", "c", "c"],
        "product_id": ["x", "y", "z", "x", "y", "z"],
        "metadata": pd.Series([[1], "[1]", [2], "[2]", [1], "[1]"], dtype=object),
    })

    context = inspect(frame, options=ProfileOptions(max_key_columns=2))

    assert any(
        key.columns == ("order_id", "product_id") and key.verified_unique
        for key in context.candidate_keys
    )
    assert context.grain.columns == ("order_id", "product_id")
