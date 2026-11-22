import pandas as pd

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


def test_composite_null_handling_preserves_drop_duplicates_semantics():
    unique_pairs = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": [None, "A", None, "B"],
    })
    duplicate_null_pair = pd.DataFrame({
        "order_id": [1001, 1001, 1002, 1002],
        "product_id": [None, None, "A", "B"],
    })
    assert ("order_id", "product_id") in {key.columns for key in _keys(unique_pairs)}
    assert ("order_id", "product_id") not in {key.columns for key in _keys(duplicate_null_pair)}


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
