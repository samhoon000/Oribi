import pandas as pd
import pytest

from oribi import inspect

ZONE = "America/New_York"


def _warnings(timestamps):
    context = inspect(pd.DataFrame({"recorded_at": timestamps}))
    return [warning for warning in context.warnings if warning.code == "potential_temporal_gap"]


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        ("2025-03-06", "2025-03-10T09:00:00-04:00"),
        ("2025-10-30", "2025-11-03T09:00:00-05:00"),
    ],
)
def test_daily_calendar_gap_across_dst(start, expected):
    times = pd.date_range(f"{start} 09:00", periods=7, freq="D", tz=ZONE)
    assert _warnings(times) == []
    context = inspect(pd.DataFrame({"recorded_at": times.delete(4)}))
    private = context.to_dict(include_values=True)
    (warning,) = [w for w in private["warnings"] if w["code"] == "potential_temporal_gap"]
    details = warning["evidence"][0]["details"]
    assert expected not in context.to_json()
    assert details["missing_intervals"] == 1
    assert details["expected_timestamps"] == [expected]
    assert details["gap_start"] == details["gap_end"] == expected


@pytest.mark.parametrize("transition", ["2025-03-09 02:30", "2025-11-02 01:30"])
def test_invalid_local_gap_time_is_not_reported(transition):
    missing = pd.Timestamp(transition)
    local = pd.date_range(missing - pd.Timedelta(days=3), periods=7, freq="D").delete(3)
    assert _warnings(local.tz_localize(ZONE)) == []


def test_varying_local_times_disqualify_calendar_cadence():
    local = pd.DatetimeIndex([
        "2025-03-06 09:00", "2025-03-07 09:00", "2025-03-08 09:00",
        "2025-03-09 09:00", "2025-03-11 09:01", "2025-03-12 09:01",
    ])
    assert _warnings(local.tz_localize(ZONE)) == []


def test_weak_daily_support_disqualifies_calendar_cadence():
    local = pd.DatetimeIndex([
        "2025-03-06 09:00", "2025-03-07 09:00", "2025-03-09 09:00",
        "2025-03-10 09:00", "2025-03-13 09:00",
    ])
    assert _warnings(local.tz_localize(ZONE)) == []


def test_elapsed_cadence_can_cross_dst_at_varying_local_hours():
    timestamps = pd.date_range("2025-03-06 14:00", periods=7, freq="24h", tz="UTC")
    (warning,) = _warnings(timestamps.delete(4).tz_convert(ZONE))
    assert warning.evidence[0].details["expected_timestamps"] == ("2025-03-10T10:00:00-04:00",)


def test_calendar_cadence_accepts_sixty_percent_daily_support():
    times = pd.date_range("2025-03-06 09:00", periods=8, freq="D", tz=ZONE)
    warnings = _warnings(times.delete([2, 5]))
    assert [w.evidence[0].details["expected_timestamps"] for w in warnings] == [
        ("2025-03-08T09:00:00-05:00",),
        ("2025-03-11T09:00:00-04:00",),
    ]
