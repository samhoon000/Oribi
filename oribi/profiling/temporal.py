"""Conservative observations of calendar values and temporal cadence."""

import re
from collections import Counter
from dataclasses import replace
from datetime import date, datetime

import pandas as pd

from ..models.evidence import Evidence, Warning

_MIN_UNIQUE_TIMESTAMPS = 4
_MIN_CADENCE_INTERVALS = 2
_MIN_CADENCE_SHARE = 0.6
_ISO_CALENDAR_VALUE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"(?:[T ][0-9]{2}:[0-9]{2}(?::[0-9]{2}(?P<fraction>\.[0-9]{1,9})?)?"
    r"(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?)?"
)


def _valid_iso_calendar_value(value: str) -> bool:
    match = _ISO_CALENDAR_VALUE.fullmatch(value)
    if match is None:
        return False
    # The regex validates fractions; remove them for Python 3.10 compatibility.
    if match.group("fraction"):
        start, end = match.span("fraction")
        value = value[:start] + value[end:]
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


def date_like_evidence(values: pd.Series, kind: str, distinct_count: int) -> Evidence | None:
    """Recognize calendar values without converting their stored representation."""
    if values.empty:
        return None
    details = {"observed_count": len(values), "distinct_count": distinct_count}
    first = values.iloc[0]
    if kind == "string":
        if distinct_count < 2 or not _valid_iso_calendar_value(first):
            return None
        try:
            distinct_values = values.unique()
        except (TypeError, ValueError):
            return None
        if all(_valid_iso_calendar_value(value) for value in distinct_values):
            return Evidence("iso_temporal_values", "All distinct non-missing strings are ISO dates or timestamps", details)
    elif kind in {"date", "datetime", "datetime64"} or (
        isinstance(first, date) and all(isinstance(value, date) for value in values)
    ):
        return Evidence("temporal_object_values", "All non-missing values are date or datetime objects", details)
    return None


def temporal_gap_warnings(series: pd.Series, column_name: str, *, max_examples: int = 3,
                          max_warnings: int = 10) -> tuple[Warning, ...]:
    """Report possible gaps only when a dominant fixed cadence is supported."""
    timestamps = pd.DatetimeIndex(series.dropna().drop_duplicates().sort_values())
    if len(timestamps) < _MIN_UNIQUE_TIMESTAMPS:
        return ()

    try:
        deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    except (pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta, OverflowError):
        return ()
    ranked = Counter(deltas).most_common()
    cadence, dominant_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == dominant_count:
        return ()
    if dominant_count < _MIN_CADENCE_INTERVALS or dominant_count / len(deltas) < _MIN_CADENCE_SHARE:
        return ()

    steps_by_interval: list[int] = []
    for delta in deltas:
        steps = delta / cadence
        if steps < 1 or steps != int(steps):
            return ()
        steps_by_interval.append(int(steps))
    if not any(steps > 1 for steps in steps_by_interval):
        return ()

    cadence_text = _format_timedelta(cadence)
    warnings: list[Warning] = []
    gap_count = 0
    for index, steps in enumerate(steps_by_interval):
        missing_count = steps - 1
        if not missing_count:
            continue
        gap_count += 1
        if gap_count > max_warnings:
            continue
        before, after = timestamps[index], timestamps[index + 1]
        examples = [before + cadence * step for step in range(1, min(steps, max_examples + 1))]
        gap_start = before + cadence
        gap_end = after - cadence
        evidence = Evidence(
            "temporal_gap",
            f"{missing_count} timestamp(s) may be absent relative to an inferred {cadence_text} cadence",
            {
                "cadence": cadence_text,
                "gap_start": gap_start.isoformat(),
                "gap_end": gap_end.isoformat(),
                "missing_intervals": missing_count,
                "expected_timestamps": [value.isoformat() for value in examples],
                "examples_omitted": missing_count - len(examples),
                "observed_before": before.isoformat(),
                "observed_after": after.isoformat(),
            },
        )
        warnings.append(Warning(
            "potential_temporal_gap", "low", "Potential temporal gap",
            f"{column_name} has a possible temporal gap relative to an inferred {cadence_text} cadence.",
            (evidence,),
            "The interval may reflect missing observations or a normal pause in data collection.",
            "Confirm the expected sampling schedule before treating the interval as missing data.",
        ))
    if gap_count > max_warnings and warnings:
        last = warnings[-1]
        evidence = last.evidence[0]
        omitted = gap_count - max_warnings
        warnings[-1] = replace(
            last,
            message=f"{last.message} {omitted} additional gap(s) omitted.",
            evidence=(replace(evidence, details={**evidence.details, "gaps_omitted": omitted}),),
        )
    return tuple(warnings)


def _format_timedelta(value: pd.Timedelta) -> str:
    """Format common fixed cadence durations consistently."""
    if value.microseconds or value.nanoseconds:
        return str(value)
    seconds = value.days * 86400 + value.seconds
    for size, unit in ((86400, "day"), (3600, "hour"), (60, "minute"), (1, "second")):
        if seconds % size == 0:
            amount = seconds // size
            return f"{amount} {unit}{'' if amount == 1 else 's'}"
    return str(value)
