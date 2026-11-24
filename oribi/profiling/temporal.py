"""Conservative observations of calendar values and temporal cadence."""

import re
from collections import Counter
from dataclasses import replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from ..models.evidence import Evidence, Warning

_MIN_UNIQUE_TIMESTAMPS = 4
_MIN_CADENCE_INTERVALS = 2
_MIN_CADENCE_SHARE = 0.6
_DAY = pd.Timedelta(days=1)
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


def _daily_calendar_steps(timestamps: pd.DatetimeIndex) -> list[int] | None:
    """Require a common local clock time and dominant one-day spacing."""
    local = timestamps.tz_localize(None)
    steps = []
    for before, after in zip(local[:-1], local[1:], strict=True):
        days, remainder = divmod(after - before, _DAY)
        if days < 1 or remainder:
            return None
        steps.append(days)
    daily_count = steps.count(1)
    if daily_count < _MIN_CADENCE_INTERVALS or daily_count / len(steps) < _MIN_CADENCE_SHARE:
        return None
    return steps


def _fixed_cadence(timestamps: pd.DatetimeIndex) -> tuple[pd.Timedelta, list[int]] | None:
    deltas = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    ranked = Counter(deltas).most_common()
    cadence, dominant_count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == dominant_count:
        return None
    if dominant_count < _MIN_CADENCE_INTERVALS or dominant_count / len(deltas) < _MIN_CADENCE_SHARE:
        return None

    steps_by_interval: list[int] = []
    for delta in deltas:
        # True division rounds large ratios and can hide a nonzero remainder.
        steps, remainder = divmod(delta, cadence)
        if steps < 1 or remainder != pd.Timedelta(0):
            return None
        steps_by_interval.append(steps)
    return cadence, steps_by_interval


def temporal_gap_warnings(series: pd.Series, column_name: str, *, max_examples: int = 3,
                          max_warnings: int = 10) -> tuple[Warning, ...]:
    """Report possible gaps on a supported local daily or fixed elapsed cadence.

    Calendar counts describe nominal local slots. Skip gaps with ambiguous or
    nonexistent boundaries and omit any such timestamps from the examples.
    """
    timestamps = pd.DatetimeIndex(series.dropna().drop_duplicates().sort_values())
    if len(timestamps) < _MIN_UNIQUE_TIMESTAMPS:
        return ()

    timezone = timestamps.tz
    try:
        calendar_steps = _daily_calendar_steps(timestamps) if timezone is not None else None
        if calendar_steps is not None:
            cadence, steps_by_interval = _DAY, calendar_steps
        else:
            fixed = _fixed_cadence(timestamps)
            if fixed is None:
                return ()
            cadence, steps_by_interval = fixed
    except (pd.errors.OutOfBoundsDatetime, pd.errors.OutOfBoundsTimedelta, OverflowError):
        return ()
    if not any(steps > 1 for steps in steps_by_interval):
        return ()

    cadence_text = _format_timedelta(cadence)
    warnings: list[Warning] = []
    gap_count = 0
    for index, steps in enumerate(steps_by_interval):
        missing_count = steps - 1
        if not missing_count:
            continue
        before, after = timestamps[index], timestamps[index + 1]
        local_before, local_after = before, after
        if calendar_steps is not None:
            local_before, local_after = before.tz_localize(None), after.tz_localize(None)
        gap_start = local_before + cadence
        gap_end = local_after - cadence
        if calendar_steps is not None:
            # Use the zone name for pandas versions with limited ZoneInfo localization.
            zone = timezone.key if isinstance(timezone, ZoneInfo) else timezone
            bounds = pd.DatetimeIndex([gap_start, gap_end]).tz_localize(
                zone, ambiguous="NaT", nonexistent="NaT",
            )
            if bounds.hasnans:
                continue
            gap_start, gap_end = bounds[0], bounds[1]

        gap_count += 1
        if gap_count > max_warnings:
            continue
        examples = [
            local_before + cadence * step for step in range(1, min(steps, max_examples + 1))
        ]
        if calendar_steps is not None and examples:
            examples = list(pd.DatetimeIndex(examples).tz_localize(
                zone, ambiguous="NaT", nonexistent="NaT",
            ).dropna())
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
