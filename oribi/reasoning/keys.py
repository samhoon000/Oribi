"""Candidate-key collection and separate grain selection."""

from collections.abc import Mapping
from itertools import combinations

import pandas as pd

from ..models.evidence import Evidence, GrainContext, KeyCandidate
from ..models.profile import StatisticsContext

_MIN_SINGLE_UNIQUENESS = 0.98
_MAX_SINGLE_MISSING_RATE = 0.02
_MIN_COMPOSITE_UNIQUENESS = 0.95
_SINGLE_KEY_ROLES = {"identifier", "dimension", "categorical", "boolean", "text", "unknown"}
_COMPOSITE_ROLES = _SINGLE_KEY_ROLES | {"temporal"}


def detect_candidate_keys(
    frame: pd.DataFrame,
    names: list[str],
    stats: Mapping[str, StatisticsContext],
    roles: dict[str, str],
    max_columns: int = 8,
    max_exact_rows: int = 100_000,
    exact_distinct: Mapping[str, bool] | None = None,
) -> tuple[KeyCandidate, ...]:
    """Collect qualifying single and bounded pair candidates.

    Pair search considers at most ``max_columns`` non-unique, non-measure
    columns (28 pairs at the default limit of eight). It runs at or below
    ``max_exact_rows`` and retains all qualifying candidates in the searched set.
    """
    row_count = len(frame)
    if row_count == 0:
        return ()
    sample_support = row_count / (row_count + 4)

    single_candidates: list[KeyCandidate] = []
    for name in sorted(names):
        column_stats = stats[name]
        rate = column_stats.distinct_count / row_count
        if (
            (exact_distinct is None or exact_distinct.get(name, True))
            and roles.get(name) in _SINGLE_KEY_ROLES
            and getattr(frame[name].dtype, "kind", None) != "m"
            and rate >= _MIN_SINGLE_UNIQUENESS
            and column_stats.missing_rate <= _MAX_SINGLE_MISSING_RATE
        ):
            duplicate_count = row_count - column_stats.distinct_count - column_stats.missing_count
            verified = duplicate_count == 0 and column_stats.missing_count == 0
            evidence = Evidence(
                "single_column_uniqueness",
                f"{column_stats.distinct_count} distinct values across {row_count} rows",
                {"distinct_count": column_stats.distinct_count, "row_count": row_count,
                 "duplicate_count": duplicate_count, "missing_count": column_stats.missing_count,
                 "verified_unique": verified},
            )
            single_candidates.append(KeyCandidate(
                (name,),
                "single",
                round(rate * sample_support, 2),
                rate,
                (evidence,),
                duplicate_count,
                column_stats.missing_count,
                verified,
            ))

    # A component already qualifying alone adds no useful composite candidate.
    single_columns = {candidate.columns[0] for candidate in single_candidates}
    potential = sorted(
        (
            name for name in names
            if name not in single_columns
            and _is_key_component(roles.get(name))
            and stats[name].distinct_count > 1
        ),
        key=lambda name: (-stats[name].distinct_rate, stats[name].missing_rate, name),
    )[:max_columns]

    pair_candidates: list[KeyCandidate] = []
    if row_count <= max_exact_rows:
        for pair in combinations(potential, 2):
            columns = tuple(sorted(pair))
            try:
                pair_frame = frame[list(columns)]
                distinct_pairs = int(pair_frame.drop_duplicates().shape[0])
                missing_count = int(pair_frame.isna().any(axis=1).sum())
            except (TypeError, ValueError):
                continue
            rate = distinct_pairs / row_count
            if rate >= _MIN_COMPOSITE_UNIQUENESS:
                duplicate_count = row_count - distinct_pairs
                verified = duplicate_count == 0 and missing_count == 0
                confidence = rate * sample_support
                evidence = Evidence(
                    "composite_uniqueness",
                    f"{distinct_pairs} distinct pairs across {row_count} rows",
                    {
                        "distinct_pairs": distinct_pairs,
                        "row_count": row_count,
                        "uniqueness_rate": rate,
                        "duplicate_count": duplicate_count,
                        "missing_count": missing_count,
                        "verified_unique": verified,
                    },
                )
                pair_candidates.append(KeyCandidate(
                    columns, "composite", round(confidence, 2), rate, (evidence,),
                    duplicate_count, missing_count, verified,
                ))

    # Candidate retention does not depend on this ordering. Simpler keys are
    # listed first, with deterministic uniqueness/confidence/name tie-breakers.
    return tuple(sorted(
        (*single_candidates, *pair_candidates),
        key=lambda candidate: (
            len(candidate.columns),
            -candidate.uniqueness_rate,
            -candidate.confidence,
            candidate.columns,
        ),
    ))


def _is_key_component(role: str | None) -> bool:
    """Permit measured, non-measure column roles as composite key components."""
    return role in _COMPOSITE_ROLES


def select_grain(
    candidates: tuple[KeyCandidate, ...],
    roles: dict[str, str] | None = None,
) -> GrainContext:
    """Choose one preferred grain from the retained candidates."""
    candidates = tuple(candidate for candidate in candidates if candidate.verified_unique)
    if not candidates:
        return GrainContext("undetermined", None, (), 0.0, ())

    # Preserve the existing preference for a single-column identifier when
    # available; candidate detection still retains every composite candidate.
    singles = [candidate for candidate in candidates if candidate.kind == "single"]
    composites = [candidate for candidate in candidates if candidate.kind == "composite"]
    temporal_composites = [
        candidate for candidate in composites
        if roles and any(roles.get(name) == "temporal" for name in candidate.columns)
    ]
    has_identifier_single = any(roles and roles.get(candidate.columns[0]) == "identifier" for candidate in singles)
    pool = temporal_composites if temporal_composites and singles and not has_identifier_single else (singles or composites)
    role_cost = {"identifier": 0, "dimension": 1, "categorical": 1, "temporal": 2,
                 "boolean": 3, "text": 3, "unknown": 4}
    best = min(
        pool,
        key=lambda candidate: (
            sum(role_cost.get(roles.get(name, ""), 4) for name in candidate.columns) if roles else 0,
            -candidate.uniqueness_rate,
            -candidate.confidence,
            candidate.columns,
        ),
    )
    label = " + ".join(best.columns)
    return GrainContext("estimated", label, best.columns, best.confidence, best.evidence)


def detect_keys(
    frame: pd.DataFrame,
    names: list[str],
    stats: Mapping[str, StatisticsContext],
    roles: dict[str, str],
    max_columns: int = 8,
    max_exact_rows: int = 100_000,
    exact_distinct: Mapping[str, bool] | None = None,
) -> tuple[tuple[KeyCandidate, ...], GrainContext]:
    """Collect candidate keys, then independently select a grain hypothesis."""
    candidates = detect_candidate_keys(
        frame, names, stats, roles,
        max_columns=max_columns,
        max_exact_rows=max_exact_rows,
        exact_distinct=exact_distinct,
    )
    row_count = len(frame)
    if len(frame) == 0:
        grain = GrainContext(
            "undetermined", None, (), 0.0,
            (Evidence("empty_dataset", "No rows are available to estimate a key or grain"),),
        )
    elif any(candidate.verified_unique for candidate in candidates):
        grain = select_grain(candidates, roles)
        temporal_candidates = [
            name for name in names
            if roles.get(name) == "temporal"
            and stats[name].distinct_count == row_count
            and stats[name].missing_count == 0
        ]
        if (
            len(grain.columns) == 1
            and roles.get(grain.columns[0]) == "unknown"
            and temporal_candidates
        ):
            name = max(temporal_candidates, key=lambda item: (stats[item].distinct_rate, item))
            uniqueness = stats[name].distinct_count / row_count
            grain = GrainContext(
                "estimated",
                name,
                (name,),
                round(uniqueness * row_count / (row_count + 4), 2),
                (Evidence(
                    "unique_temporal_observations",
                    f"{stats[name].distinct_count} distinct temporal values across {row_count} rows",
                    {"distinct_count": stats[name].distinct_count, "row_count": row_count, "uniqueness_rate": uniqueness},
                ),),
            )
    elif row_count and roles:
        temporal_candidates = [
            name for name in names
            if roles.get(name) == "temporal"
            and stats[name].distinct_count == row_count
            and stats[name].missing_count == 0
        ]
        if temporal_candidates:
            name = max(temporal_candidates, key=lambda item: (stats[item].distinct_rate, item))
            uniqueness = stats[name].distinct_count / row_count
            confidence = round(uniqueness * min(1.0, row_count / (row_count + 4)), 2)
            grain = GrainContext(
                "estimated",
                name,
                (name,),
                confidence,
                (Evidence(
                    "unique_temporal_observations",
                    f"{stats[name].distinct_count} distinct temporal values across {row_count} rows",
                    {"distinct_count": stats[name].distinct_count, "row_count": row_count, "uniqueness_rate": uniqueness},
                ),),
            )
        else:
            grain = _undetermined_grain(frame, max_exact_rows, bool(candidates))
    else:
        grain = _undetermined_grain(frame, max_exact_rows, bool(candidates))
    return candidates, grain


def _undetermined_grain(frame: pd.DataFrame, max_exact_rows: int, approximate: bool = False) -> GrainContext:
    row_count = len(frame)
    code = "key_search_bounded" if row_count > max_exact_rows else ("approximate_keys_only" if approximate else "no_strong_key")
    message = (
        "Composite key search was skipped because the DataFrame exceeds max_exact_rows"
        if row_count > max_exact_rows
        else ("Near-unique candidates have observed collisions or missing values" if approximate
              else "No single column or tested column pair was sufficiently unique")
    )
    return GrainContext(
        "undetermined", None, (), 0.0,
        (Evidence(code, message, {"row_count": row_count, "max_exact_rows": max_exact_rows}),),
    )
