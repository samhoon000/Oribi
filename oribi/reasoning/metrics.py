"""Infer evidence-backed analytical metric definitions from dataset context."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models.context import ColumnContext
from ..models.evidence import Evidence, GrainContext, MetricDefinition, RelationshipContext
from .semantic import column_name_words

_MAX_METRICS = 20
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_RESERVED_KEYWORDS = {
    "abort",
    "action",
    "add",
    "after",
    "all",
    "alter",
    "always",
    "analyze",
    "and",
    "any",
    "as",
    "asc",
    "attach",
    "autoincrement",
    "before",
    "begin",
    "between",
    "by",
    "cascade",
    "case",
    "cast",
    "check",
    "collate",
    "column",
    "commit",
    "conflict",
    "constraint",
    "create",
    "cross",
    "current",
    "current_date",
    "current_time",
    "current_timestamp",
    "current_user",
    "database",
    "default",
    "deferrable",
    "deferred",
    "delete",
    "desc",
    "detach",
    "distinct",
    "do",
    "drop",
    "each",
    "else",
    "end",
    "escape",
    "except",
    "exclude",
    "exclusive",
    "exists",
    "explain",
    "fail",
    "false",
    "fetch",
    "filter",
    "first",
    "following",
    "for",
    "foreign",
    "from",
    "full",
    "generated",
    "glob",
    "group",
    "groups",
    "having",
    "if",
    "ignore",
    "immediate",
    "in",
    "index",
    "indexed",
    "initially",
    "inner",
    "insert",
    "instead",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "key",
    "last",
    "lateral",
    "leading",
    "leave",
    "left",
    "like",
    "limit",
    "match",
    "materialized",
    "natural",
    "no",
    "not",
    "nothing",
    "notnull",
    "null",
    "nulls",
    "of",
    "offset",
    "on",
    "only",
    "or",
    "order",
    "others",
    "outer",
    "over",
    "partition",
    "plan",
    "pragma",
    "preceding",
    "primary",
    "query",
    "raise",
    "range",
    "recursive",
    "references",
    "regexp",
    "reindex",
    "release",
    "rename",
    "replace",
    "restrict",
    "returning",
    "right",
    "rollback",
    "row",
    "rows",
    "savepoint",
    "select",
    "set",
    "some",
    "table",
    "temp",
    "temporary",
    "then",
    "ties",
    "to",
    "trailing",
    "transaction",
    "trigger",
    "true",
    "unbounded",
    "union",
    "unique",
    "update",
    "user",
    "using",
    "vacuum",
    "values",
    "view",
    "virtual",
    "when",
    "where",
    "window",
    "with",
    "without",
}
_IDENTIFIER_SUFFIX_TOKENS = {"id", "key", "code", "no", "num", "number"}
_ROW_KEY_STEMS = {"row", "record", "line", "line_item", "entry", "item_line"}
_VALUE_MEASURE_TOKENS = (
    "revenue",
    "sales",
    "amount",
    "total",
    "subtotal",
    "spend",
    "charge",
    "payment",
    "gmv",
    "value",
)
_VALUE_PRIORITY = {token: idx for idx, token in enumerate(_VALUE_MEASURE_TOKENS)}
_NON_VALUE_MEASURE_TOKENS = {
    "tax",
    "discount",
    "fee",
    "cost",
    "profit",
    "margin",
    "qty",
    "quantity",
    "units",
    "items",
    "count",
    "price",
}
_QUANTITY_TOKENS = {"qty", "quantity", "units", "items"}
_REVENUE_TOKENS = ("revenue", "sales")
_KIND_ORDER = {
    "entity_count": 0,
    "measure_total": 1,
    "measure_per_entity": 2,
    "entity_ratio": 3,
    "measure_ratio": 4,
}

_Candidate = tuple[tuple[int, float, str], MetricDefinition]


def _sql_col(name: str) -> str:
    """Format a column name safely as an ANSI SQL identifier."""
    if _SQL_IDENTIFIER.match(name) and name.lower() not in _SQL_RESERVED_KEYWORDS:
        return name
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _entity_stem(name: str) -> tuple[str, bool]:
    """Extract a normalized entity stem and whether a suffix token was stripped."""
    words = column_name_words(name)
    if len(words) >= 2 and words[-1] in _IDENTIFIER_SUFFIX_TOKENS:
        return "_".join(words[:-1]), True
    return "_".join(words) or "entity", False


def _measure_slug(name: str) -> str:
    words = column_name_words(name)
    return "_".join(words) or "measure"


def _pluralize(stem: str) -> str:
    """Return a simple plural form for metric naming (e.g. order -> orders)."""
    if stem.endswith(("ss", "us", "is", "as", "x", "z", "ch", "sh")):
        return f"{stem}es"
    if stem.endswith("s"):
        return stem
    if len(stem) >= 2 and stem.endswith("y") and stem[-2] not in "aeiou":
        return f"{stem[:-1]}ies"
    return f"{stem}s"


def _is_float_column(col: ColumnContext) -> bool:
    dtype_lower = col.dtype.lower()
    if "float" in dtype_lower or "double" in dtype_lower:
        return True
    return any(e.code == "floating_point_type" for e in col.semantic.evidence)


def _sql_sum_numerator(col: ColumnContext) -> str:
    sql_name = _sql_col(col.name)
    if _is_float_column(col):
        return f"SUM({sql_name})"
    return f"1.0 * SUM({sql_name})"


def _is_additive_measure(col: ColumnContext) -> bool:
    return col.semantic.role == "measure" and any(
        rec.operation == "SUM" and rec.disposition == "recommended" for rec in col.recommendations
    )


def _identifier_has_name_hint(col: ColumnContext) -> bool:
    return any(e.code == "name_identifier_hint" for e in col.semantic.evidence)


def _measure_has_name_hint(col: ColumnContext) -> bool:
    return any(e.code == "name_measure_hint" for e in col.semantic.evidence)


def _value_measure_rank(col: ColumnContext) -> int | None:
    words = set(column_name_words(col.name))
    if not words.isdisjoint(_NON_VALUE_MEASURE_TOKENS):
        return None
    matches = [_VALUE_PRIORITY[token] for token in words if token in _VALUE_PRIORITY]
    return min(matches) if matches else None


def _observed_count(col: ColumnContext, row_count: int) -> int:
    if col.cardinality is not None:
        return col.cardinality.observed_count
    return max(0, row_count - col.statistics.missing_count)


def _coverage(col: ColumnContext, row_count: int) -> float:
    if row_count <= 0:
        return 0.0
    return _observed_count(col, row_count) / row_count


def _sample_support(observed: int) -> float:
    if observed <= 0:
        return 0.0
    return 0.7 + 0.3 * observed / (observed + 12)


def _cte_alias(prefix: str, col_name: str) -> str:
    """Return a safe unquoted ANSI SQL identifier for a grain CTE."""
    stem, _ = _entity_stem(col_name)
    return f"{prefix}_{stem}"


def _dedup_caveat(
    m_col: ColumnContext,
    det_col: ColumnContext,
    *,
    grain_mismatch_with: str | None = None,
) -> str:
    if grain_mismatch_with is not None:
        return (
            f"{m_col.name} is constant at {det_col.name} grain rather than "
            f"{grain_mismatch_with} grain; SQL deduplicates {m_col.name} to "
            f"{det_col.name} grain before aggregating."
        )
    return (
        f"{m_col.name} is constant within each {det_col.name} across repeated rows; "
        f"SQL deduplicates to {det_col.name} grain to avoid double-counting "
        "(verify if identical line-item values are coincidental)."
    )


def _dedup_evidence(m_col: ColumnContext, det_col: ColumnContext, row_count: int) -> Evidence:
    return Evidence(
        "measure_constant_per_entity",
        f"{m_col.name} takes one distinct value per {det_col.name} across repeated rows, so SQL deduplicates to {det_col.name} grain",
        {
            "entity_column": det_col.name,
            "entity_distinct_count": det_col.statistics.distinct_count,
            "observed_count": _observed_count(m_col, row_count),
        },
    )


def _pick_measure_determinant(
    m_col: ColumnContext,
    constant_measures: frozenset[tuple[str, str]],
    col_by_name: dict[str, ColumnContext],
    *,
    preferred_col: str | None = None,
    grain_cols: set[str] | frozenset[str] = frozenset(),
) -> ColumnContext | None:
    determinants = [
        col_by_name[entity_name]
        for entity_name, measure_name in constant_measures
        if measure_name == m_col.name and entity_name in col_by_name
    ]
    if not determinants:
        return None
    return min(
        determinants,
        key=lambda col: (
            col.statistics.distinct_count,
            0 if col.name == preferred_col else 1,
            0 if _entity_stem(col.name)[0] == "order" else 1,
            0 if col.name in grain_cols else 1,
            -col.semantic.confidence,
            col.name,
        ),
    )


def _merge_caveats(base: tuple[str, ...], *extra: str) -> tuple[str, ...]:
    merged = list(base)
    seen = set(base)
    for item in extra:
        if item and item not in seen:
            seen.add(item)
            merged.append(item)
    return tuple(merged)


def _column_caveats(
    cols: Sequence[ColumnContext],
    *,
    includes_sum: bool,
    has_duplicate_rows: bool,
) -> tuple[str, ...]:
    caveats: list[str] = []
    seen: set[str] = set()
    for col in cols:
        for warning in col.warnings:
            if warning.code == "high_missingness":
                msg = (
                    f"{col.name} is missing {col.statistics.missing_rate:.0%} of values; "
                    "aggregates exclude NULLs."
                )
            elif warning.code == "negative_measure_values" and includes_sum:
                msg = (
                    f"{col.name} contains negative values; "
                    "confirm refunds or adjustments before summing."
                )
            elif warning.code == "high_skew" and includes_sum:
                msg = (
                    f"{col.name} is highly skewed; "
                    "consider pairing averages with median or quantiles."
                )
            else:
                continue
            if msg not in seen:
                seen.add(msg)
                caveats.append(msg)
    if has_duplicate_rows and includes_sum:
        dup_msg = "Dataset contains duplicate rows; verify grain before summing."
        if dup_msg not in seen:
            caveats.append(dup_msg)
    return tuple(caveats)


def _unique_name(desired: str, seen: set[str]) -> str:
    if desired not in seen:
        seen.add(desired)
        return desired
    suffix = 2
    while f"{desired}_{suffix}" in seen:
        suffix += 1
    unique = f"{desired}_{suffix}"
    seen.add(unique)
    return unique


def _pair_measure_confidence(
    numerator: ColumnContext,
    denominator: ColumnContext,
    row_count: int,
) -> float:
    avg_sem = (numerator.semantic.confidence + denominator.semantic.confidence) / 2.0
    min_cov = min(_coverage(numerator, row_count), _coverage(denominator, row_count))
    gen = _sample_support(
        min(_observed_count(numerator, row_count), _observed_count(denominator, row_count))
    )
    return round(min(0.98, 0.45 * avg_sem + 0.25 * min_cov + 0.20 * gen + 0.10), 2)


def _infer_entity_counts(
    identifiers: Sequence[ColumnContext],
    *,
    grain_cols: set[str],
    grain_confidence: float,
    one_to_many_cols: set[str],
    row_count: int,
    has_duplicate_rows: bool,
) -> list[_Candidate]:
    results: list[_Candidate] = []
    for id_col in identifiers:
        observed = _observed_count(id_col, row_count)
        stem, _ = _entity_stem(id_col.name)
        name = "distinct_id_count" if stem == "id" else f"{stem}_count"
        label_stem = "distinct id" if stem == "id" else stem.replace("_", " ")
        label = f"{label_stem.capitalize()} count"
        repeats = id_col.statistics.distinct_count < observed
        in_grain = id_col.name in grain_cols
        in_hierarchy = id_col.name in one_to_many_cols
        structural = 1.0 if (in_grain or in_hierarchy or repeats) else 0.75
        confidence = round(
            min(
                0.98,
                0.45 * id_col.semantic.confidence
                + 0.25 * _coverage(id_col, row_count)
                + 0.20 * _sample_support(observed)
                + 0.10 * structural,
            ),
            2,
        )
        evidence = [
            Evidence(
                "identifier_entity_role",
                f"{id_col.name} has an identifier role with {id_col.statistics.distinct_count:,} distinct values",
                {
                    "distinct_count": id_col.statistics.distinct_count,
                    "observed_count": observed,
                    "semantic_confidence": id_col.semantic.confidence,
                },
            )
        ]
        if repeats:
            evidence.append(
                Evidence(
                    "repeated_entity_observations",
                    f"{id_col.name} repeats across rows ({id_col.statistics.distinct_count:,} distinct across {observed:,} observed rows)",
                    {
                        "distinct_count": id_col.statistics.distinct_count,
                        "observed_count": observed,
                    },
                )
            )
            rationale = (
                f"Counts distinct {id_col.name} entities rather than rows because "
                f"{id_col.name} repeats across observations."
            )
        else:
            rationale = f"Counts distinct {id_col.name} entities across observed rows."
        if in_grain:
            evidence.append(
                Evidence(
                    "entity_at_observation_grain",
                    f"{id_col.name} contributes to the estimated observation grain",
                    {"grain_confidence": grain_confidence},
                )
            )
        priority_tier = (
            1 if (in_grain or in_hierarchy or repeats or _identifier_has_name_hint(id_col)) else 2
        )
        results.append(
            (
                (priority_tier, -confidence, name),
                MetricDefinition(
                    name=name,
                    label=label,
                    kind="entity_count",
                    sql=f"COUNT(DISTINCT {_sql_col(id_col.name)})",
                    columns=(id_col.name,),
                    confidence=confidence,
                    rationale=rationale,
                    evidence=tuple(evidence),
                    caveats=_column_caveats(
                        (id_col,),
                        includes_sum=False,
                        has_duplicate_rows=has_duplicate_rows,
                    ),
                ),
            )
        )
    return results


def _infer_measure_totals(
    additive_measures: Sequence[ColumnContext],
    col_by_name: dict[str, ColumnContext],
    *,
    grain_cols: set[str],
    constant_measures: frozenset[tuple[str, str]],
    row_count: int,
    has_duplicate_rows: bool,
) -> list[_Candidate]:
    results: list[_Candidate] = []
    for m_col in additive_measures:
        observed = _observed_count(m_col, row_count)
        slug = _measure_slug(m_col.name)
        m_words = column_name_words(m_col.name)
        if m_words and m_words[0] == "total":
            name = slug
            label = slug.replace("_", " ").capitalize()
        else:
            name = f"total_{slug}"
            label = f"Total {slug.replace('_', ' ')}"
        named_measure = _measure_has_name_hint(m_col)
        confidence = round(
            min(
                0.98,
                0.45 * m_col.semantic.confidence
                + 0.25 * _coverage(m_col, row_count)
                + 0.20 * _sample_support(observed)
                + 0.10 * (1.0 if named_measure else 0.75),
            ),
            2,
        )
        evidence_list = [
            Evidence(
                "additive_measure_role",
                f"{m_col.name} is an additive measure with recommended SUM aggregation",
                {
                    "observed_count": observed,
                    "missing_count": m_col.statistics.missing_count,
                    "semantic_confidence": m_col.semantic.confidence,
                },
            )
        ]
        det_col = _pick_measure_determinant(
            m_col, constant_measures, col_by_name, grain_cols=grain_cols
        )
        base_caveats = _column_caveats(
            (m_col,),
            includes_sum=True,
            has_duplicate_rows=has_duplicate_rows,
        )
        if det_col is not None:
            alias = _cte_alias("per", det_col.name)
            det_sql = _sql_col(det_col.name)
            m_sql = _sql_col(m_col.name)
            sql = (
                f"WITH {alias} AS (SELECT {det_sql}, MAX({m_sql}) AS measure_value "
                f"FROM dataset WHERE {det_sql} IS NOT NULL GROUP BY {det_sql}) "
                f"SELECT SUM(measure_value) FROM {alias}"
            )
            rationale = (
                f"Sums deduplicated {m_col.name} at {det_col.name} grain to avoid "
                "double-counting values repeated across rows."
            )
            evidence_list.append(_dedup_evidence(m_col, det_col, row_count))
            caveats = _merge_caveats(base_caveats, _dedup_caveat(m_col, det_col))
        else:
            sql = f"SUM({_sql_col(m_col.name)})"
            rationale = f"Sums additive measure {m_col.name} across observed rows."
            caveats = base_caveats
        priority_tier = 1 if named_measure else 2
        results.append(
            (
                (priority_tier, -confidence, name),
                MetricDefinition(
                    name=name,
                    label=label,
                    kind="measure_total",
                    sql=sql,
                    columns=(m_col.name,),
                    confidence=confidence,
                    rationale=rationale,
                    evidence=tuple(evidence_list),
                    caveats=caveats,
                ),
            )
        )
    return results


def _infer_measure_per_entity(
    identifiers: Sequence[ColumnContext],
    additive_measures: Sequence[ColumnContext],
    col_by_name: dict[str, ColumnContext],
    *,
    grain_cols: set[str],
    grain_confidence: float,
    one_to_many_cols: set[str],
    constant_measures: frozenset[tuple[str, str]],
    entity_determinants: frozenset[tuple[str, str]],
    row_count: int,
    has_duplicate_rows: bool,
) -> list[_Candidate]:
    order_value_candidates = [
        (rank, m_col.name)
        for m_col in additive_measures
        if (rank := _value_measure_rank(m_col)) is not None
    ]
    primary_order_value_col = (
        sorted(order_value_candidates)[0][1] if order_value_candidates else None
    )

    results: list[_Candidate] = []
    for id_col in identifiers:
        stem, _ = _entity_stem(id_col.name)
        e_observed = _observed_count(id_col, row_count)
        repeats = id_col.statistics.distinct_count < e_observed
        in_grain = id_col.name in grain_cols
        in_hierarchy = id_col.name in one_to_many_cols
        has_name_hint = _identifier_has_name_hint(id_col)

        if stem in _ROW_KEY_STEMS and not repeats:
            continue
        if not (has_name_hint or repeats or in_hierarchy):
            continue

        for m_col in additive_measures:
            m_observed = _observed_count(m_col, row_count)
            m_slug = _measure_slug(m_col.name)
            is_aov = stem == "order" and m_col.name == primary_order_value_col
            if is_aov:
                name = "average_order_value"
                label = "Average order value"
            else:
                name = f"{m_slug}_per_{stem}"
                label = f"{m_slug.replace('_', ' ').capitalize()} per {stem.replace('_', ' ')}"

            avg_sem = (m_col.semantic.confidence + id_col.semantic.confidence) / 2.0
            min_cov = min(_coverage(m_col, row_count), _coverage(id_col, row_count))
            gen = _sample_support(min(m_observed, e_observed))
            structural = 1.0 if (repeats or in_grain or in_hierarchy) else 0.75
            confidence = round(
                min(0.98, 0.40 * avg_sem + 0.20 * min_cov + 0.20 * gen + 0.20 * structural),
                2,
            )
            evidence = [
                Evidence(
                    "additive_measure_and_entity",
                    f"Combines additive measure {m_col.name} with identifier {id_col.name} ({id_col.statistics.distinct_count:,} distinct entities)",
                    {
                        "entity_distinct_count": id_col.statistics.distinct_count,
                        "measure_observed_count": m_observed,
                        "entity_observed_count": e_observed,
                    },
                )
            ]
            if repeats:
                evidence.append(
                    Evidence(
                        "repeated_entity_observations",
                        f"{id_col.name} repeats across rows, so aggregating {m_col.name} per distinct {id_col.name} avoids row-grain distortion",
                        {
                            "distinct_count": id_col.statistics.distinct_count,
                            "observed_count": e_observed,
                        },
                    )
                )
            if in_grain:
                evidence.append(
                    Evidence(
                        "entity_at_observation_grain",
                        f"{id_col.name} participates in the inferred observation grain",
                        {"grain_confidence": grain_confidence},
                    )
                )
            if in_hierarchy:
                evidence.append(
                    Evidence(
                        "entity_hierarchy_support",
                        f"{id_col.name} participates in an observed one-to-many entity hierarchy",
                        {"distinct_count": id_col.statistics.distinct_count},
                    )
                )
            if is_aov or (m_col.name == primary_order_value_col and repeats):
                priority_tier = 0
            elif _measure_has_name_hint(m_col) and (repeats or in_hierarchy):
                priority_tier = 3
            else:
                priority_tier = 4

            det_col = _pick_measure_determinant(
                m_col,
                constant_measures,
                col_by_name,
                preferred_col=id_col.name,
                grain_cols=grain_cols,
            )
            base_caveats = _column_caveats(
                (m_col, id_col),
                includes_sum=True,
                has_duplicate_rows=has_duplicate_rows,
            )
            if det_col is not None:
                alias = _cte_alias("per", det_col.name)
                det_sql = _sql_col(det_col.name)
                m_sql = _sql_col(m_col.name)
                id_sql = _sql_col(id_col.name)
                sum_expr = (
                    "SUM(measure_value)"
                    if _is_float_column(m_col)
                    else "1.0 * SUM(measure_value)"
                )
                if det_col.name == id_col.name:
                    sql = (
                        f"WITH {alias} AS (SELECT {det_sql}, MAX({m_sql}) AS measure_value "
                        f"FROM dataset WHERE {det_sql} IS NOT NULL GROUP BY {det_sql}) "
                        f"SELECT {sum_expr} / NULLIF(COUNT({det_sql}), 0) FROM {alias}"
                    )
                    rationale = (
                        f"Deduplicates {m_col.name} to {det_col.name} grain before computing "
                        f"per-{stem} average so repeated rows do not double-count {m_col.name}."
                    )
                    caveats = _merge_caveats(base_caveats, _dedup_caveat(m_col, det_col))
                elif (
                    (det_col.name, id_col.name) in entity_determinants
                    and det_col.statistics.missing_count == 0
                ):
                    sql = (
                        f"WITH {alias} AS (SELECT {det_sql}, MAX({id_sql}) AS entity_value, "
                        f"MAX({m_sql}) AS measure_value FROM dataset "
                        f"WHERE {det_sql} IS NOT NULL GROUP BY {det_sql}) "
                        f"SELECT {sum_expr} / NULLIF(COUNT(DISTINCT entity_value), 0) FROM {alias}"
                    )
                    rationale = (
                        f"Deduplicates {m_col.name} to {det_col.name} grain before dividing "
                        f"by distinct {id_col.name} entities."
                    )
                    caveats = _merge_caveats(base_caveats, _dedup_caveat(m_col, det_col))
                else:
                    sql = (
                        f"WITH {alias} AS (SELECT {det_sql}, MAX({m_sql}) AS measure_value "
                        f"FROM dataset WHERE {det_sql} IS NOT NULL GROUP BY {det_sql}) "
                        f"SELECT (SELECT {sum_expr} FROM {alias}) / "
                        f"NULLIF((SELECT COUNT(DISTINCT {id_sql}) FROM dataset), 0)"
                    )
                    rationale = (
                        f"Deduplicates {m_col.name} at {det_col.name} grain before dividing "
                        f"by distinct {id_col.name} entities."
                    )
                    caveats = _merge_caveats(
                        base_caveats,
                        _dedup_caveat(m_col, det_col, grain_mismatch_with=id_col.name),
                    )
                evidence.append(_dedup_evidence(m_col, det_col, row_count))
            else:
                sql = f"{_sql_sum_numerator(m_col)} / NULLIF(COUNT(DISTINCT {_sql_col(id_col.name)}), 0)"
                rationale = (
                    f"Computes total {m_col.name} per distinct {id_col.name} "
                    "using NULLIF to avoid division by zero."
                )
                caveats = base_caveats

            results.append(
                (
                    (priority_tier, -confidence, name),
                    MetricDefinition(
                        name=name,
                        label=label,
                        kind="measure_per_entity",
                        sql=sql,
                        columns=(m_col.name, id_col.name),
                        confidence=confidence,
                        rationale=rationale,
                        evidence=tuple(evidence),
                        caveats=caveats,
                    ),
                )
            )
    return results


def _infer_entity_ratios(
    relationships: Sequence[RelationshipContext],
    col_by_name: dict[str, ColumnContext],
    id_names: set[str],
    *,
    multi_parent_pairs: frozenset[tuple[str, str]],
    row_count: int,
    has_duplicate_rows: bool,
) -> list[_Candidate]:
    rel_priority = {"one_to_many": 0, "functional_dependency": 1, "approximate_mapping": 2}
    selected_rels: dict[tuple[str, str], RelationshipContext] = {}
    for rel in relationships:
        if rel.kind == "one_to_many":
            pair = (rel.source_column, rel.target_column)
        elif rel.kind == "functional_dependency" and rel.mapping_cardinality == "many_to_one":
            pair = (rel.target_column, rel.source_column)
        elif rel.kind == "approximate_mapping":
            pair = (rel.target_column, rel.source_column)
        else:
            continue
        parent_name, child_name = pair
        if parent_name not in id_names or child_name not in id_names:
            continue
        parent_col = col_by_name[parent_name]
        child_col = col_by_name[child_name]
        if rel.kind != "one_to_many" and parent_col.statistics.distinct_count >= child_col.statistics.distinct_count:
            continue
        existing = selected_rels.get(pair)
        if existing is None or rel_priority[rel.kind] < rel_priority[existing.kind]:
            selected_rels[pair] = rel

    results: list[_Candidate] = []
    for (parent_name, child_name), rel in sorted(selected_rels.items()):
        group_col = col_by_name[parent_name]
        entity_col = col_by_name[child_name]
        group_stem, _ = _entity_stem(group_col.name)
        entity_stem, _ = _entity_stem(entity_col.name)
        entity_plural = _pluralize(entity_stem)
        name = f"{entity_plural}_per_{group_stem}"
        label = f"{entity_plural.replace('_', ' ').capitalize()} per {group_stem.replace('_', ' ')}"
        avg_sem = (group_col.semantic.confidence + entity_col.semantic.confidence) / 2.0
        min_cov = min(_coverage(group_col, row_count), _coverage(entity_col, row_count))
        is_multi_parent = (
            (parent_name, child_name) in multi_parent_pairs
            or rel.violation_count > 0
            or rel.affected_groups > 0
            or (rel.kind == "one_to_many" and rel.coverage < 1.0 and rel.missing_pair_count == 0)
        )
        has_missing_parent = rel.missing_pair_count > 0
        raw_confidence = 0.50 * rel.confidence + 0.30 * avg_sem + 0.20 * min_cov
        if is_multi_parent:
            raw_confidence *= rel.coverage
        confidence = round(min(0.98, raw_confidence), 2)

        evidence_list = [
            Evidence(
                "one_to_many_entity_hierarchy" if rel.kind == "one_to_many" else "entity_hierarchy_mapping",
                f"Observed mapping from {group_col.name} to {entity_col.name}",
                {
                    "relationship_kind": rel.kind,
                    "coverage": rel.coverage,
                    "relationship_confidence": rel.confidence,
                    "group_distinct_count": group_col.statistics.distinct_count,
                    "entity_distinct_count": entity_col.statistics.distinct_count,
                },
            )
        ]
        base_caveats = _column_caveats(
            (entity_col, group_col),
            includes_sum=False,
            has_duplicate_rows=has_duplicate_rows,
        )
        group_sql = _sql_col(group_col.name)
        entity_sql = _sql_col(entity_col.name)
        if is_multi_parent:
            alias = _cte_alias("per", group_col.name)
            sql = (
                f"WITH {alias} AS (SELECT {group_sql}, COUNT(DISTINCT {entity_sql}) AS entity_count "
                f"FROM dataset WHERE {group_sql} IS NOT NULL GROUP BY {group_sql}) "
                f"SELECT 1.0 * SUM(entity_count) / NULLIF(COUNT({group_sql}), 0) FROM {alias}"
            )
            rationale = (
                f"Counts distinct {entity_col.name} entities within each {group_col.name} before "
                f"averaging because some {entity_col.name} values belong to multiple {group_col.name} parents."
            )
            evidence_list.append(
                Evidence(
                    "multi_parent_entity_relationship",
                    f"Some {entity_col.name} values map to multiple {group_col.name} parents",
                    {
                        "coverage": rel.coverage,
                        "violation_count": rel.violation_count,
                        "affected_groups": rel.affected_groups,
                    },
                )
            )
            caveats = _merge_caveats(
                base_caveats,
                (
                    f"Some {entity_col.name} values belong to multiple {group_col.name} parents; "
                    f"SQL counts distinct {entity_col.name} per {group_col.name} before averaging "
                    "so shared entities are counted for each parent."
                ),
            )
        elif has_missing_parent:
            sql = (
                f"1.0 * COUNT(DISTINCT CASE WHEN {group_sql} IS NOT NULL THEN {entity_sql} END) / "
                f"NULLIF(COUNT(DISTINCT {group_sql}), 0)"
            )
            rationale = (
                f"Measures average distinct {entity_col.name} entities per distinct {group_col.name}, "
                f"excluding rows where {group_col.name} is NULL."
            )
            caveats = _merge_caveats(
                base_caveats,
                (
                    f"Rows missing {group_col.name} are excluded from the {entity_col.name} "
                    f"numerator to avoid inflating the per-{group_stem} ratio."
                ),
            )
        else:
            sql = (
                f"1.0 * COUNT(DISTINCT {entity_sql}) / "
                f"NULLIF(COUNT(DISTINCT {group_sql}), 0)"
            )
            rationale = (
                f"Measures average distinct {entity_col.name} entities per "
                f"distinct {group_col.name} based on the observed one-to-many relationship."
            )
            caveats = base_caveats

        results.append(
            (
                (2, -confidence, name),
                MetricDefinition(
                    name=name,
                    label=label,
                    kind="entity_ratio",
                    sql=sql,
                    columns=(entity_col.name, group_col.name),
                    confidence=confidence,
                    rationale=rationale,
                    evidence=tuple(evidence_list),
                    caveats=caveats,
                ),
            )
        )
    return results


def _build_measure_ratio(
    name: str,
    label: str,
    numerator: ColumnContext,
    denominator: ColumnContext,
    rationale: str,
    col_by_name: dict[str, ColumnContext],
    *,
    grain_cols: set[str],
    constant_measures: frozenset[tuple[str, str]],
    row_count: int,
    has_duplicate_rows: bool,
) -> _Candidate:
    confidence = _pair_measure_confidence(numerator, denominator, row_count)
    evidence_list = [
        Evidence(
            "additive_measure_ratio",
            f"Both {numerator.name} and {denominator.name} are additive measures",
            {
                "numerator_observed_count": _observed_count(numerator, row_count),
                "denominator_observed_count": _observed_count(denominator, row_count),
            },
        )
    ]
    base_caveats = _column_caveats(
        (numerator, denominator),
        includes_sum=True,
        has_duplicate_rows=has_duplicate_rows,
    )
    num_det = _pick_measure_determinant(
        numerator, constant_measures, col_by_name, grain_cols=grain_cols
    )
    den_det = _pick_measure_determinant(
        denominator, constant_measures, col_by_name, grain_cols=grain_cols
    )
    num_sql = _sql_col(numerator.name)
    den_sql = _sql_col(denominator.name)
    if num_det is None and den_det is None:
        sql = f"{_sql_sum_numerator(numerator)} / NULLIF(SUM({den_sql}), 0)"
        caveats = base_caveats
    elif num_det is not None and den_det is not None and num_det.name == den_det.name:
        alias = _cte_alias("per", num_det.name)
        det_sql = _sql_col(num_det.name)
        num_sum = (
            "SUM(numerator_value)"
            if _is_float_column(numerator)
            else "1.0 * SUM(numerator_value)"
        )
        sql = (
            f"WITH {alias} AS (SELECT {det_sql}, MAX({num_sql}) AS numerator_value, "
            f"MAX({den_sql}) AS denominator_value FROM dataset "
            f"WHERE {det_sql} IS NOT NULL GROUP BY {det_sql}) "
            f"SELECT {num_sum} / NULLIF(SUM(denominator_value), 0) FROM {alias}"
        )
        evidence_list.append(_dedup_evidence(numerator, num_det, row_count))
        evidence_list.append(_dedup_evidence(denominator, den_det, row_count))
        caveats = _merge_caveats(
            base_caveats,
            _dedup_caveat(numerator, num_det),
            _dedup_caveat(denominator, den_det),
        )
    else:
        ctes: list[str] = []
        extra_caveats: list[str] = []
        if num_det is not None:
            num_alias = _cte_alias("num_per", num_det.name)
            num_det_sql = _sql_col(num_det.name)
            ctes.append(
                f"{num_alias} AS (SELECT {num_det_sql}, MAX({num_sql}) AS measure_value "
                f"FROM dataset WHERE {num_det_sql} IS NOT NULL GROUP BY {num_det_sql})"
            )
            num_select = f"(SELECT SUM(measure_value) FROM {num_alias})"
            evidence_list.append(_dedup_evidence(numerator, num_det, row_count))
            extra_caveats.append(_dedup_caveat(numerator, num_det))
        else:
            num_select = f"(SELECT SUM({num_sql}) FROM dataset)"
        if den_det is not None:
            den_alias = _cte_alias("den_per", den_det.name)
            den_det_sql = _sql_col(den_det.name)
            ctes.append(
                f"{den_alias} AS (SELECT {den_det_sql}, MAX({den_sql}) AS measure_value "
                f"FROM dataset WHERE {den_det_sql} IS NOT NULL GROUP BY {den_det_sql})"
            )
            den_select = f"(SELECT SUM(measure_value) FROM {den_alias})"
            evidence_list.append(_dedup_evidence(denominator, den_det, row_count))
            extra_caveats.append(_dedup_caveat(denominator, den_det))
        else:
            den_select = f"(SELECT SUM({den_sql}) FROM dataset)"
        float_prefix = "" if _is_float_column(numerator) else "1.0 * "
        sql = f"WITH {', '.join(ctes)} SELECT {float_prefix}{num_select} / NULLIF({den_select}, 0)"
        caveats = _merge_caveats(base_caveats, *extra_caveats)

    return (
        (0, -confidence, name),
        MetricDefinition(
            name=name,
            label=label,
            kind="measure_ratio",
            sql=sql,
            columns=(numerator.name, denominator.name),
            confidence=confidence,
            rationale=rationale,
            evidence=tuple(evidence_list),
            caveats=caveats,
        ),
    )


def _infer_measure_ratios(
    additive_measures: Sequence[ColumnContext],
    col_by_name: dict[str, ColumnContext],
    *,
    grain_cols: set[str],
    constant_measures: frozenset[tuple[str, str]],
    row_count: int,
    has_duplicate_rows: bool,
) -> list[_Candidate]:
    revenue_cols = sorted(
        (
            col
            for col in additive_measures
            if any(token in column_name_words(col.name) for token in _REVENUE_TOKENS)
        ),
        key=lambda col: (
            0 if "revenue" in column_name_words(col.name) else 1,
            -col.semantic.confidence,
            col.name,
        ),
    )
    if not revenue_cols:
        return []

    rev_col = revenue_cols[0]
    rev_token = "revenue" if "revenue" in column_name_words(rev_col.name) else "sales"
    results: list[_Candidate] = []

    profit_cols = sorted(
        (col for col in additive_measures if "profit" in column_name_words(col.name)),
        key=lambda col: (-col.semantic.confidence, col.name),
    )
    if profit_cols and profit_cols[0].name != rev_col.name:
        prof_col = profit_cols[0]
        results.append(
            _build_measure_ratio(
                "profit_margin",
                "Profit margin",
                prof_col,
                rev_col,
                (
                    f"Computes aggregate {prof_col.name} over aggregate {rev_col.name} "
                    "rather than averaging row-level margins."
                ),
                col_by_name,
                grain_cols=grain_cols,
                constant_measures=constant_measures,
                row_count=row_count,
                has_duplicate_rows=has_duplicate_rows,
            )
        )

    quantity_cols = sorted(
        (
            col
            for col in additive_measures
            if not _QUANTITY_TOKENS.isdisjoint(column_name_words(col.name))
        ),
        key=lambda col: (-col.semantic.confidence, col.name),
    )
    if quantity_cols and quantity_cols[0].name != rev_col.name:
        qty_col = quantity_cols[0]
        results.append(
            _build_measure_ratio(
                f"{rev_token}_per_unit",
                f"{rev_token.capitalize()} per unit",
                rev_col,
                qty_col,
                (
                    f"Computes aggregate {rev_col.name} per summed unit of {qty_col.name} "
                    "using NULLIF to avoid division by zero."
                ),
                col_by_name,
                grain_cols=grain_cols,
                constant_measures=constant_measures,
                row_count=row_count,
                has_duplicate_rows=has_duplicate_rows,
            )
        )

    return results


def infer_metrics(
    columns: Sequence[ColumnContext],
    *,
    grain: GrainContext | None = None,
    relationships: tuple[RelationshipContext, ...] = (),
    row_count: int = 0,
    has_duplicate_rows: bool = False,
    entity_constant_measures: frozenset[tuple[str, str]] = frozenset(),
    entity_determinants: frozenset[tuple[str, str]] = frozenset(),
    multi_parent_pairs: frozenset[tuple[str, str]] = frozenset(),
    partially_covered_measures: frozenset[str] = frozenset(),
) -> tuple[MetricDefinition, ...]:
    """Infer conservative, evidence-backed analytical metric definitions.

    Metrics are derived strictly from already-computed column roles,
    recommendations, cardinality, grain, and within-frame relationships.
    """
    if row_count <= 0 or not columns:
        return ()

    col_by_name = {col.name: col for col in columns}
    grain_cols = set(grain.columns) if grain and grain.status == "estimated" else set()
    grain_confidence = grain.confidence if grain else 0.0
    one_to_many_rels = [
        rel
        for rel in relationships
        if rel.kind == "one_to_many"
        and rel.source_column in col_by_name
        and rel.target_column in col_by_name
    ]
    one_to_many_cols = {
        name for rel in one_to_many_rels for name in (rel.source_column, rel.target_column)
    }

    identifiers = [
        col
        for col in columns
        if col.semantic.role == "identifier"
        and col.statistics.distinct_count >= 2
        and _observed_count(col, row_count) > 0
    ]
    additive_measures = [
        col for col in columns if _is_additive_measure(col) and _observed_count(col, row_count) > 0
    ]
    if not identifiers and not additive_measures:
        return ()

    id_names = {col.name for col in identifiers}
    repeating_id_names = {
        col.name
        for col in identifiers
        if col.statistics.distinct_count < _observed_count(col, row_count)
    }
    measure_names = {col.name for col in additive_measures}
    effective_constant_measures = set(entity_constant_measures)
    effective_determinants = set(entity_determinants)
    for rel in relationships:
        if (
            rel.kind == "functional_dependency"
            and rel.coverage == 1.0
            and rel.violation_count == 0
            and rel.missing_pair_count == 0
        ):
            if rel.source_column in repeating_id_names and rel.target_column in measure_names:
                effective_constant_measures.add((rel.source_column, rel.target_column))
            if rel.source_column in id_names and rel.target_column in id_names:
                effective_determinants.add((rel.source_column, rel.target_column))
        elif (
            rel.kind == "one_to_many"
            and rel.coverage == 1.0
            and rel.missing_pair_count == 0
            and rel.source_column in id_names
            and rel.target_column in id_names
        ):
            effective_determinants.add((rel.target_column, rel.source_column))

    frozen_constant_measures = frozenset(effective_constant_measures)
    frozen_determinants = frozenset(effective_determinants)
    fully_covered_measures = {measure for _, measure in frozen_constant_measures}
    ambiguous_measures = partially_covered_measures - fully_covered_measures
    additive_measures = [col for col in additive_measures if col.name not in ambiguous_measures]

    candidates: list[_Candidate] = [
        *_infer_entity_counts(
            identifiers,
            grain_cols=grain_cols,
            grain_confidence=grain_confidence,
            one_to_many_cols=one_to_many_cols,
            row_count=row_count,
            has_duplicate_rows=has_duplicate_rows,
        ),
        *_infer_measure_totals(
            additive_measures,
            col_by_name,
            grain_cols=grain_cols,
            constant_measures=frozen_constant_measures,
            row_count=row_count,
            has_duplicate_rows=has_duplicate_rows,
        ),
        *_infer_measure_per_entity(
            identifiers,
            additive_measures,
            col_by_name,
            grain_cols=grain_cols,
            grain_confidence=grain_confidence,
            one_to_many_cols=one_to_many_cols,
            constant_measures=frozen_constant_measures,
            entity_determinants=frozen_determinants,
            row_count=row_count,
            has_duplicate_rows=has_duplicate_rows,
        ),
        *_infer_entity_ratios(
            relationships,
            col_by_name,
            id_names,
            multi_parent_pairs=multi_parent_pairs,
            row_count=row_count,
            has_duplicate_rows=has_duplicate_rows,
        ),
        *_infer_measure_ratios(
            additive_measures,
            col_by_name,
            grain_cols=grain_cols,
            constant_measures=frozen_constant_measures,
            row_count=row_count,
            has_duplicate_rows=has_duplicate_rows,
        ),
    ]
    if not candidates:
        return ()

    candidates.sort(key=lambda item: (item[0][0], item[0][1], item[0][2], item[1].columns))
    selected = [metric for _, metric in candidates[:_MAX_METRICS]]
    selected.sort(key=lambda m: (_KIND_ORDER.get(m.kind, 99), -m.confidence, m.name, m.columns))

    seen_names: set[str] = set()
    finalized: list[MetricDefinition] = []
    for metric in selected:
        unique = _unique_name(metric.name, seen_names)
        if unique != metric.name:
            metric = MetricDefinition(
                name=unique,
                label=metric.label,
                kind=metric.kind,
                sql=metric.sql,
                columns=metric.columns,
                confidence=metric.confidence,
                rationale=metric.rationale,
                evidence=metric.evidence,
                caveats=metric.caveats,
            )
        finalized.append(metric)
    return tuple(finalized)
