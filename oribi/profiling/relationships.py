"""Conservative column dependency detection, independent of DataFrame index names."""

import math
from collections.abc import Sequence
from functools import cache
from typing import NamedTuple

import pandas as pd

from ..models.context import ColumnContext
from ..models.evidence import (
    CardinalityContext,
    ColumnRelationshipContext,
    Evidence,
    GrainContext,
    KeyCandidate,
    RelationshipContext,
    SemanticContext,
)


def _nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return int(series.dropna().astype(str).nunique())


def _generalization_support(observed_count: int) -> float:
    """A modest sample-size factor; direct observed evidence remains intact."""
    if observed_count <= 0:
        return 0.0
    return 0.7 + 0.3 * observed_count / (observed_count + 12)


def _score(components: dict[str, float], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Return a transparent weighted score and its inputs/contributions."""
    breakdown = dict(components)
    for name, weight in weights.items():
        breakdown[f"{name}_weight"] = weight
        breakdown[f"{name}_contribution"] = components[name] * weight
    return round(min(1.0, sum(components[name] * weight for name, weight in weights.items())), 2), breakdown


def infer_column_relationships(
    cardinalities: dict[str, CardinalityContext],
    semantics: dict[str, SemanticContext],
    logical_types: dict[str, str],
    key_candidates: tuple[KeyCandidate, ...],
    grain: GrainContext,
    relationships: tuple[RelationshipContext, ...],
) -> dict[str, tuple[ColumnRelationshipContext, ...]]:
    """Relate columns to observed rows, grain, and repeated groups."""
    inferred: dict[str, list[ColumnRelationshipContext]] = {name: [] for name in cardinalities}

    def add(name: str, relation: ColumnRelationshipContext) -> None:
        if not any(
            existing.kind == relation.kind and existing.related_columns == relation.related_columns
            for existing in inferred[name]
        ):
            inferred[name].append(relation)

    candidates_by_columns = {candidate.columns: candidate for candidate in key_candidates}

    for name, card in cardinalities.items():
        # Relationship candidates can be useful below the stricter grain
        # threshold, but temporal and measure semantics take precedence over
        # coincidental uniqueness.
        candidate = next((
            item for item in key_candidates
            if item.columns == (name,)
        ), None)
        uniqueness_ratio = card.uniqueness_ratio or 0.0
        if candidate is None and (
            uniqueness_ratio < 0.80
            or semantics[name].role in {"measure", "temporal", "boolean"}
        ):
            continue
        if candidate is not None or uniqueness_ratio >= 0.80:
            card = cardinalities[name]
            uniqueness = card.uniqueness_ratio or 0.0
            coverage = card.observed_count / card.row_count if card.row_count else 0.0
            generalization = _generalization_support(card.observed_count)
            semantic = semantics[name].confidence
            confidence, components = _score(
                {"uniqueness": uniqueness, "coverage": coverage,
                 "generalization": generalization, "semantic_support": semantic},
                {"uniqueness": 0.45, "coverage": 0.20,
                 "generalization": 0.20, "semantic_support": 0.15},
            )
            evidence = candidate.evidence if candidate else (Evidence(
                "mostly_unique_observation_candidate",
                f"{card.unique_count} distinct values across {card.observed_count} non-missing rows",
                {"distinct_count": card.unique_count, "observed_count": card.observed_count,
                 "row_count": card.row_count, "uniqueness_ratio": uniqueness,
                 "coverage": coverage},
            ),)
            add(name, ColumnRelationshipContext(
                "identifies_observations",
                confidence,
                evidence,
                evidence_strength=uniqueness,
                confidence_components=components,
            ))
    if grain.status == "estimated" and len(grain.columns) > 1:
        selected = candidates_by_columns.get(grain.columns)
        evidence = selected.evidence if selected else grain.evidence
        row_counts = [cardinalities[name].observed_count for name in grain.columns]
        row_count = min(row_counts, default=0)
        coverage = min((
            cardinalities[name].observed_count / cardinalities[name].row_count
            if cardinalities[name].row_count else 0.0
            for name in grain.columns
        ), default=0.0)
        uniqueness = selected.uniqueness_rate if selected else 0.0
        confidence, components = _score(
            {"uniqueness": uniqueness, "coverage": coverage,
             "generalization": _generalization_support(row_count),
             "semantic_support": sum(semantics[name].confidence for name in grain.columns) / len(grain.columns)},
            {"uniqueness": 0.45, "coverage": 0.20,
             "generalization": 0.20, "semantic_support": 0.15},
        )
        for name in grain.columns:
            add(name, ColumnRelationshipContext(
                "jointly_identifies_observations",
                confidence,
                evidence,
                tuple(other for other in grain.columns if other != name),
                evidence_strength=uniqueness,
                confidence_components=components,
            ))
    elif (
        grain.status == "estimated"
        and grain.columns
        and not any(rel.kind == "identifies_observations" for rel in inferred[grain.columns[0]])
    ):
        name = grain.columns[0]
        card = cardinalities[name]
        uniqueness = card.uniqueness_ratio or 0.0
        coverage = card.observed_count / card.row_count if card.row_count else 0.0
        confidence, components = _score(
            {"uniqueness": uniqueness, "coverage": coverage,
             "generalization": _generalization_support(card.observed_count),
             "semantic_support": semantics[name].confidence},
            {"uniqueness": 0.45, "coverage": 0.20,
             "generalization": 0.20, "semantic_support": 0.15},
        )
        add(name, ColumnRelationshipContext(
            "identifies_observations", confidence, grain.evidence,
            evidence_strength=uniqueness, confidence_components=components,
        ))

    for relation in relationships:
        if relation.kind == "one_to_many":
            add(relation.source_column, ColumnRelationshipContext(
                "groups_observations",
                relation.confidence,
                relation.evidence,
                (relation.target_column,),
                evidence_strength=relation.coverage,
            ))

    for name, cardinality in cardinalities.items():
        col_semantic = semantics[name]
        coverage = cardinality.observed_count / cardinality.row_count if cardinality.row_count else 0.0
        generalization = _generalization_support(cardinality.observed_count)
        has_group_relation = any(rel.kind == "groups_observations" for rel in inferred[name])
        if col_semantic.role == "measure":
            # A measure can only be attached to a named grain when that grain
            # has meaningful sample support. Otherwise its observed level is
            # simply the row, and attaching it to a coincidentally unique
            # column would overstate what the data demonstrates.
            supported_grain = (
                grain.columns
                if grain.status == "estimated"
                and grain.confidence >= 0.75
                and cardinality.row_count >= 20
                else ()
            )
            col_evidence = Evidence(
                "numeric_measure_at_observation_grain",
                f"numeric column observed across {cardinality.observed_count} non-missing rows",
                {"observed_count": cardinality.observed_count, "grain": supported_grain},
            )
            confidence, components = _score(
                {"numeric_type": 1.0, "semantic_support": col_semantic.confidence,
                 "coverage": coverage, "generalization": generalization},
                {"numeric_type": 0.45, "semantic_support": 0.20,
                 "coverage": 0.20, "generalization": 0.15},
            )
            add(name, ColumnRelationshipContext(
                "measure_at_grain",
                confidence,
                (col_evidence,),
                supported_grain,
                coverage,
                confidence_components=components,
            ))
        elif logical_types[name] == "temporal":
            col_evidence = Evidence(
                "temporal_observation_attribute",
                f"datetime-typed column with {cardinality.observed_count} non-missing values",
                {"observed_count": cardinality.observed_count, "missing_count": cardinality.row_count - cardinality.observed_count},
            )
            confidence, components = _score(
                {"datetime_type": 1.0, "semantic_support": col_semantic.confidence,
                 "coverage": coverage, "generalization": generalization},
                {"datetime_type": 0.55, "semantic_support": 0.10,
                 "coverage": 0.20, "generalization": 0.15},
            )
            add(name, ColumnRelationshipContext(
                "temporal_attribute",
                confidence,
                (col_evidence,),
                grain.columns,
                coverage,
                confidence_components=components,
            ))
        elif col_semantic.role in {"dimension", "categorical", "boolean"}:
            ratio = cardinality.uniqueness_ratio or 0.0
            if cardinality.unique_count > 0 and ratio < 1 and not has_group_relation:
                col_evidence = Evidence(
                    "reusable_values_across_observations",
                    f"{cardinality.unique_count} observed values reused across {cardinality.observed_count} rows",
                    {
                        "unique_count": cardinality.unique_count,
                        "observed_count": cardinality.observed_count,
                        "uniqueness_ratio": ratio,
                    },
                )
                reuse = 1 - ratio
                group_shape = min(1.0, (cardinality.average_records_per_value or 1.0) / 2.0)
                confidence, components = _score(
                    {"reuse": reuse, "coverage": coverage,
                     "group_shape": group_shape, "generalization": generalization},
                    {"reuse": 0.50, "coverage": 0.20,
                     "group_shape": 0.15, "generalization": 0.15},
                )
                add(name, ColumnRelationshipContext(
                    "groups_observations",
                    confidence,
                    (col_evidence,),
                    evidence_strength=1 - ratio,
                    confidence_components=components,
                ))
        elif col_semantic.role == "identifier" and cardinality.unique_count > 1 and not has_group_relation:
            ratio = cardinality.uniqueness_ratio or 0.0
            if ratio < 1:
                col_evidence = Evidence(
                    "repeated_identifier_values",
                    f"{cardinality.unique_count} identifier-like values reused across {cardinality.observed_count} rows",
                    {"unique_count": cardinality.unique_count, "observed_count": cardinality.observed_count, "uniqueness_ratio": ratio},
                )
                confidence, components = _score(
                    {"reuse": 1 - ratio, "coverage": coverage,
                     "generalization": generalization,
                     "semantic_support": col_semantic.confidence},
                    {"reuse": 0.50, "coverage": 0.20,
                     "generalization": 0.15, "semantic_support": 0.15},
                )
                add(name, ColumnRelationshipContext(
                    "groups_observations",
                    confidence,
                    (col_evidence,),
                    evidence_strength=1 - ratio,
                    confidence_components=components,
                ))
    return {name: tuple(relations) for name, relations in inferred.items()}


def detect_relationships(
    frame: pd.DataFrame,
    roles: dict[str, str],
    max_rows: int = 100_000,
    candidate_keys: tuple[KeyCandidate, ...] = (),
    grain: GrainContext | None = None,
    max_relationships: int = 40,
    max_relationship_columns: int = 40,
) -> tuple[RelationshipContext, ...]:
    """Find observed value mappings and repeated group-to-entity links.

    Directional mapping candidates must vary before they count toward the
    column budget. Sources must also reuse values, but targets may be unique.

    One-to-many links require an observed candidate key on the entity side,
    with each entity mapping to one repeated group value. Column names do not
    establish a relationship.
    """
    if (
        len(frame) == 0
        or len(frame) > max_rows
        or max_relationships <= 0
        or max_relationship_columns <= 0
    ):
        return ()
    search_columns = min(max_relationship_columns, max(8, math.ceil(math.sqrt(max_relationships * 4))))
    max_distinct = min(500, max(20, len(frame) // 2))

    @cache
    def distinct_count(name: str) -> int:
        return _nunique(frame[name])

    @cache
    def non_null_count(name: str) -> int:
        return int(frame[name].notna().sum())

    def rank(name: str) -> tuple[int, str]:
        return distinct_count(name), name

    source_observed_counts: dict[str, int] = {}
    eligible_targets: list[str] = []
    for name, role in roles.items():
        if role == "unknown":
            continue
        column_distinct = distinct_count(name)
        if not 2 <= column_distinct <= max_distinct:
            continue
        eligible_targets.append(name)
        if role in {"identifier", "dimension", "categorical", "boolean", "text", "temporal"}:
            observed = non_null_count(name)
            # A unique determinant maps trivially to every row-level value.
            if column_distinct < observed:
                source_observed_counts[name] = observed

    mapping_sources = sorted(source_observed_counts, key=rank)[:search_columns]
    mapping_targets = sorted(eligible_targets, key=rank)[:search_columns]
    results: list[RelationshipContext] = []
    for source in mapping_sources:
        source_values = distinct_count(source)
        source_observed = source_observed_counts[source]
        for target in mapping_targets:
            if source == target:
                continue
            try:
                pairs = frame[[source, target]].dropna().drop_duplicates()
            except (TypeError, ValueError):
                continue
            mapping_counts = pairs.groupby(pairs[source], dropna=True, observed=True, sort=False)[target].nunique()
            unique_mapping = int(mapping_counts.eq(1).sum())
            coverage = unique_mapping / source_values
            affected_groups = int(mapping_counts.gt(1).sum())
            violations = int((mapping_counts - 1).clip(lower=0).sum())
            paired_rows = int(frame[[source, target]].notna().all(axis=1).sum())
            missing_pairs = len(frame) - paired_rows
            if coverage >= .98:
                target_values = distinct_count(target)
                exact_mapping = unique_mapping == source_values and not violations and not missing_pairs
                repeated_fraction = max(0.0, (source_observed - source_values) / source_observed)
                confidence, components = _score(
                    {"mapping_coverage": coverage, "repeated_determinant_support": repeated_fraction,
                     "generalization": _generalization_support(source_observed)},
                    {"mapping_coverage": 0.55, "repeated_determinant_support": 0.25,
                     "generalization": 0.20},
                )
                results.append(RelationshipContext(
                    source, target,
                    "functional_dependency" if exact_mapping else "approximate_mapping",
                    coverage,
                    confidence,
                    (Evidence("value_mapping", f"{coverage:.1%} of observed {source} values map to one {target} value", {"source_distinct": int(source_values), "target_distinct": int(target_values), "coverage": coverage, "violation_count": violations, "affected_groups": affected_groups, "paired_row_count": paired_rows, "missing_pair_count": missing_pairs}),),
                    confidence_components=components,
                    violation_count=violations,
                    affected_groups=affected_groups,
                    paired_row_count=paired_rows,
                    missing_pair_count=missing_pairs,
                    mapping_cardinality=(
                        "one_to_one" if source_values == target_values else "many_to_one"
                    ) if exact_mapping else None,
                ))
    dependency_edges = {
        (relation.source_column, relation.target_column): relation
        for relation in results if relation.kind == "functional_dependency"
    }
    known_one_to_one = {
        frozenset((relation.source_column, relation.target_column))
        for relation in results if relation.kind == "one_to_one"
    }
    for (source, target), forward in tuple(dependency_edges.items()):
        reverse = dependency_edges.get((target, source))
        pair = frozenset((source, target))
        if reverse is None or pair in known_one_to_one:
            continue
        forward_details = forward.evidence[0].details
        distinct = forward_details.get("source_distinct")
        if not isinstance(distinct, int) or distinct != forward_details.get("target_distinct"):
            continue
        coverage = min(forward.coverage, reverse.coverage)
        # Both directions have already passed the reused-value source filter.
        observed_count = min(source_observed_counts[source], source_observed_counts[target])
        repeated_fraction = max(0.0, (observed_count - distinct) / observed_count) if observed_count else 0.0
        confidence, components = _score(
            {"mapping_coverage": coverage, "repeated_value_support": repeated_fraction,
             "generalization": _generalization_support(observed_count)},
            {"mapping_coverage": 0.55, "repeated_value_support": 0.25,
             "generalization": 0.20},
        )
        results.append(RelationshipContext(
            source,
            target,
            "one_to_one",
            coverage,
            confidence,
            (Evidence(
                "bijective_value_mapping",
                f"each observed value maps one-to-one between {source} and {target}",
                {"distinct_values": distinct, "coverage": coverage},
            ),),
            confidence_components=components,
            paired_row_count=forward.paired_row_count,
            mapping_cardinality="one_to_one",
        ))
        known_one_to_one.add(pair)
    # Repeated group values mapping uniquely to observed entity keys support
    # a one-to-many link. The key must be established by value uniqueness.
    single_keys = [
        key.columns[0] for key in candidate_keys
        if key.kind == "single" and len(key.columns) == 1
        and key.verified_unique
        and (
            roles.get(key.columns[0]) == "identifier"
            or (grain is not None and key.columns[0] in grain.columns)
        )
    ]
    # A group needs multiple distinct values and at least one reused value.
    eligible_groups = sorted(
        (
            name for name, role in roles.items()
            if role in {"identifier", "dimension", "categorical", "boolean"}
            and 2 <= distinct_count(name) < non_null_count(name)
        ),
        key=lambda name: (roles[name] != "identifier", rank(name)),
    )[:search_columns]
    for entity in sorted(single_keys)[:search_columns]:
        entity_count = distinct_count(entity)
        if entity_count < 2:
            continue
        for group in sorted(eligible_groups):
            if group == entity:
                continue
            group_count = distinct_count(group)
            if group_count < 2 or group_count >= entity_count:
                continue
            try:
                pairs = frame[[group, entity]].dropna().drop_duplicates()
            except (TypeError, ValueError):
                continue
            if pairs.empty:
                continue
            entity_group_counts = pairs.groupby(pairs[entity], dropna=True, observed=True, sort=False)[group].nunique()
            coverage = float(entity_group_counts.eq(1).sum() / entity_count)
            if coverage < 1.0:
                continue
            per_group = pairs.groupby(pairs[group], dropna=True, observed=True, sort=False)[entity].nunique()
            if per_group.empty or int(per_group.max()) <= 1:
                continue
            group_reuse = max(0.0, (entity_count - group_count) / entity_count)
            paired_rows = int(frame[[group, entity]].notna().all(axis=1).sum())
            missing_pairs = len(frame) - paired_rows
            confidence, components = _score(
                {"mapping_coverage": coverage, "group_reuse": group_reuse,
                 "generalization": _generalization_support(entity_count)},
                {"mapping_coverage": 0.55, "group_reuse": 0.25,
                 "generalization": 0.20},
            )
            results.append(RelationshipContext(
                group,
                entity,
                "one_to_many",
                coverage,
                confidence,
                (Evidence(
                    "entity_group_mapping",
                    f"{coverage:.1%} of observed {entity} values map to one {group} value; "
                    f"{entity_count:,} entity values occur across {group_count:,} groups",
                    {
                        "entity_distinct": entity_count,
                        "group_distinct": group_count,
                        "average_entities_per_group": float(per_group.mean()),
                        "minimum_entities_per_group": int(per_group.min()),
                        "maximum_entities_per_group": int(per_group.max()),
                        "coverage": coverage,
                        "paired_row_count": paired_rows,
                        "missing_pair_count": missing_pairs,
                    },
                ),),
                confidence_components=components,
                paired_row_count=paired_rows,
                missing_pair_count=missing_pairs,
                mapping_cardinality="one_to_many",
            ))

    # Keep a deterministic orientation for two-way functional dependencies.
    # Retaining this existing relationship kind preserves the public snapshot;
    # renderers may prefer the more specific one-to-one summary.
    functional_dependencies: dict[frozenset[str], RelationshipContext] = {}
    retained = []
    for relation in results:
        if relation.kind != "functional_dependency":
            retained.append(relation)
            continue
        pair = frozenset((relation.source_column, relation.target_column))
        current = functional_dependencies.get(pair)
        if current is None or relation.source_column < current.source_column:
            functional_dependencies[pair] = relation
    retained.extend(functional_dependencies.values())

    # Preserve distinct relationship kinds where both are independently useful.
    deduped: dict[tuple[str, str, str], RelationshipContext] = {}
    for relation in retained:
        key = (relation.source_column, relation.target_column, relation.kind)
        current = deduped.get(key)
        if current is None or relation.coverage > current.coverage:
            deduped[key] = relation
    priority = {"one_to_many": 0, "one_to_one": 1, "functional_dependency": 2, "approximate_mapping": 3}
    ranked = sorted(
        deduped.values(),
        key=lambda r: (priority.get(r.kind, 4), -r.coverage, -r.confidence, r.source_column, r.target_column),
    )[:max_relationships]
    return tuple(sorted(ranked, key=lambda r: (r.source_column, r.target_column, r.kind)))


def _col_observed_count(col: ColumnContext, row_count: int) -> int:
    if col.cardinality is not None:
        return col.cardinality.observed_count
    return max(0, row_count - col.statistics.missing_count)


class MetricGrainFacts(NamedTuple):
    """Observed entity-grain measure facts and any scan skip or support limits."""

    entity_constant_measures: frozenset[tuple[str, str]] = frozenset()
    entity_determinants: frozenset[tuple[str, str]] = frozenset()
    multi_parent_pairs: frozenset[tuple[str, str]] = frozenset()
    partially_covered_measures: frozenset[str] = frozenset()
    skip_reason: str | None = None
    unchecked_measures: tuple[tuple[str, str], ...] = ()


def detect_metric_grain_facts(
    frame: pd.DataFrame,
    columns: Sequence[ColumnContext],
    relationships: Sequence[RelationshipContext],
    *,
    max_rows: int = 100_000,
    max_columns: int = 40,
) -> MetricGrainFacts:
    """Collect entity-grain measure constancy, determinant, and multi-parent facts.

    Also returns the names of measures that repeat under an incomplete
    determinant, so losing the determinant cannot trigger a raw sum, along
    with any scan skip reason or per-measure support limits.
    """
    row_count = len(frame)
    if row_count == 0:
        return MetricGrainFacts()
    if row_count > max_rows:
        return MetricGrainFacts(skip_reason="max_exact_rows")
    if max_columns <= 0:
        return MetricGrainFacts(skip_reason="zero_relationship_columns")

    all_identifiers = sorted(
        (
            col
            for col in columns
            if col.semantic.role == "identifier"
            and col.statistics.distinct_count >= 2
            and _col_observed_count(col, row_count) > 0
        ),
        key=lambda col: col.name,
    )
    identifiers = all_identifiers[:max_columns]
    repeating_ids = [
        col
        for col in identifiers
        if col.statistics.distinct_count < _col_observed_count(col, row_count)
    ]
    all_additive_measures = sorted(
        (
            col
            for col in columns
            if col.semantic.role == "measure"
            and _col_observed_count(col, row_count) > 0
            and any(rec.operation == "SUM" and rec.disposition == "recommended" for rec in col.recommendations)
        ),
        key=lambda col: col.name,
    )
    additive_measures = all_additive_measures[:max_columns]

    unchecked_measures: dict[str, str] = {}
    skipped_repeating_ids = any(
        col.statistics.distinct_count < _col_observed_count(col, row_count)
        for col in all_identifiers[max_columns:]
    )
    if skipped_repeating_ids:
        for col in all_additive_measures:
            unchecked_measures[col.name] = "max_relationship_columns"
    elif repeating_ids:
        for col in all_additive_measures[max_columns:]:
            unchecked_measures[col.name] = "max_relationship_columns"

    entity_constant_measures: set[tuple[str, str]] = set()
    partially_covered_measures: set[str] = set()
    for entity_col in repeating_ids:
        for measure_col in additive_measures:
            try:
                paired = frame[[entity_col.name, measure_col.name]].dropna()
                if paired.empty:
                    continue
                group_sizes = paired.groupby(
                    paired[entity_col.name], dropna=True, observed=True, sort=False
                ).size()
                if int(group_sizes.max()) < 2:
                    continue
                distinct_pairs = paired.drop_duplicates()
                distinct_per_entity = (
                    distinct_pairs.groupby(
                        distinct_pairs[entity_col.name], dropna=True, observed=True, sort=False
                    )[measure_col.name]
                    .nunique()
                )
            except (TypeError, ValueError):
                unchecked_measures.setdefault(measure_col.name, "unsupported")
                continue
            if not distinct_per_entity.empty and int(distinct_per_entity.max()) == 1:
                if len(paired) == _col_observed_count(measure_col, row_count):
                    entity_constant_measures.add((entity_col.name, measure_col.name))
                else:
                    partially_covered_measures.add(measure_col.name)

    entity_determinants: set[tuple[str, str]] = set()
    determinant_names = sorted({entity_name for entity_name, _ in entity_constant_measures})
    for det_name in determinant_names:
        for target_col in identifiers:
            if target_col.name == det_name:
                continue
            try:
                id_pairs = frame[[det_name, target_col.name]].dropna().drop_duplicates()
                if id_pairs.empty:
                    continue
                target_per_det = id_pairs.groupby(
                    id_pairs[det_name], dropna=True, observed=True, sort=False
                )[target_col.name].nunique()
            except (TypeError, ValueError):
                continue
            if not target_per_det.empty and int(target_per_det.max()) == 1:
                entity_determinants.add((det_name, target_col.name))

    multi_parent_pairs: set[tuple[str, str]] = set()
    col_names = {col.name for col in columns}
    for rel in relationships:
        if rel.kind == "one_to_many":
            parent_name, child_name = rel.source_column, rel.target_column
        elif rel.kind in {"functional_dependency", "approximate_mapping"}:
            parent_name, child_name = rel.target_column, rel.source_column
        else:
            continue
        if parent_name not in col_names or child_name not in col_names:
            continue
        try:
            paired = frame[[parent_name, child_name]].dropna().drop_duplicates()
            if paired.empty:
                continue
            parents_per_child = paired.groupby(
                paired[child_name], dropna=True, observed=True, sort=False
            )[parent_name].nunique()
        except (TypeError, ValueError):
            continue
        if not parents_per_child.empty and int(parents_per_child.max()) > 1:
            multi_parent_pairs.add((parent_name, child_name))

    return MetricGrainFacts(
        frozenset(entity_constant_measures),
        frozenset(entity_determinants),
        frozenset(multi_parent_pairs),
        frozenset(partially_covered_measures),
        None,
        tuple(sorted(unchecked_measures.items())),
    )
