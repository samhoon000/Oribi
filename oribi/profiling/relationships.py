"""Conservative within-frame column dependency detection."""

import math

import pandas as pd

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
        if relation.kind not in {existing.kind for existing in inferred[name]}:
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
            has_group_relation = any(rel.kind == "groups_observations" for rel in inferred[name])
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
        elif col_semantic.role == "identifier" and cardinality.unique_count > 1:
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
    eligible_sources = [
        name for name, role in roles.items()
        if role in {"identifier", "dimension", "categorical", "boolean", "text", "temporal"}
    ]
    eligible_targets = [name for name, role in roles.items() if role != "unknown"]
    max_distinct = min(500, max(20, len(frame) // 2))
    eligible_sources = [name for name in eligible_sources if _nunique(frame[name]) <= max_distinct]
    eligible_targets = [name for name in eligible_targets if _nunique(frame[name]) <= max_distinct]
    def rank(name: str) -> tuple[int, str]:
        return (_nunique(frame[name]), name)

    eligible_sources = sorted(eligible_sources, key=rank)[:search_columns]
    eligible_targets = sorted(eligible_targets, key=rank)[:search_columns]
    results: list[RelationshipContext] = []
    for source in eligible_sources:
        for target in eligible_targets:
            if source == target:
                continue
            try:
                pairs = frame[[source, target]].dropna().drop_duplicates()
            except (TypeError, ValueError):
                continue
            source_values = _nunique(frame[source])
            if source_values < 2:
                continue
            source_observed = int(frame[source].notna().sum())
            if source_values >= source_observed:
                # A unique determinant maps trivially to every row-level value.
                continue
            mapping_counts = pairs.groupby(source, dropna=True, observed=True, sort=False)[target].nunique()
            unique_mapping = int(mapping_counts.eq(1).sum())
            coverage = unique_mapping / source_values
            affected_groups = int(mapping_counts.gt(1).sum())
            violations = int((mapping_counts - 1).clip(lower=0).sum())
            paired_rows = int(frame[[source, target]].notna().all(axis=1).sum())
            missing_pairs = len(frame) - paired_rows
            target_values = _nunique(frame[target])
            if coverage >= .98 and target_values > 1:
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
        if (
            distinct >= int(frame[source].notna().sum())
            or distinct >= int(frame[target].notna().sum())
        ):
            # A bijection between unique columns is only row alignment, not
            # meaningful value-level evidence of a reusable relationship.
            continue
        coverage = min(forward.coverage, reverse.coverage)
        observed_count = min(int(frame[source].notna().sum()), int(frame[target].notna().sum()))
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
    eligible_groups = sorted(
        (name for name, role in roles.items() if role in {"identifier", "dimension", "categorical", "boolean"}),
        key=lambda name: (roles[name] != "identifier", rank(name)),
    )[:search_columns]
    for entity in sorted(single_keys)[:search_columns]:
        entity_count = _nunique(frame[entity])
        if entity_count < 2:
            continue
        for group in sorted(eligible_groups):
            if group == entity:
                continue
            group_count = _nunique(frame[group])
            if group_count < 2 or group_count >= entity_count:
                continue
            try:
                pairs = frame[[group, entity]].dropna().drop_duplicates()
            except (TypeError, ValueError):
                continue
            if pairs.empty:
                continue
            entity_group_counts = pairs.groupby(entity, dropna=True, observed=True, sort=False)[group].nunique()
            coverage = float(entity_group_counts.eq(1).sum() / entity_count)
            if coverage < 1.0:
                continue
            per_group = pairs.groupby(group, dropna=True, observed=True, sort=False)[entity].nunique()
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
