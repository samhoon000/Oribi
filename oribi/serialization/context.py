"""Compact AI-oriented rendering of an existing context snapshot."""

from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from numbers import Integral, Number
from typing import Any, NamedTuple, SupportsInt

from ..models.evidence import Warning
from .aggregations import aggregation_conditions
from .categories import format_categories


class _OptionalText(NamedTuple):
    text: str
    evidence: str = ""
    values: str = ""


_ContextLine = str | _OptionalText


def prioritize_warnings(warnings: Iterable[Warning]) -> list[Warning]:
    """Return distinct warnings in severity order."""
    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    distinct: list[Warning] = []
    for warning in warnings:
        if warning not in distinct:
            distinct.append(warning)
    return sorted(
        distinct,
        key=lambda warning: (severity_rank.get(warning.severity, 4), warning.code, warning.message),
    )


def _render_with_evidence(sections: Sequence[Sequence[_ContextLine]], max_chars: int) -> str | None:
    """Allocate space to required text, evidence, then optional value details."""
    lines: list[_ContextLine] = []
    for section in sections:
        if lines:
            lines.append("")
        lines.extend(section)
    available = max_chars - (len(lines) - 1)
    available -= sum(len(line if isinstance(line, str) else line.text) for line in lines)
    if available < 0:
        return None

    rendered: list[str] = []
    value_lines: list[tuple[int, str]] = []
    for line in lines:
        if isinstance(line, str):
            rendered.append(line)
        else:
            text = line.text
            if len(line.evidence) <= available:
                text += line.evidence
                available -= len(line.evidence)
            if line.values:
                value_lines.append((len(rendered), line.values))
            rendered.append(text)
    for index, values in value_lines:
        if len(values) <= available:
            rendered[index] += values
            available -= len(values)
    return "\n".join(rendered)


def _compact_column_label(name: Any) -> str:
    """Render a column label on one line for compact context only."""
    return " ".join(str(name).split())


def _format_aggregations(operations: Any) -> str:
    """Render aggregation names without treating a single string as a sequence."""
    if isinstance(operations, str):
        names = [operations] if operations else []
    else:
        names = list(operations)
    return " \u00b7 ".join(names) if names else "\u2014"


def _relationship_lines(context: Any) -> list[str]:
    """Describe how columns relate to records, grain, and other columns."""
    lines = ["RELATIONSHIPS"]
    rendered_composite_keys: set[frozenset[str]] = set()
    for column in sorted(context.columns, key=lambda item: item.name):
        for relation in column.relationships:
            confidence = f"{relation.confidence:.0%} confidence"
            if relation.kind == "identifies_observations":
                evidence = relation.evidence[0].details if relation.evidence else {}
                card = column.cardinality
                unique = card.unique_count if card else evidence.get("distinct_count", column.statistics.distinct_count)
                rows = card.observed_count if card else evidence.get("row_count", context.profile.row_count)
                statement = (
                    "appears to uniquely identify observations"
                    if card and card.uniqueness_ratio == 1 and card.missing_count == 0
                    else "is a mostly-unique observation-key candidate"
                )
                strength = f"{relation.evidence_strength:.0%} uniqueness evidence; " if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {_compact_column_label(column.name)} {statement} "
                    f"({unique:,}/{rows:,} values distinct; {strength}{confidence})"
                )
            elif relation.kind == "jointly_identifies_observations":
                composite_key = frozenset((column.name, *relation.related_columns))
                if composite_key in rendered_composite_keys:
                    continue
                rendered_composite_keys.add(composite_key)
                components = " + ".join(
                    _compact_column_label(part) for part in (column.name, *relation.related_columns)
                )
                evidence = relation.evidence[0].statement if relation.evidence else "observed joint uniqueness"
                strength = f"{relation.evidence_strength:.0%} uniqueness evidence; " if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {components} jointly identify an observation "
                    f"({evidence}; {strength}{confidence})"
                )
            elif relation.kind == "groups_observations":
                card = column.cardinality
                detail = relation.evidence[0].statement if relation.evidence else "repeated values"
                evidence_details = relation.evidence[0].details if relation.evidence else {}
                average_entities = evidence_details.get("average_entities_per_group")
                if relation.related_columns and average_entities is not None:
                    detail = f"one-to-many; about {average_entities:.1f} related observations per group"
                elif card and card.average_records_per_value is not None:
                    detail = (
                        f"{card.unique_count:,} unique values across {card.observed_count:,} "
                        f"non-missing observations; average {card.average_records_per_value:.2f} observations/value"
                    )
                if relation.related_columns:
                    detail += f"; maps observations to {', '.join(_compact_column_label(part) for part in relation.related_columns)}"
                strength = f"; {relation.evidence_strength:.0%} evidence strength" if relation.evidence_strength is not None else ""
                lines.append(
                    f"- {_compact_column_label(column.name)} groups observations ({detail}{strength}; {confidence})"
                )
            elif relation.kind == "measure_at_grain":
                grain = (
                    " + ".join(_compact_column_label(part) for part in relation.related_columns)
                    if relation.related_columns
                    else "the observed record grain"
                )
                evidence = relation.evidence[0].statement if relation.evidence else "numeric observations"
                lines.append(
                    f"- {_compact_column_label(column.name)} is a quantitative measure at {grain} "
                    f"({evidence}; {confidence})"
                )
            elif relation.kind == "temporal_attribute":
                evidence = relation.evidence[0].statement if relation.evidence else "datetime-typed values"
                lines.append(
                    f"- {_compact_column_label(column.name)} is a temporal attribute associated with observations "
                    f"({evidence}; {confidence})"
                )

    one_to_one_pairs = {
        frozenset((relation.source_column, relation.target_column))
        for relation in context.relationships if relation.kind == "one_to_one"
    }
    for relation in context.relationships:
        if relation.kind == "one_to_many":
            continue  # Already summarized from the grouping column.
        if relation.kind == "functional_dependency" and frozenset((relation.source_column, relation.target_column)) in one_to_one_pairs:
            continue
        if relation.kind == "one_to_one":
            evidence = relation.evidence[0].statement if relation.evidence else "observed value mapping"
            lines.append(
                f"- {_compact_column_label(relation.source_column)} ↔ {_compact_column_label(relation.target_column)}: observed one-to-one value mapping "
                f"({evidence}; {relation.coverage:.0%} coverage; {relation.confidence:.0%} confidence)"
            )
        else:
            evidence = relation.evidence[0].statement if relation.evidence else "observed value mapping"
            if relation.kind == "approximate_mapping":
                lines.append(
                    f"- {_compact_column_label(relation.source_column)} → {_compact_column_label(relation.target_column)}: approximate mapping only "
                    f"({relation.affected_groups} conflicting groups; {relation.violation_count} violations; "
                    f"{relation.missing_pair_count} incomplete rows; {relation.coverage:.0%} group coverage)"
                )
                continue
            if relation.mapping_cardinality:
                evidence = f"{relation.mapping_cardinality.replace('_', '-')}; {evidence}"
            lines.append(
                f"- {_compact_column_label(relation.source_column)} → {_compact_column_label(relation.target_column)}: observed functional dependency "
                f"({evidence}; {relation.coverage:.0%} coverage; {relation.confidence:.0%} confidence)"
            )
    return lines if len(lines) > 1 else []


def _column_role_lines(context: Any) -> list[_ContextLine]:
    labels = {
        "identifier": "identifier candidate",
        "dimension": "categorical dimension",
        "categorical": "categorical dimension",
        "measure": "numeric measure",
        "temporal": "temporal dimension",
        "boolean": "boolean/flag",
        "text": "text",
        "unknown": "unknown",
    }
    lines: list[_ContextLine] = ["COLUMN ROLES"]
    for column in sorted(context.columns, key=lambda item: item.name):
        role = labels.get(column.semantic.role, column.semantic.role)
        line = f"- {_compact_column_label(column.name)}: {role} ({column.semantic.confidence:.0%} confidence)"
        basis = next((
            evidence.statement for evidence in column.semantic.evidence
            if evidence.code in {"iso_temporal_values", "temporal_object_values"}
        ), "")
        lines.append(_OptionalText(line, evidence=f" · {basis}") if basis else line)
    return lines if len(lines) > 1 else []


def _cardinality_lines(context: Any) -> list[_ContextLine]:
    lines: list[_ContextLine] = ["CARDINALITY"]
    for column in sorted(context.columns, key=lambda item: item.name):
        card = column.cardinality
        if card is None:
            continue
        line = f"- {_compact_column_label(column.name)}: {card.unique_count:,} unique / {card.row_count:,} rows"
        if card.uniqueness_ratio is None:
            line += "; no observed values"
        else:
            line += f" ({card.uniqueness_ratio:.0%} of observed rows unique)"
            line += f"; {card.observed_count:,} non-missing"
            line += f"; {card.missing_count:,} missing ({card.missing_rate:.0%})"
            line += f"; average {card.average_records_per_value:.2f} records/value"
            line += f" (min {card.minimum_records_per_value}, max {card.maximum_records_per_value})"
        lines.append(_OptionalText(line, evidence=f" · {card.constancy.statement}") if card.constancy else line)
    return lines if len(lines) > 1 else []


def _metric_lines(context: Any) -> list[_ContextLine]:
    metrics = getattr(context, "metrics", ())
    if not metrics:
        return []
    lines: list[_ContextLine] = ["METRICS"]
    for metric in metrics:
        sql = " ".join(metric.sql.splitlines())
        line = f"- {metric.name} ({metric.confidence:.0%} confidence): {sql}"
        details = []
        if metric.rationale:
            details.append(" ".join(metric.rationale.split()))
        if metric.caveats:
            details.append(f"Caveat: {' '.join(metric.caveats[0].split())}")
        evidence = f" · {' '.join(details)}" if details else ""
        lines.append(_OptionalText(line, evidence=evidence) if evidence else line)
    return lines


def _format_range_value(value: Any) -> str:
    """Render observed bounds without redundant midnight or decimal suffixes.

    Timezone-aware midnight datetimes keep their time and offset so the
    actual instant remains unambiguous. Naive midnight values still use
    the date-only form.
    """
    if isinstance(value, datetime):
        # time() drops the sub-microsecond component of pandas timestamps.
        if value.tzinfo is None and value.time() == time.min and getattr(value, "nanosecond", 0) == 0:
            return value.date().isoformat()
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Number) and isinstance(value, SupportsInt) and not isinstance(value, (bool, complex)):
        try:
            integer = int(value)
            if value == integer:
                return str(integer)
        except (OverflowError, TypeError, ValueError):
            pass
    return str(value)


def to_context(context: Any, *, max_chars: int = 8000, include_values: bool = False) -> str:
    if isinstance(max_chars, bool) or not isinstance(max_chars, Integral):
        raise ValueError("max_chars must be an integer")
    max_chars = int(max_chars)
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    p = context.profile
    header = ["DATASET", f"{p.row_count:,} rows \u00d7 {p.column_count:,} columns"]
    if p.exact_row_limit_exceeded:
        header.append("Exact row limit exceeded.")
    essential_header = len(header)
    if p.duplicate_row_count:
        header.append(f"Duplicate rows: {p.duplicate_row_count:,}")
    header.append("Confidence percentages are heuristic scores, not probabilities.")
    if p.index is not None:
        labels = ", ".join(str(name) if name is not None else f"level {i}" for i, name in enumerate(p.index.names))
        header.append(
            f"Index ({labels}): {p.index.unique_count:,} unique / {p.row_count:,} rows; "
            f"{p.index.duplicate_count:,} duplicates; {p.index.missing_count:,} rows with null index values. "
            "Use the DataFrame index or reset_index() to access these fields."
        )

    grain_lines = ["GRAIN"]
    if context.grain and context.grain.status == "estimated":
        grain_columns = " + ".join(_compact_column_label(part) for part in context.grain.columns) or context.grain.label or "Unknown"
        if len(context.grain.columns) > 1:
            description = "candidate composite observation grain"
        elif any(column.name in context.grain.columns and column.logical_type == "temporal" for column in context.columns):
            description = "candidate temporal observation grain"
        else:
            description = "candidate observation grain"
        grain_lines.append(f"{grain_columns} \u00b7 {description} \u00b7 {context.grain.confidence:.0%} confidence")
    else:
        grain_lines.append("Unknown \u00b7 insufficient evidence")

    all_warnings = list(context.warnings)
    for col in context.columns:
        all_warnings.extend(col.warnings)
    warning_lines: list[_ContextLine] = ["WARNINGS"] if all_warnings else []
    printed_warnings: list[Any] = []
    for warning in all_warnings:
        if warning in printed_warnings:
            continue
        if warning.code == "high_missingness":
            column = next((c for c in context.columns if warning in c.warnings), None)
            if column is None:
                continue
            message = f"\u26a0 {_compact_column_label(column.name)} \u00b7 {column.statistics.missing_rate:.0%} missing"
        else:
            message = f"\u26a0 {warning.message}"
        if warning.evidence:
            basis = " ".join(warning.evidence[0].statement.split())
            warning_lines.append(_OptionalText(message, evidence=f" · {basis}"))
        else:
            warning_lines.append(message)
        printed_warnings.append(warning)
    column_lines: list[_ContextLine] = ["COLUMNS"]
    guidance: list[tuple[bool, str]] = []
    ranked = sorted(context.columns, key=lambda c: c.name)
    for col in ranked:
        s = col.statistics
        aggregations = [
            rec for rec in col.recommendations
            if rec.disposition in {"recommended", "caution"}
        ]
        label = _compact_column_label(col.name)
        fields: list[_ContextLine] = [
            "",
            label,
            f"{col.dtype} \u00b7 {col.semantic.role} \u00b7 {col.semantic.confidence:.0%} confidence",
        ]
        stats = f"Missing {s.missing_rate:.0%} \u00b7 Unique {s.distinct_count:,}"
        if include_values and (s.minimum is not None or s.maximum is not None):
            stats += f" \u00b7 Range {_format_range_value(s.minimum)} \u2192 {_format_range_value(s.maximum)}"
        fields.append(stats)
        operations = _format_aggregations(rec.operation for rec in aggregations)
        conditions = aggregation_conditions(aggregations, col.semantic.role)
        aggregation_line = f"Aggregations: {operations}{conditions}"
        agg_basis = list(dict.fromkeys(
            " ".join(evidence.statement.split())
            for rec in aggregations for evidence in rec.evidence
        ))
        if agg_basis:
            fields.append(_OptionalText(aggregation_line, evidence=f"\nAggregation basis: {' · '.join(agg_basis[:2])}"))
        else:
            fields.append(aggregation_line)
        useful = [
            rec for rec in col.recommendations
            if rec.disposition == "useful"
        ]
        if useful:
            fields.append(f"Useful operations: {_format_aggregations(rec.operation for rec in useful)}")
        inappropriate = [
            rec for rec in col.recommendations
            if rec.disposition == "not_applicable"
        ]
        if inappropriate:
            inappropriate_line = f"Inappropriate aggregations: {_format_aggregations(rec.operation for rec in inappropriate)}"
            fields.append(inappropriate_line)
        if aggregations or inappropriate:
            # Keep the restrictions and qualifications attached when space is tight.
            operations_line = aggregation_line if aggregations else ""
            if inappropriate:
                operations_line = f"{operations_line} | {inappropriate_line}" if operations_line else inappropriate_line
            guidance.append((bool(inappropriate) or any(rec.disposition == "caution" for rec in aggregations),
                             f"- {label}: {operations_line}"))
        optional_values = ""
        if include_values and col.categories is not None:
            optional_values = "\n" + format_categories(col.categories, s.distinct_count)
        elif include_values and s.top_values:
            optional_values = "\nLocal value samples: " + ", ".join(f"{v} ({n})" for v, n in s.top_values)
        if optional_values:
            last = fields[-1]
            fields[-1] = _OptionalText(last, values=optional_values) if isinstance(last, str) else last._replace(values=optional_values)
        column_lines.extend(fields)

    sections: list[Sequence[_ContextLine]] = [header, grain_lines]
    role_lines = _column_role_lines(context)
    if role_lines:
        sections.append(role_lines)
    if warning_lines:
        sections.append(warning_lines)
    relationship_lines = _relationship_lines(context)
    if relationship_lines:
        sections.append(relationship_lines)
    cardinality_lines = _cardinality_lines(context)
    if cardinality_lines:
        sections.append(cardinality_lines)
    metric_lines = _metric_lines(context)
    if metric_lines:
        sections.append(metric_lines)
    sections.append(column_lines)
    result = _render_with_evidence(sections, max_chars)
    if result is not None:
        return result
    # Keep the schema, warnings and actionable operations ahead of evidence,
    # detailed cardinality and relationship prose. Retain complete lines only.
    marker = "[Context truncated.]"
    kept = header[:essential_header]
    available = max_chars - len("\n".join((*kept, marker)))

    def add_section(title: str, lines: Sequence[str]) -> None:
        nonlocal available
        heading = ["", title]
        overhead = len(title) + 2
        for line in lines:
            size = overhead + len(line) + 1
            if size <= available:
                kept.extend(heading)
                kept.append(line)
                available -= size
                heading = []
                overhead = 0

    add_section(grain_lines[0], grain_lines[1:])
    add_section("COLUMNS", [
        f"- {_compact_column_label(col.name)}: {col.dtype}/{col.semantic.role}"
        + (f" · {col.statistics.missing_rate:.0%} missing" if col.statistics.missing_count else "")
        for col in ranked
    ])
    if warning_lines:
        prioritized = prioritize_warnings(all_warnings)
        selected_warnings = prioritized[:10]
        messages = [f"⚠ {warning.message}" for warning in selected_warnings]
        if len(prioritized) > len(selected_warnings):
            messages.append(f"{len(prioritized) - len(selected_warnings)} additional warnings omitted.")
        add_section("WARNINGS", messages)
    add_section("AGGREGATION GUIDANCE", [line for _, line in sorted(guidance, key=lambda item: not item[0])])
    if header[essential_header:]:
        add_section("DATASET DETAILS", header[essential_header:])
    if metric_lines:
        add_section("METRICS", [
            line if isinstance(line, str) else line.text for line in metric_lines[1:10]
        ])
    if relationship_lines:
        add_section(relationship_lines[0], relationship_lines[1:10])
    if cardinality_lines:
        add_section("CARDINALITY", [
            line if isinstance(line, str) else line.text for line in cardinality_lines[1:10]
        ])
    return "\n".join((*kept, marker))
