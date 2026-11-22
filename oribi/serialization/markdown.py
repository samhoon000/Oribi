"""Human-readable context report."""

import re
from typing import Any

from .aggregations import aggregation_conditions
from .categories import format_categories

_MARKDOWN_SPECIALS = re.compile(r"[\\`*\[\]<>~|]|(?<![0-9A-Za-z])_|_(?![0-9A-Za-z])")
_LINE_BREAKS = re.compile(r"\s*[\r\n]+\s*")


def _md(value: Any) -> str:
    """Render a column name, value, or message as literal Markdown text.

    Markdown control characters are backslash-escaped and line breaks are
    collapsed to a single space, so a column name cannot start a new block or
    change the formatting around it. Underscores between letters or digits
    (``order_id``) are left alone because they never create emphasis.
    """
    text = _LINE_BREAKS.sub(" ", str(value))
    text = _MARKDOWN_SPECIALS.sub(lambda match: "\\" + match.group(0), text)
    return "\\" + text if text.startswith("#") else text


def _md_column_name(name: str) -> str:
    """Render empty column names with a visible Markdown-only label."""
    return "(empty)" if name == "" else _md(name)


def to_markdown(context: Any, *, include_values: bool = False) -> str:
    profile = context.profile
    lines = [
        "# Dataset context",
        "",
        "## Dataset summary",
        "",
        f"- Rows: {profile.row_count:,}",
        f"- Columns: {profile.column_count:,}",
    ]

    if profile.duplicate_row_count is not None:
        lines.append(f"- Duplicate rows: {profile.duplicate_row_count:,}")

    if profile.exact_row_limit_exceeded:
        lines.append("- Exact row limit exceeded.")

    if profile.index is not None:
        index = profile.index
        names = ", ".join(
            _md(name) if name is not None else f"level {i}"
            for i, name in enumerate(index.names)
        )
        lines.append(
            f"- DataFrame index ({names}): {index.unique_count:,} unique; "
            f"{index.duplicate_count:,} duplicates; "
            f"{index.missing_count:,} rows with null index values. "
            "Use the index or reset_index() to access its fields."
        )

    lines += ["", "## Grain", ""]

    if context.grain and context.grain.status == "estimated":
        label = (
            " + ".join(_md_column_name(c) for c in context.grain.columns)
            or _md(context.grain.label)
        )
        lines.append(
            f"**{label}** · candidate observation grain · "
            f"confidence {context.grain.confidence:.0%}"
        )
    else:
        lines.append("Unknown · insufficient evidence")

    if context.grain and context.grain.evidence:
        lines += ["", "<details>", "<summary>Grain inference evidence</summary>", ""]
        lines.extend(f"- {_md(e.statement)}" for e in context.grain.evidence)
        lines += ["", "</details>"]

    role_labels = {
        "identifier": "identifier candidate",
        "dimension": "categorical dimension",
        "categorical": "categorical dimension",
        "measure": "numeric measure",
        "temporal": "temporal dimension",
        "boolean": "boolean/flag",
        "text": "text",
        "unknown": "unknown",
    }

    lines += ["", "## Column roles", ""]
    lines.extend(
        f"- **{_md_column_name(column.name)}**: "
        f"{role_labels.get(column.semantic.role, column.semantic.role)} "
        f"({column.semantic.confidence:.0%} confidence)"
        for column in sorted(context.columns, key=lambda item: item.name)
    )

    if context.relationships or any(
        column.relationships for column in context.columns
    ):
        lines += ["", "## Relationships", ""]
        rendered_composite_keys: set[frozenset[str]] = set()

        for column in sorted(context.columns, key=lambda item: item.name):
            for relationship in column.relationships:
                statement = (
                    _md(relationship.evidence[0].statement)
                    if relationship.evidence
                    else "observed structural evidence"
                )
                strength = (
                    f"; {relationship.evidence_strength:.0%} evidence strength"
                    if relationship.evidence_strength is not None
                    else ""
                )

                if relationship.kind == "identifies_observations":
                    exact = (
                        column.cardinality is not None
                        and column.cardinality.uniqueness_ratio == 1
                        and column.cardinality.missing_count == 0
                    )
                    description = (
                        "appears to identify observations"
                        if exact
                        else "is a mostly-unique observation-key candidate"
                    )
                    lines.append(
                        f"- **{_md_column_name(column.name)}** {description} "
                        f"({statement}{strength}; "
                        f"{relationship.confidence:.0%} confidence)."
                    )

                elif relationship.kind == "jointly_identifies_observations":
                    composite_key = frozenset(
                        (column.name, *relationship.related_columns)
                    )
                    if composite_key in rendered_composite_keys:
                        continue

                    rendered_composite_keys.add(composite_key)
                    components = " + ".join(
                        _md_column_name(c)
                        for c in (column.name, *relationship.related_columns)
                    )
                    lines.append(
                        f"- **{components}** jointly identify observations "
                        f"({statement}{strength}; "
                        f"{relationship.confidence:.0%} confidence)."
                    )

                elif relationship.kind == "groups_observations":
                    lines.append(
                        f"- **{_md_column_name(column.name)}** groups observations "
                        f"({statement}{strength}; "
                        f"{relationship.confidence:.0%} confidence)."
                    )

                elif relationship.kind == "measure_at_grain":
                    grain = (
                        " + ".join(
                            _md_column_name(c)
                            for c in relationship.related_columns
                        )
                        if relationship.related_columns
                        else "the observed record grain"
                    )
                    lines.append(
                        f"- **{_md_column_name(column.name)}** is a quantitative "
                        f"measure at {grain} ({statement}{strength}; "
                        f"{relationship.confidence:.0%} confidence)."
                    )

                elif relationship.kind == "temporal_attribute":
                    lines.append(
                        f"- **{_md_column_name(column.name)}** is a temporal "
                        f"attribute associated with observations "
                        f"({statement}{strength}; "
                        f"{relationship.confidence:.0%} confidence)."
                    )

        one_to_one_pairs = {
            frozenset((relation.source_column, relation.target_column))
            for relation in context.relationships
            if relation.kind == "one_to_one"
        }

        for relation in context.relationships:
            if (
                relation.kind == "functional_dependency"
                and frozenset(
                    (relation.source_column, relation.target_column)
                ) in one_to_one_pairs
            ):
                continue

            statement = (
                _md(relation.evidence[0].statement)
                if relation.evidence
                else "observed value mapping"
            )
            detail = (
                f"{statement} · {relation.coverage:.1%} coverage · "
                f"{relation.confidence:.0%} confidence"
            )

            if relation.kind == "functional_dependency" and relation.mapping_cardinality:
                detail = (
                    f"{relation.mapping_cardinality.replace('_', '-')} · "
                    f"{detail}"
                )

            if relation.kind == "approximate_mapping":
                detail += (
                    f" · {relation.affected_groups} conflicting groups"
                    f" · {relation.violation_count} violations"
                    f" · {relation.missing_pair_count} incomplete rows"
                )

            if relation.kind == "one_to_many" and relation.evidence:
                average = relation.evidence[0].details.get(
                    "average_entities_per_group"
                )
                if average is not None:
                    detail += (
                        f" · average {average:.2f} "
                        f"{_md_column_name(relation.target_column)} values per "
                        f"{_md_column_name(relation.source_column)}"
                    )

            lines.append(
                f"- **{_md_column_name(relation.source_column)} → "
                f"{_md_column_name(relation.target_column)}: "
                f"{relation.kind}** ({detail})"
            )

    lines += ["", "## Columns", ""]

    for col in context.columns:
        s = col.statistics

        lines += [
            f"### {_md_column_name(col.name)}",
            "",
            f"**{col.dtype}** ({col.logical_type}) · "
            f"**{col.semantic.role}** "
            f"(confidence {col.semantic.confidence:.2f})",
            "",
            f"- Missing: {s.missing_rate:.1%} ({s.missing_count:,})",
            f"- Unique: {s.distinct_count:,}",
        ]

        if col.cardinality is not None:
            card = col.cardinality

            if card.uniqueness_ratio is None:
                lines.append(
                    f"- Cardinality: no observed values across "
                    f"{card.row_count:,} rows"
                )
            else:
                lines.append(
                    f"- Cardinality: {card.unique_count:,} unique across "
                    f"{card.observed_count:,} observed values "
                    f"({card.uniqueness_ratio:.1%} uniqueness); "
                    f"{card.average_records_per_value:.2f} records/value "
                    f"(min {card.minimum_records_per_value}, "
                    f"max {card.maximum_records_per_value}); "
                    f"{card.missing_count:,} missing "
                    f"({card.missing_rate:.1%})"
                )

            if card.constancy is not None:
                lines.append(f"- {_md(card.constancy.statement)}")

        if include_values and (
            s.minimum is not None or s.maximum is not None
        ):
            lines.append(
                f"- Range: {_md(s.minimum)} to {_md(s.maximum)}"
            )

        if include_values and col.categories is not None:
            lines.append(
                f"- {format_categories(col.categories, s.distinct_count, escape=_md)}"
            )
        elif include_values and s.top_values:
            samples = ", ".join(
                f"{_md(value)} ({count})"
                for value, count in s.top_values
            )
            lines.append(f"- Local value samples: {samples}")

        aggregations = [
            rec
            for rec in col.recommendations
            if rec.disposition in {"recommended", "caution"}
        ]
        operations = (
            ", ".join(rec.operation for rec in aggregations) or "None"
        )
        conditions = aggregation_conditions(
            aggregations, col.semantic.role
        )
        lines.append(
            f"- Recommended aggregations: {operations}{conditions}"
        )

        useful = [
            rec.operation
            for rec in col.recommendations
            if rec.disposition == "useful"
        ]
        if useful:
            lines.append(f"- Useful operations: {', '.join(useful)}")

        inappropriate = [
            rec.operation
            for rec in col.recommendations
            if rec.disposition == "not_applicable"
        ]
        if inappropriate:
            lines.append(
                f"- Inappropriate aggregations: {', '.join(inappropriate)}"
            )

        if col.warnings:
            lines.extend(
                f"- Warning [{warning.severity}]: "
                f"{_md(warning.title)} (see Analytical warnings)"
                for warning in col.warnings
            )

        if col.semantic.evidence:
            lines += [
                "",
                "<details>",
                "<summary>Inference evidence</summary>",
                "",
            ]
            lines.extend(
                f"- {_md(e.statement)}"
                for e in col.semantic.evidence
            )
            lines += ["", "</details>"]

        lines.append("")

    if context.warnings:
        lines += ["", "## Analytical warnings", ""]

        for warning in context.warnings:
            lines.append(
                f"- **{_md(warning.title)}** ({warning.severity}): "
                f"{_md(warning.message)}"
            )
            if warning.evidence:
                lines.append(
                    f"  - Basis: {_md(warning.evidence[0].statement)}"
                )
            if warning.implication:
                lines.append(
                    f"  - Implication: {_md(warning.implication)}"
                )
            if warning.recommendation:
                lines.append(
                    f"  - Suggested: {_md(warning.recommendation)}"
                )

    return "\n".join(lines).strip()