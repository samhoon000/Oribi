# Changelog

All notable changes to Oribi are recorded here.

## Unreleased

### Fixed

- Serialize NumPy datetime samples as ISO timestamps, preserving nanosecond precision in opt-in dictionary and JSON exports.

## 0.3.0 — 2026-11-24

### Added and improved

- Infer evidence-backed analytical metrics in `DatasetContext.metrics`, including entity counts, additive totals, per-entity averages, and ratios. Export metric definitions and SQL with evidence and caveats. (#126)
- Add `DatasetContext.get_column(name)` and allow `ProfileOptions(top_k=0)` to disable top-value sampling. (#99, #101)
- Add categorical value context, constancy evidence, duplicate-check warnings, and a structured indicator for profiles that exceed the exact row limit. Improve numeric role inference and identify more distinct forms of identifier names. (#77, #79, #80, #97, #114)
- Explain aggregation choices in Markdown and compact reports, including why identifiers are unsuitable for sums or averages, when to count distinct entities, and when grouping by dimensions is useful. Preserve cautions for skew, negative values, rates, and unit prices or costs. (#70, #72, #75, #98, #156, #169)

### Fixed

- Preserve every verified one-to-many target in column annotations and compact context, and reserve relationship scan slots for eligible columns. (#117, #131, #147, #153, #176)
- Exclude native elapsed durations from composite keys while preserving datetime keys. Keep near-unique columns eligible for verified composite keys, prevent fallback object counts from verifying keys, and leave grain ambiguous when independent timestamps provide equally strong evidence. (#124, #129, #154, #171, #175, #176)
- Correct metric deduplication when keys have incomplete coverage, keep deduplication caveats alongside SQL, and quote SQL reserved column names. (#145, #151, #173, #174)
- Improve profiling of sparse numbers, nullable integers, decimals, object-backed booleans, pandas periods and intervals, and unhashable index labels. Avoid narrow-dtype overflow and unsupported claims of zero dispersion. (#96, #120, #150, #155, #163, #166, #170, #172, #173)
- Preserve nanosecond datetime bounds and report separate temporal gaps accurately, including local daily schedules across daylight saving. Skip gap detection when timestamp deltas exceed pandas limits. (#116, #121, #122, #139, #167)
- Stabilize top-value ties and detach mutable values nested inside tuple samples. Structured opt-in exports handle extended floats and raise clear errors for circular samples or mapping keys that collide during serialization. (#76, #95, #119, #165, #168)
- Validate profile and compact-report limits, retain useful context when samples exceed the output budget, and render empty or multiline labels and cardinality ratios clearly. (#69, #102, #103, #125, #138, #152, #157)

### Compatibility

- Runtime requirements remain Python 3.10–3.14 and pandas 1.5.3 through 3.x. Structured exports retain schema version `2` and gain analytical metric and profiling fields. Consumers with strict field allowlists should account for the additions. Opt-in structured exports reject mapping-key collisions that could previously overwrite values.

### Documentation and tooling

- Refresh the README and quickstart, including examples for key and grain inspection and for resetting indexes with colliding names. Add Ruff and mypy to CI. (#81, #82, #93, #100, #118, #164)

Thanks to @abhishekrajv7-cmd, @AryanSingh2029, @ashenthefox, @avaldivieso-meraki, @bhuvi-777, @chocolinax, @dev-burnern, @dh1011, @fbirds5230, @flipcyde101, @hnimXgibhut, @IG-ABHINAV, @LuciferAksh, @paulxg12, @samhoon000, @sharmao07, @sudharshan-s-1286, @sundaysebasidian-byte, @swpswpswp, and @TranTrungNhan123 for contributing to this release.

## 0.2.0 — 2026-11-19

### Compatibility

- Dictionary and JSON exports now use schema version `2`. Exports omit sampled cell values and exact bounds by default. Use `include_values=True` for local samples and bounds. Dictionary and JSON exports retain frequency counts. Consumers that parse the previous Markdown or compact context layout should update their parsers.
- `ProfileOptions` validates booleans and integer limits when constructed. Native complex-valued columns raise `InvalidInputError`.

### Added and improved

- Report per-column cardinality, relationships to the observation grain, and one-to-one and one-to-many value mappings. Distinguish exact dependencies from approximate mappings with contradiction and paired-row evidence. Exclude numeric measures from entity grouping inference.
- Report verified unique keys separately from approximate candidates. Colliding or null-bearing key candidates cannot establish grain. Improve role inference for repeated low-diversity codes and numeric columns.
- Profile meaningful pandas indexes separately and improve the Markdown and compact context reports. Compact output retains schema information on wide frames.
- Bound relationship discovery and temporal-gap warnings and examples. Document that the `max_exact_rows` limit includes the boundary for quantiles, composite keys, and observed relationships.
- Warn when a text column has more than 100 distinct values and is not inferred as an identifier.

### Fixed

- Count duplicate indexes consistently with pandas and handle non-string index names and repeated MultiIndex level names.
- Profile real numeric object columns with missing values, exclude infinities from summary moments and quantiles, and preserve exact integer bounds. Keep positional construction of `StatisticsContext` compatible with 0.1.0.
- Exclude unused categories from top values and detach sampled mutable lists and dictionaries from source cells.
- Classify elapsed durations as `duration` and exclude them from single-column key selection. Preserve nanosecond cadences and timezone offsets in temporal output.
- Escape Markdown in column names and opt-in value samples. Show each full analytical warning once.

### Documentation and packaging

- Refresh the README and contributor file map. Validate the wheel and source distribution, test the installed wheel on Python 3.10–3.14, and publish through GitHub Actions when a GitHub Release is published.

## 0.1.0 — Initial release

- Introduce the `oribi` pandas DataFrame profiling API.
- Report column statistics, evidence-backed semantic roles, recommendations, warnings, relationships, candidate keys, and likely grain.
- Add conservative temporal-gap warnings for regular datetime series.
- Provide summary, dictionary, JSON, Markdown, and bounded AI-oriented context output.
