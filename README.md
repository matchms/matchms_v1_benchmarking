# Benchmarks for matchms 1.0

Reproducible scripts supporting the **matchms 1.0** software evaluation.

Two benchmarks are provided:

- **Spectrum handling** (Sections 2–7): compares operations on a Python `list[matchms.Spectrum]` with operations on a `matchms.SpectraCollection` using the same source spectra, including public filtering methods.
- **Similarity computation** (Section 8): measures dense all-vs-all cosine, modified-cosine and entropy scoring, as well as indexed entropy library search.

Both scripts save raw wall-clock timings, and their respective notebooks regenerate tables and figures. Neither benchmark evaluates identification accuracy, end-to-end file ingestion, model inference or total application memory consumption. No production dataset or performance measurements are bundled with the code.

## Contents

| File | Purpose |
| --- | --- |
| `benchmark_matchms_spectra_handling.py` | Command-line benchmark; samples a dataset, validates results and saves timing/provenance files. |
| `benchmarking_analysis_matchms_v1_spectra_handling.ipynb` | Loads **raw timings**, checks run completeness and regenerates manuscript and supplementary figures. |
| `benchmark_matchms_scoring_paper_v2.py` | Dense all-vs-all scoring, index construction and fixed-query entropy library search. |
| `benchmarking_analysis_matchms_v1_similarity_computation.ipynb` | Validates raw scoring timings and recreates manuscript and supplementary figures. |
| `pyproject.toml`, `uv.lock` | Shared environment specification; retain the lockfile used for the paper. |
| `requirements.txt` | Optional minimum environment dependencies, with `matchms==1.0.0` pinned. |
| `tests/test_benchmark_helpers.py` | Lightweight regression tests for benchmark utilities. |
| `tests/test_matchms_benchmark_smoke.py` | Optional spectrum-handling API smoke test against an installed matchms 1.x. |
| `tests/test_scoring_benchmark.py` | Scoring benchmark regression tests; mock APIs test orchestration without real scoring. |
| `tests/test_scoring_real_api.py` | Integration smoke tests against installed matchms and MS-Entropy. |

## 1. Install the environment

Use a dedicated Python environment, preferably **Python 3.13**, and install the released version of matchms used for the preprint:

```bash
uv sync
```

Both scripts check that `matchms` has major version **1**. For the preprint, keep `matchms==1.0.0` pinned. The scoring benchmark also requires the CPU `ms_entropy` package for its external baseline. If it is not already listed in your project dependencies, add it once:

```bash
uv add ms-entropy
```

Commit the resolved `uv.lock` to retain the exact MS-Entropy version and all transitive dependency versions. Use `uv sync --locked` when reproducing the archived environment. The external baseline can be omitted with `--skip-msentropy`; no GPU dependency is required.

Save a full environment snapshot for *each* benchmark run:

```bash
uv pip freeze > environment_freeze.txt
```

Store that snapshot alongside the run's result files (see the production command below). The analysis notebooks do not import matchms or MS-Entropy; they need NumPy, pandas, Matplotlib, IPython and Jupyter.

## 2. Supply an input dataset for spectrum handling

The default input is `matchms_scoring_benchmark_set.mgf` in the current directory. This file is **not included**; obtain it from the study's data repository or provide another MS/MS library. The script calls `matchms.importing.load_spectra(..., metadata_harmonization=True)` to read it. Formats supported by this matchms importer include MGF, MSP, JSON, mzML, mzXML and matchms pickle files; for publication we recommend an openly deposited MGF with a permanent identifier and usage license.

**Data requirements:**

- The input must have at least as many spectra as the largest requested sample size (`100,000` by default).
- At least two selected spectra must have finite positive `precursor_mz` values spanning a usable range. These values are needed for metadata-based quartile filtering.
- For scientifically meaningful list-versus-collection checks, input spectra should have valid, sorted fragment m/z arrays and intensities. The collection's CSR backend discretizes m/z at `--mz-precision` (default `1e-6` Da); peaks colliding in the same bin can be merged. A discrepancy discovered by the benchmark's correctness checks causes an error rather than an unverified speed-up.
- Public filtering cases may reject spectra or peaks based on metadata or intensities. The benchmark verifies agreement of returned rows, peak counts, precursor values, TICs, and sample full fragment rows before timing.

The sample uses a **seeded reservoir sampler** over the entire input file. For streaming formats such as MGF the sampler does not need to hold the full library; a matchms importer for some other formats may itself read the full file into memory. Unlike selecting the first N spectra, every source row has a chance of inclusion. The selected spectra are shuffled once; prefixes of length 100, 1,000, 10,000 and 100,000 form **nested samples**. The exact sample is repeatable for the **same input file, record order, parser and seed**, not independent of input ordering. Save the source-file hash and `sample_selection.npz` to reproduce it.

## 3. Run a small spectrum-handling smoke benchmark

Starting in the directory containing the scripts:

```bash
uv run python benchmark_matchms_spectra_handling.py \
  --dataset /path/to/matchms_scoring_benchmark_set.mgf \
  --output-dir results/smoke \
  --sizes 100,1000 \
  --repeats 2 \
  --basic-only \
  --fail-fast
```

This verifies the simple read-only queries, row selection and direct metadata filtering. `--basic-only` deliberately **omits** the fragment-m/z materialization and public filter/pipeline cases; these are restored in the full run. Note that even a smoke benchmark samples the entire input file once to select spectra uniformly.

## 4. Production spectrum-handling benchmark

Run on an otherwise idle machine. Avoid comparing measurements collected under concurrent CPU-heavy workloads, different software installations, different CPU governor/power settings or different input-library versions. Use **a fresh output directory for each run or machine**.

```bash
uv run python benchmark_matchms_spectra_handling.py \
    --dataset /path/to/matchms_scoring_benchmark_set.mgf \
    --output-dir results/matchms_1_0_0_run_01 \
    --sizes 100,1000,10000,100000 \
    --seed 56789 \
    --repeats 5 \
    --target-repeat-seconds 0.2 \
    --max-inner-loops 10000 \
    --mz-precision 1e-6 \
    --hash-dataset \
    --fail-fast
```

**Important:** `--hash-dataset` reads the entire input once more to compute its SHA256; this work is **outside** the operation timings. The benchmark also excludes dataset loading, sampling, and input collection creation from individual operation timings. It measures `SpectraCollection` construction separately.

### Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset PATH` | `matchms_scoring_benchmark_set.mgf` | Path to the input spectrum file. |
| `--output-dir DIR` | `benchmark_spectra_handling` | Output directory. Must be new or empty unless `--overwrite` is used. |
| `--sizes N,N,...` | `100,1000,10000,100000` | Strictly increasing, unique nested sample sizes. |
| `--seed INT` | `56789` | Seed for reservoir sampling and within-size row selection. |
| `--repeats INT` | `5` | Number of timed blocks per operation **and** representation. |
| `--target-repeat-seconds FLOAT` | `0.20` | Desired timing-block duration for calibrating cheap operations. |
| `--max-inner-loops INT` | `10000` | Maximum calls within a timed block. Expensive cases are additionally capped at 10. |
| `--mz-precision FLOAT` | `1e-6` | Collection fragment-bin resolution in Da. |
| `--basic-only` | off | Omit the more expensive materialization and public filter/pipeline cases. |
| `--hash-dataset` | off | Record SHA256 of the original source file. Recommended for publication. |
| `--fail-fast` | off | Stop when a timed operation fails; recommended for publication. Warmup/validation errors stop the run regardless. |
| `--overwrite` | off | Replace known benchmark files in a nonempty output directory. This **does not merge runs**; use cautiously. |

Use `uv run python benchmark_matchms_spectra_handling.py --help` for the live CLI description.

## 5. What is benchmarked in spectrum handling?

### Queries and handling

1. Extract numeric precursor m/z values; construct a precursor-m/z histogram with fixed bins.
2. Compute per-spectrum peak counts and a peak-count histogram.
3. Compute total ion current per spectrum.
4. Select 10% of rows (contiguous or deterministic random subset, capped at 10,000 rows).
5. Build a precursor-m/z range mask using the 25th and 75th percentiles of the **largest sampled set**; filter rows with that mask.
6. Materialize spectra with peaks in an inclusive 100–500 m/z window.

### Public matchms filters and pipeline

- `require_precursor_mz`
- `normalize_intensities`
- `select_by_relative_intensity(intensity_from=0.001)`
- `remove_peaks_relative_to_precursor_mz`
- `reduce_to_number_of_peaks(n_max=500)`
- Basic peak-cleaning pipeline: normalize → relative-intensity filter → remove precursor region → top-500 reduction.

Individual public filters use their **default clone behavior** on each spectrum/collection. The four-step pipeline starts with **one explicit copy**, then applies the four filters using `clone=False`. Each timed invocation returns new results; the original benchmark inputs remain available for the next repeat. The pipeline is a deliberately small processing example, **not** matchms's full default annotation/preprocessing pipeline.

### Timing and correctness

For every size and case, the program:

1. Constructs a list and a `SpectraCollection` from the same sampled spectrum objects; it times collection construction in a **separate** experiment.
2. Calls and warms up both representations **before** timing, then validates agreement. Per-row peak counts, precursor m/z and TIC are validated across all output spectra; several complete peak rows are also checked.
3. Calibrates the number of inner calls separately for each representation to reduce clock-resolution noise for fast operations. Result inspection happens outside the timed loop so differences in output-container types do not add unrelated overhead.
4. Measures `repeats` timed blocks per representation using `time.perf_counter`. The order of representations alternates between blocks. Garbage collection is invoked just before each block and remains enabled during execution. The same process is used for all cases.
5. Records **seconds per call** as timed block duration divided by inner-loop count. The analysis computes the **median** across outer repeats; raw timings allow independent inspection of variability.

Native BLAS/OpenMP thread-count variables are set to `1` before importing numerical libraries. This makes the comparison single-threaded with respect to those native thread pools; it does **not** guarantee that every library-internal worker or every operating-system scheduler behaves identically.

**Interpretation caution:** Python list row slicing and filtering can simply collect references to existing objects; `SpectraCollection` row selection creates synchronized metadata and fragment storage. `filter_precursor_range` and `require_precursor_mz` have the same issue. These user-level operation times appear in the data and some figures but must not be presented as apples-to-apples kernel measurements. The consolidated figure excludes the two direct row-subset benchmarks and marks other reference-only row filters with `*`.

## 6. Spectrum-handling output files

A successful run creates:

```text
results/matchms_1_0_0_run_01/
├── benchmark_config.json        # settings, run timestamps, versions, hardware, script SHA256
├── sample_selection.json        # sampling method, seed, sizes and range bounds
├── sample_selection.npz         # selected source indices in nested-sample order
├── construction_timings.csv     # raw timed repeats for collection construction
├── construction_summary.csv     # aggregate per size (median, mean, std, min, max)
├── operation_timings.csv        # raw operation timings, validation semantics, output sizes
├── operation_summary.csv        # aggregate per operation, size, representation
├── environment_freeze.txt      # generated by pip freeze; if you add it
└── figures/                    # created when the analysis notebook is executed
    ├── matchms_v1_spectra_handling_benchmarking.png/.pdf/.svg
    ├── scaling_speedups.png/.pdf/.svg
    ├── construction_time.png/.pdf/.svg
    └── largest_size_summary.txt
```

The raw CSVs are appended as each measurement finishes. The summary CSVs are rebuilt after each completed size **and** when execution exits, including after failures. A failed run can therefore have incomplete raw and summary data. The notebook detects incomplete or failed runs and **refuses to plot them**. A run is complete only when all expected `size × case × representation × repeat` combinations exist and have valid positive times.

Relevant raw-timing columns include `category`, `operation`, `representation` (`list_spectrum`/`spectra_collection`), `n_spectra`, `repeat`, `inner_loops`, `seconds_per_call`, `status`, `error`, `semantics` and output-shape summaries. `seconds_per_call` is in **seconds**; the notebook converts it to **milliseconds** where indicated.

`benchmark_config.json` includes the installed `matchms` version and import location, Python/NumPy/pandas/SciPy versions, CPU and platform details, chosen benchmark operations, the absolute source dataset path, optional input SHA256, and the benchmark-script SHA256. Note that the original absolute file path may be machine-specific, and memory/physical CPU details require optional `psutil`.

## 7. Regenerate spectrum-handling tables and figures

The notebook reads the four raw/config files plus `sample_selection.json` and creates several analysis tables and plots.

The notebook recomputes medians from raw rows, checks run completeness, prints dataset/environment provenance, and produces the consolidated figure, construction plot, a three-panel scaling plot, category-wise exploratory plots, timing-variability diagnostics, and an illustrative amortized-conversion analysis. 

**Definition:** `speedup_collection = median_time_list / median_time_collection`. A value of `2` means the collection call took half as much time; `0.5` means the list call took half as much time. 


## 8. Similarity computation benchmarks

### 8.1 Dataset and experiments

The scoring script expects the **already cleaned** MS/MS library `matchms_scoring_benchmark_set.mgf`. It loads the library using `load_ms2_dataset`, with a fragment m/z precision of `1e-6` Da. The default indexed-search experiment requires at least 500,000 source spectra. Selected spectra must have finite positive precursor m/z values, nonempty valid fragments and positive total ion current. Deposit the exact cleaned input and its preparation workflow with the preprint; the script does not reproduce upstream scientific cleaning.

A seeded permutation of source row indices defines the reference order. The default experiments are:

| Experiment | Workload |
| --- | --- |
| Dense all-vs-all | 100, 200, 500, 1,000, 2,000 and 5,000 spectra; three repeats per configuration. |
| Index construction | 1k, 2k, 5k, 10k, 20k, 50k, 100k, 200k, 500k, 1M and 2M reference rows; three builds per configuration. |
| Indexed query search | Ten fixed query spectra against each index; five repeats per configuration. |

All sizes use nested reference prefixes. The 1M and 2M libraries are constructed by repeating the same 500k base collection two and four times, respectively. They are **not** independent sets of one or two million spectra. The ten fixed queries are sampled from that base using `seed + 1`, without removal from the reference pool. This is a throughput benchmark, not an identification experiment.

The all-vs-all methods are `CosineGreedy`, `CosineHungarian`, `CosineLinear`, `CosineBlink`, `Cosine`, `ModifiedCosineGreedy`, `ModifiedCosineHungarian`, `ModifiedCosineLinear`, `ModifiedCosine`, `Entropy` (fragment and hybrid), `EntropySearch`, and the MS-Entropy low-level CPU core. Indexed search compares fragment `Entropy`, `EntropySearch` and MS-Entropy using dense score-only results. The five matchms matrix configurations with explicit parallel execution (`Cosine`, `ModifiedCosine`, both `Entropy` modes and `EntropySearch`) are run with `n_jobs=1` and `n_jobs=-1` by default. All indexed-query timings use `n_jobs=1` / serial CPU execution.

### 8.2 Smoke test before the full run

Run this first from the project directory:

```bash
uv run python benchmark_matchms_similarity_computation.py \
    --smoke \
    --output-dir results/scoring_smoke
```

This generates a deterministic 64-spectrum MGF, exercises all matrix methods at sizes 8 and 16, and tests indexed search at sizes 16, 32, 64 and 128 (including library duplication). It uses one repeat, two workers for parallel matrix cases, fail-fast behavior and full score validation. It tests the public APIs and output pipeline **without loading the large production library**. Its timings are not suitable for publication. Run with `--skip-msentropy` if you only want to test matchms.

The script prints its resolved filename, a script SHA256 prefix, package versions and Python version before loading the dataset. Check these if you suspect the wrong script or environment is being used. The raw configuration records the full SHA256.

### 8.3 Production benchmark

Use a fresh directory distinct from the spectrum-handling run:

```bash
uv run python benchmark_matchms_similarity_computation.puy \
    --dataset /path/to/matchms_scoring_benchmark_set.mgf \
    --output-dir results/scoring_matchms_1_0_0_run_01 \
    --seed 56789 \
    --all-vs-all-repeats 3 \
    --query-index-repeats 3 \
    --query-search-repeats 5 \
    --hash-dataset \
    --fail-fast
```

Then record the environment:

```bash
uv pip freeze > results/scoring_matchms_1_0_0_run_01/environment_freeze.txt
```

Use the same hardware, dataset, package lock and CPU/power settings as the other paper measurements. The full experiment can be slow and memory-intensive: the 2M-reference case materializes a repeated collection and builds large in-memory indexes. A 5,000 × 5,000 dense score matrix alone contains 25 million values (about 100 MB at float32 or 200 MB at float64), before auxiliary arrays. Reduce the largest reference sizes or cap the number of workers if necessary; report the actual settings. `--skip-query` or `--skip-all-vs-all` lets you run an individual experiment in a separate output directory. The combined manuscript figure requires both stages in one complete run.

### 8.4 Command-line options

| Option | Default | Description |
| --- | --- | --- |
| `--dataset PATH` | `matchms_scoring_benchmark_set.mgf` | Cleaned spectrum library. With `--smoke` and no dataset, generate a small MGF. |
| `--output-dir DIR` | `results/similarity` | Output directory; use a fresh directory per run. |
| `--seed INT` | `56789` | Reference permutation and query selection. |
| `--matrix-sizes N,N,...` | `100,200,500,1000,2000,5000` | Strictly increasing, unique matrix sizes. |
| `--reference-sizes N,N,...` | 1k–2M (see above) | Strictly increasing, unique index sizes. Sizes above the base must be exact multiples of it. |
| `--base-reference-size INT` | `500000` | Reference pool size and replication unit; at least 10 and at least the largest active matrix size. |
| `--all-vs-all-repeats INT` | `3` | Timed matrix calls per configuration. |
| `--query-index-repeats INT` | `3` | Timed index builds per configuration. |
| `--query-search-repeats INT` | `5` | Timed searches of the fixed ten-query batch. |
| `--matrix-input {native,collection}` | `native` | Use pre-materialized lists for specialized non-indexed methods and collections for indexed methods; `collection` restores the original all-collection benchmark. |
| `--parallel-jobs INT` | `-1` | Requested parallel worker count; `1` omits parallel matrix measurements. |
| `--skip-all-vs-all` | off | Run index construction and querying only. |
| `--skip-query` | off | Run all-vs-all only; only the required matrix prefix is retained after import. |
| `--skip-msentropy` | off | Omit the external baseline in both experiments. |
| `--validate-values` | off | Inspect all returned score values outside timers (may copy large matrices). Warm-ups always validate values. |
| `--hash-dataset` | off | Record the source-file SHA256 outside the timers. Recommended for publication. |
| `--fail-fast` | off | Stop on the first timed failure; without it, ordinary timing failures are recorded and the run is ultimately marked failed. |
| `--overwrite` | off | Replace known scoring result files. Does not resume or merge runs. |
| `--smoke` | off | Smaller default workload, full score checks and fail-fast. |

Use `uv run python benchmark_matchms_scoring_paper_v2.py --help` for the live CLI description. For an initial real-data run, you can use `--skip-query --matrix-sizes 100,200 --parallel-jobs 2` (the source file is still imported in full).

### 8.5 Timing and interpretation

- **Common input and method settings.** Match tolerance is 0.01 Da. The extra noise cutoff and precursor-region removal are disabled for matchms scorers (BLINK uses `prefilter=False`); entropy scores use float32. Other constructor defaults are kept, with method-specific parameters recorded in the run configuration. The linear cosine methods and `EntropySearch` may merge peaks; BLINK uses discretization. MS-Entropy converts to float32, enforces peak separation with a 0.021 Da cleaning setting, and normalizes intensities to sum to one before core weighting. General entropy's fragment and hybrid scores are different workloads.
- **All-vs-all.** Time the public `matrix()` call, including method-internal preparation and index construction, with a fresh scorer per repeat. In the default `native` input policy, the non-indexed methods receive a list materialized once from the selected collection outside timing. This avoids repeatedly reconstructing `Spectrum` objects in pair-oriented matrix loops. Indexed methods receive the collection directly. All methods therefore start from the same collection-derived spectra. `--matrix-input collection` restores the original input policy; do not mix measurements from the two policies. MS-Entropy's all-vs-all timer includes core construction, index building and complete dense scoring of preconverted records, but excludes reference conversion.
- **Index build and search.** Time reference preparation and index construction via the matchms `build_index()` API. MS-Entropy's one-time reference conversion is recorded separately and excluded from index timings; its low-level index may also build neutral-loss structures. Keep only the final successful build if the final repeat succeeds, releasing previous indexes before the next build. Time each search call using the existing index and the ten raw queries; MS-Entropy query conversion is included each time. No precursor-window, score-threshold or top-k filtering is applied. The timers include any implementation-specific initialization that occurs within the API calls after warm-up.
- **Validation and measurement.** Before timing, warm each selected path, validate shapes and finite score ranges, check serial/parallel agreement on the warm-up matrices, and check that MS-Entropy inputs are unchanged across repeated builds/searches. Each timed result is checked for the expected dense score-only shape; `--validate-values` additionally checks every score. A fresh process limits implicit BLAS/OpenMP thread pools to one thread, while `n_jobs` controls explicit matchms parallelism. `n_jobs=-1` requests the implementation's automatic worker policy, not a guarantee that every logical core is active. Each repeat uses `perf_counter`, with garbage collection before timing. Report median wall-clock times and observed repeat ranges, not confidence intervals.

**Important:** These are native-API runtime comparisons, not tests of score equality between different similarity algorithms or matchms versions. Native preprocessing, peak merging, output precision, assignment rules and reference-conversion costs can differ. For a strictly same-input entropy kernel comparison, a separate experiment on a shared, pre-separated representation would be needed.

### 8.6 Output and plotting

```text
results/scoring_matchms_1_0_0_run_01/
├── benchmark_config.json              # status, settings, methods, versions, hardware, hashes
├── sample_selection.npz               # reference source indices, query positions and source indices
├── representation_preparation.csv     # MS-Entropy reference conversion time and peak counts (if enabled)
├── all_vs_all_timings.csv              # raw matrix repeats
├── all_vs_all_summary.csv
├── query_index_timings.csv             # raw index-build repeats
├── query_index_summary.csv
├── query_search_timings.csv            # raw ten-query batch repeats, seconds_per_query
├── query_search_summary.csv
├── environment_freeze.txt              # if added as above
└── figures/                           # created by the notebook
    ├── matchms_v1_similarity_computation_benchmarking.png/.pdf/.svg
    ├── all_vs_all_cosine.png/.pdf/.svg
    ├── all_vs_all_modified_cosine.png/.pdf/.svg
    ├── all_vs_all_entropy.png/.pdf/.svg
    ├── query_index_build_time.png/.pdf/.svg
    └── ...                            # parallel scaling, query runtime/speedups and analysis tables
```

Files for intentionally omitted stages are not produced. Raw timing files are atomically updated after each repeat; summaries are generated at exit, including ordinary failures. A failed or interrupted run is not automatically resumable. A fatal process termination may leave the configuration marked `running`. Use a fresh output directory to rerun; preserve failed raw CSVs for diagnosis. If you use `--overwrite`, previously generated figures are no longer valid; remove them and regenerate the notebook outputs. The script refuses to overwrite a directory identified as belonging to a different benchmark.

Open `benchmarking_analysis_matchms_v1_similarity_computation.ipynb` in the shared environment:

```bash
uv run jupyter lab
```

Set `RESULTS_DIR` in its first code cell to your scoring run, then restart the kernel and run all cells. The notebook uses the revised script's **schema version 2** output, checks the expected grid of raw rows (including intentionally omitted stages), rejects failed, missing or duplicate repeats and recomputes summaries independently. Old scoring CSVs should be regenerated with the revised script. The main four-panel figure shows serial all-vs-all cosine, modified cosine, entropy and ten-query entropy search. Panels A–C default to **N² dense output cells**, not the number of actual pairwise computations. Squares mark replicated reference libraries. Query time divided by ten is the average time per query within the batch, not independently measured single-query latency. Speedups are ratios of median times.

For optional regression and API smoke tests:

```bash
uv run pytest -q tests/test_scoring_benchmark.py tests/test_scoring_real_api.py
```

## Limitations

These results measure Python API operations under selected workloads and hardware settings. They are not assessments of mass-spectral identification accuracy, end-to-end file ingestion time or total peak-data memory footprint. Different m/z distributions, spectral peak counts, hardware and software versions can change both the relative advantage of list and collection representations and the relative speed of similarity implementations. The notebooks report descriptive timing comparisons, not statistical significance. Archive the input data identifier and checksum, preparation workflow, locked environment, script revision, raw timings and plotting notebooks alongside the preprint.
