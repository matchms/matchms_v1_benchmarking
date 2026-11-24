# Spectrum-handling benchmarks for matchms 1.0

Reproducible scripts supporting the **matchms 1.0** software evaluation of the collection-oriented spectrum representation.

The benchmark compares operations on a Python `list[matchms.Spectrum]` with the corresponding operations on a `matchms.SpectraCollection` using the **same source spectra**. It measures ordinary spectrum handling and public filtering methods, **not** cosine/entropy similarity search, model inference, data loading, or total application memory consumption.

The main outputs are raw wall-clock timings and a separate, reproducible notebook that computes tables and figures. No performance measurements or dataset are bundled with the code in this directory.

## Contents

| File | Purpose |
| --- | --- |
| `benchmark_matchms_spectra_handling.py` | Command-line benchmark; samples a dataset, validates results and saves timing/provenance files. |
| `benchmarking_analysis_matchms_v1_spectra_handling.ipynb` | Loads **raw timings**, checks run completeness and regenerates manuscript and supplementary figures. |
| `requirements.txt` | Minimum environment dependencies, with `matchms==1.0.0` pinned. |
| `tests/test_benchmark_helpers.py` | Lightweight regression tests for benchmark utilities. |
| `tests/test_matchms_benchmark_smoke.py` | Optional API smoke test against an installed matchms 1.x. |

## 1. Install the environment

Use a dedicated Python environment, preferably **Python 3.13**, and install the released version of matchms used for the preprint:

```bash
uv sync
```

The benchmark script checks that `matchms` has major version **1**. To reproduce the published **1.0.0** measurements exactly, keep the pinned `matchms==1.0.0` dependency. Other dependency versions can change results, so save a full environment snapshot for *each* benchmark run:

```bash
uv run python -m pip freeze > environment_freeze.txt
```

Store that snapshot alongside the run's result files (see the production command below). The analysis notebook itself does not import matchms; it only needs NumPy, pandas, Matplotlib, IPython and Jupyter.

## 2. Supply an input dataset

The default input is `matchms_scoring_benchmark_set.mgf` in the current directory. This file is **not included**; obtain it from the study's data repository or provide another MS/MS library. The script calls `matchms.importing.load_spectra(..., metadata_harmonization=True)` to read it. Formats supported by this matchms importer include MGF, MSP, JSON, mzML, mzXML and matchms pickle files; for publication we recommend an openly deposited MGF with a permanent identifier and usage license.

**Data requirements:**

- The input must have at least as many spectra as the largest requested sample size (`100,000` by default).
- At least two selected spectra must have finite positive `precursor_mz` values spanning a usable range. These values are needed for metadata-based quartile filtering.
- For scientifically meaningful list-versus-collection checks, input spectra should have valid, sorted fragment m/z arrays and intensities. The collection's CSR backend discretizes m/z at `--mz-precision` (default `1e-6` Da); peaks colliding in the same bin can be merged. A discrepancy discovered by the benchmark's correctness checks causes an error rather than an unverified speed-up.
- Public filtering cases may reject spectra or peaks based on metadata or intensities. The benchmark verifies agreement of returned rows, peak counts, precursor values, TICs, and sample full fragment rows before timing.

The sample uses a **seeded reservoir sampler** over the entire input file. For streaming formats such as MGF the sampler does not need to hold the full library; a matchms importer for some other formats may itself read the full file into memory. Unlike selecting the first N spectra, every source row has a chance of inclusion. The selected spectra are shuffled once; prefixes of length 100, 1,000, 10,000 and 100,000 form **nested samples**. The exact sample is repeatable for the **same input file, record order, parser and seed**, not independent of input ordering. Save the source-file hash and `sample_selection.npz` to reproduce it.

## 3. Run a small smoke benchmark

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

## 4. Production benchmark for the preprint

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

Use `python benchmark_matchms_spectra_handling.py --help` for the live CLI description.

## 5. What is benchmarked?

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

## 6. Output files

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

## 7. Regenerate tables and figures

The notebook reads the four raw/config files plus `sample_selection.json` and creates several analysis tables and plots.

The notebook recomputes medians from raw rows, checks run completeness, prints dataset/environment provenance, and produces the consolidated figure, construction plot, a three-panel scaling plot, category-wise exploratory plots, timing-variability diagnostics, and a illustrative amortized-conversion analysis. 

**Definition:** `speedup_collection = median_time_list / median_time_collection`. A value of `2` means the collection call took half as much time; `0.5` means the list call took half as much time. 


## Limitations

These results measure Python API operations under one selected workload and hardware setup. They are not an assessment of mass-spectral identification accuracy, end-to-end file ingestion time, peak-matching/similarity scoring, or total peak-data memory footprint. Different m/z distributions, spectral peak counts, hardware and software versions can change the relative advantage of list and collection representations. The notebook reports descriptive timing comparisons; it does not claim statistical significance from five repeats.
