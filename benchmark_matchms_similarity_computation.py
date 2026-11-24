#!/usr/bin/env python3
"""Benchmark dense spectral similarity matrices and indexed entropy search.

The default workload covers the two similarity experiments for the matchms 1.0 paper:
all-vs-all score matrices (100--5,000 spectra) and ten-query library search
(1,000--2,000,000 reference rows). All timings are wall-clock seconds; library
index construction and query execution are measured separately. See README.md
for timing boundaries, preprocessing differences and reproducibility details.
Run --smoke first to exercise both experiments on a small generated MGF.
"""
from __future__ import annotations

import os

# Set before numerical imports. Use a fresh process, not an existing notebook.
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
if __name__ == "__main__":
    for name in THREAD_VARIABLES:
        os.environ[name] = "1"

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

MZ_TOLERANCE_DA = 0.01
MZ_PRECISION = 1e-6
ENTROPY_DTYPE = np.float32
ALL_VS_ALL_SIZES = (100, 200, 500, 1_000, 2_000, 5_000)
QUERY_REFERENCE_SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
                         100_000, 200_000, 500_000, 1_000_000, 2_000_000)
BASE_REFERENCE_SIZE = 500_000
N_QUERY_SPECTRA = 10
WARMUP_SIZE = 16
COMMON = dict(tolerance=MZ_TOLERANCE_DA, noise_cutoff=0.0, remove_precursor=False)
MSENTROPY_SETTINGS = dict(precursor_ions_removal_da=None, noise_threshold=0.0,
                          min_ms2_difference_in_da=2.1 * MZ_TOLERANCE_DA, max_peak_num=0)
MSENTROPY_METHOD = "msentropy_core_fragment_cpu"
SUMMARY_GROUPS = {
    "all_vs_all": ["family", "method", "n_spectra", "n_jobs", "execution",
                   "representation_preparation_included", "input_representation"],
    "query_index": ["method", "n_reference", "n_jobs", "execution", "duplication_factor",
                    "library_preparation_included"],
    "query_search": ["method", "n_reference", "n_queries", "n_jobs", "execution",
                     "duplication_factor", "query_preparation_included", "index_build_included"],
}


@dataclass(frozen=True)
class MatrixMethod:
    name: str
    family: str
    factory: Callable
    kwargs: dict
    parallel: bool = False

    def create(self):
        return self.factory(**self.kwargs)

    def input_representation(self, args):
        return "spectra_collection" if self.parallel or args.matrix_input == "collection" else "list_spectrum"

    def jobs(self, parallel_jobs):
        return (1, parallel_jobs) if self.parallel and parallel_jobs != 1 else (1,)


def build_matrix_methods():
    """Explicit public API registry; no silent fallback to different algorithms."""
    from matchms.similarity import (
        Cosine, CosineBlink, CosineGreedy, CosineHungarian, CosineLinear,
        ModifiedCosine, ModifiedCosineGreedy, ModifiedCosineHungarian,
        ModifiedCosineLinear, Entropy, EntropySearch,
    )
    return [
        MatrixMethod("cosine_hungarian", "cosine", CosineHungarian, dict(COMMON)),
        MatrixMethod("cosine_greedy", "cosine", CosineGreedy, dict(COMMON)),
        MatrixMethod("cosine", "cosine", Cosine, dict(COMMON, use_hungarian=False), True),
        MatrixMethod("cosine_blink", "cosine", CosineBlink,
                     dict(tolerance=MZ_TOLERANCE_DA, prefilter=False)),
        MatrixMethod("cosine_linear", "cosine", CosineLinear, dict(COMMON)),
        MatrixMethod("modified_cosine_hungarian", "modified_cosine", ModifiedCosineHungarian, dict(COMMON)),
        MatrixMethod("modified_cosine_greedy", "modified_cosine", ModifiedCosineGreedy, dict(COMMON)),
        MatrixMethod("modified_cosine", "modified_cosine", ModifiedCosine,
                     dict(COMMON, use_hungarian=False), True),
        MatrixMethod("modified_cosine_linear", "modified_cosine", ModifiedCosineLinear, dict(COMMON)),
        MatrixMethod("entropy_search", "entropy", EntropySearch,
                     dict(COMMON, dtype=ENTROPY_DTYPE, peak_separation="merge"), True),
        MatrixMethod("entropy_fragment", "entropy", Entropy,
                     dict(COMMON, dtype=ENTROPY_DTYPE, matching_mode="fragment"), True),
        MatrixMethod("entropy_hybrid", "entropy", Entropy,
                     dict(COMMON, dtype=ENTROPY_DTYPE, matching_mode="hybrid"), True),
    ]


def query_methods(methods):
    return [("matchms_entropy_fragment", next(m for m in methods if m.name == "entropy_fragment")),
            ("matchms_entropy_search_fragment", next(m for m in methods if m.name == "entropy_search"))]


def matrix_call(method, scorer, spectra, n_jobs):
    kwargs = dict(score_fields=("score",), progress_bar=False)
    if method.parallel:
        kwargs["n_jobs"] = n_jobs
    return scorer.matrix(spectra, **kwargs)


def indexed_search(scorer, queries, index):
    return scorer.search(queries, index, score_fields=("score",), progress_bar=False, n_jobs=1)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path, frame):
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timed_call(function):
    gc.collect()
    start = perf_counter()
    result = function()
    seconds = perf_counter() - start
    if not np.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid elapsed time: {seconds}")
    return result, seconds


def validate_scores(result, shape, *, full=False):
    """Always check shape/score-only dense output; inspect values in warm-ups.

    Scores.to_array() can copy the entire matrix. Do not do that for production
    repeats unless explicitly requested. Numeric validation is outside timers.
    """
    if getattr(result, "shape", None) != shape:
        raise RuntimeError(f"Expected shape {shape}, got {getattr(result, 'shape', None)}")
    if isinstance(result, np.ndarray):
        if result.dtype.kind != "f":
            raise RuntimeError("Expected floating-point score output.")
        array = result
    else:
        if tuple(result.score_fields) != ("score",) or result.is_sparse:
            raise RuntimeError("Expected a dense score-only Scores result.")
        array = result.to_array() if full else None
    if full:
        if array.shape != shape or array.dtype.kind != "f":
            raise RuntimeError(f"Expected floating-point scores of shape {shape}.")
        lo, hi = float(np.min(array)), float(np.max(array))
        if not np.isfinite([lo, hi]).all() or lo < -1e-4 or hi > 1.0001:
            raise RuntimeError(f"Nonfinite or out-of-range scores: min={lo}, max={hi}")
    return array


class TimingTable:
    """Small, atomically replaced raw CSV; each completed repeat is saved."""

    def __init__(self, output_dir, name, fail_fast):
        self.path = output_dir / f"{name}_timings.csv"
        self.rows = []
        self.fail_fast = fail_fast

    def record(self, base, function, validate=lambda result: None):
        result = None
        try:
            result, seconds = _timed_call(function)
            validate(result)
        except Exception as exc:
            self.failure(base, f"{type(exc).__name__}: {exc}")
            if self.fail_fast or isinstance(exc, MemoryError):
                raise
            return None
        extra = {"seconds_per_query": seconds / base["n_queries"]} if "n_queries" in base else {}
        self.rows.append({**base, "seconds": seconds, **extra, "status": "ok"})
        self.save()
        print(f"    repeat {base['repeat'] + 1}: {seconds:.6f} s", flush=True)
        return result

    def failure(self, base, error):
        extra = {"seconds_per_query": np.nan} if "n_queries" in base else {}
        self.rows.append({**base, "seconds": np.nan, **extra, "status": error})
        self.save()
        print(f"    ERROR: {error}", flush=True)

    def save(self):
        _atomic_csv(self.path, pd.DataFrame(self.rows))


def write_summary_tables(output_dir):
    for name, columns in SUMMARY_GROUPS.items():
        raw = output_dir / f"{name}_timings.csv"
        if not raw.exists():
            continue
        data = pd.read_csv(raw)
        data = data.loc[data.status.eq("ok") & np.isfinite(data.seconds) & data.seconds.gt(0)]
        summary = data.groupby(columns, dropna=False)["seconds"].agg(
            n_repeats="count", seconds_median="median", seconds_mean="mean",
            seconds_std="std", seconds_min="min", seconds_max="max").reset_index()
        _atomic_csv(output_dir / f"{name}_summary.csv", summary)


def precursor_values(collection):
    """Read-only safe under pandas Copy-on-Write; never write to a pandas view."""
    metadata = collection.metadata
    if "precursor_mz" not in metadata.columns:
        raise ValueError("Dataset is missing precursor_mz metadata.")
    values = pd.to_numeric(metadata["precursor_mz"], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan)
    invalid = ~np.isfinite(values) | (values <= 0)
    if invalid.any():
        raise ValueError(f"Found {int(invalid.sum())} nonfinite/nonpositive precursor m/z values.")
    return values


def dataset_statistics(collection):
    precursor_values(collection)
    counts = np.asarray(collection.fragments.count(axis=1)).ravel()
    if counts.size != len(collection) or np.any(counts <= 0):
        raise ValueError("Selected library contains empty spectra or invalid fragment row counts.")
    tic = np.asarray(collection.fragments.sum(axis=1)).ravel()
    if tic.size != len(collection) or not np.isfinite(tic).all() or np.any(tic <= 0):
        raise ValueError("Selected library contains spectra with nonfinite/nonpositive TIC.")
    return dict(n_spectra=len(collection), n_peaks=int(counts.sum()),
                min_peaks=int(counts.min()), median_peaks=float(np.median(counts)),
                mean_peaks=float(counts.mean()), max_peaks=int(counts.max()))


def load_benchmark_dataset(path, args):
    """Preserve the original seeded permutation and fixed-query selection."""
    from matchms.importing import load_ms2_dataset
    print(f"Loading dataset: {path}", flush=True)
    start = perf_counter()
    dataset = load_ms2_dataset(str(path), mz_precision=MZ_PRECISION)
    seconds = perf_counter() - start
    loaded = len(dataset)
    required = max(args.matrix_sizes) if args.skip_query else args.base_reference_size
    if loaded < required:
        raise ValueError(f"Need at least {required:,} spectra; found {loaded:,}.")
    indices = np.random.default_rng(args.seed).permutation(loaded)[:required].copy()
    base = dataset[indices]
    del dataset
    gc.collect()
    if args.skip_query:
        positions = np.empty(0, dtype=np.int64)
        queries = None
    else:
        positions = np.sort(np.random.default_rng(args.seed + 1).choice(
            args.base_reference_size, size=N_QUERY_SPECTRA, replace=False))
        queries = base[positions]
    print(f"Loaded {loaded:,} spectra in {seconds:.2f} s; selected {len(base):,}.", flush=True)
    return base, queries, indices, positions, loaded, seconds


def duplication_factor(n, base_size):
    if n <= 0 or base_size <= 0:
        raise ValueError("Reference sizes must be positive.")
    if n <= base_size:
        return 1
    if n % base_size:
        raise ValueError("Reference sizes above the base must be exact multiples of it.")
    return n // base_size


def make_reference_collection(base, n):
    factor = duplication_factor(n, len(base))
    if factor == 1:
        return base if n == len(base) else base[:n]
    from matchms import SpectraCollection
    return SpectraCollection.concat([base] * factor)


def make_msentropy_reference_records(base, n):
    factor = duplication_factor(n, len(base))
    return base[:n] if factor == 1 else base * factor


def import_msentropy():
    from ms_entropy import FlashEntropySearch
    from ms_entropy.entropy_search.flash_entropy_search_core import FlashEntropySearchCore
    return FlashEntropySearch, FlashEntropySearchCore


def make_msentropy_cleaner():
    cleaner, _ = import_msentropy()
    return cleaner(max_ms2_tolerance_in_da=MZ_TOLERANCE_DA)


def convert_msentropy_records(collection, cleaner, *, description=None):
    """Convert to float32, separated, sum-to-one peaks; no benchmark timers here."""
    precursor = precursor_values(collection)
    fragments = collection.fragments
    records, input_peaks, output_peaks = [], 0, 0
    iterator = range(len(collection))
    if description:
        iterator = tqdm(iterator, desc=description, unit="spectra")
    for i in iterator:
        mz, intensities = fragments.get_row(i)
        peaks = np.empty((len(mz), 2), dtype=np.float32)
        peaks[:, 0], peaks[:, 1] = mz, intensities
        cleaned = np.asarray(cleaner.clean_spectrum_for_search(
            precursor_mz=float(precursor[i]), peaks=peaks, **MSENTROPY_SETTINGS), dtype=np.float32)
        if cleaned.ndim != 2 or cleaned.shape[1] != 2 or not len(cleaned):
            raise ValueError(f"MS-Entropy cleaning returned invalid/empty peaks at row {i}.")
        # The low-level core requires positive, normalized, separated peaks.
        # Validate reference conversion outside the timers, not every query call.
        if description and (not np.isfinite(cleaned).all() or np.any(cleaned <= 0)
                            or abs(float(cleaned[:, 1].sum()) - 1) >= 1e-4
                            or np.any(np.diff(cleaned[:, 0]) <= 2 * MZ_TOLERANCE_DA)):
            raise ValueError(f"MS-Entropy core peak requirements violated at row {i}.")
        records.append(dict(precursor_mz=float(precursor[i]), peaks=cleaned))
        input_peaks += len(mz)
        output_peaks += len(cleaned)
    return records, input_peaks, output_peaks


def prepare_msentropy_records(collection, cleaner):
    start = perf_counter()
    records, n_in, n_out = convert_msentropy_records(
        collection, cleaner, description="MS-Entropy reference preparation")
    return records, dict(representation="msentropy_precleaned", n_spectra=len(records),
                         input_peaks=n_in, output_peaks=n_out, mean_input_peaks=n_in/len(records),
                         mean_output_peaks=n_out/len(records), seconds=perf_counter()-start,
                         **MSENTROPY_SETTINGS)


def build_msentropy_core(records):
    _, cls = import_msentropy()
    core = cls(max_ms2_tolerance_in_da=MZ_TOLERANCE_DA, intensity_weight="entropy")
    core.build_index(records)
    if core.total_spectra_num != len(records):
        raise RuntimeError("MS-Entropy index changed the number of reference rows.")
    return core


def msentropy_open_search(core, records):
    result = np.empty((len(records), core.total_spectra_num), dtype=np.float32)
    for i, query in enumerate(records):
        scores = core.search(method="open", target="cpu", peaks=query["peaks"],
                             ms2_tolerance_in_da=MZ_TOLERANCE_DA)
        # Reject a scalar or incorrectly shaped output instead of broadcasting.
        if getattr(scores, "shape", None) != (core.total_spectra_num,):
            raise RuntimeError("MS-Entropy returned an unexpected score-vector shape.")
        result[i] = scores
    return result


def msentropy_query(core, queries, cleaner):
    records, _, _ = convert_msentropy_records(queries, cleaner)
    return msentropy_open_search(core, records)


def msentropy_all_vs_all(records):
    return msentropy_open_search(build_msentropy_core(records), records)


def warm_up(base, queries, methods, args, cleaner):
    """Exercise selected API paths, including repeated use of MS-Entropy inputs.

    Score equivalence across different algorithms is deliberately not asserted.
    Serial/parallel parity is checked for each individual matchms method.
    """
    small = base[:min(WARMUP_SIZE, len(base))]
    n = len(small)
    print("\nWarming and validating selected similarity APIs...", flush=True)
    if not args.skip_all_vs_all:
        spectrum_list = list(small) if args.matrix_input == "native" else None
        for method in methods:
            matrix_input = spectrum_list if method.input_representation(args) == "list_spectrum" else small
            serial = None
            for jobs in method.jobs(args.parallel_jobs):
                print(f"  {method.name}, n_jobs={jobs}", flush=True)
                scorer = method.create()
                result = matrix_call(method, scorer, matrix_input, jobs)
                values = validate_scores(result, (n, n), full=True)
                if serial is None:
                    serial = values
                else:
                    np.testing.assert_allclose(values, serial, atol=2e-5, rtol=2e-5)
                del result, values, scorer
            del serial
        del spectrum_list, matrix_input
    if not args.skip_query:
        # Use more reference spectra to exercise representative index paths.
        refs = base[:min(1_000, len(base))]
        for name, method in query_methods(methods):
            print(f"  {name} (index + search)", flush=True)
            scorer = method.create()
            index = scorer.build_index(refs)
            result = indexed_search(scorer, queries, index)
            validate_scores(result, (len(queries), len(refs)), full=True)
            del result, index, scorer
        del refs
    if cleaner is not None:
        print(f"  {MSENTROPY_METHOD}", flush=True)
        records, _, _ = convert_msentropy_records(small, cleaner, description="MS-Entropy warm-up")
        snapshots = [r["peaks"].copy() for r in records]
        # Repeated builds/searches must not reweight or mutate cached input records.
        first = msentropy_all_vs_all(records)
        second = msentropy_all_vs_all(records)
        validate_scores(first, (n, n), full=True)
        validate_scores(second, (n, n), full=True)
        np.testing.assert_allclose(first, second, rtol=2e-5, atol=2e-5)
        for record, snapshot in zip(records, snapshots, strict=True):
            np.testing.assert_array_equal(record["peaks"], snapshot)
        if not args.skip_query:
            core = build_msentropy_core(records)
            validate_scores(msentropy_query(core, queries, cleaner), (len(queries), n), full=True)
    gc.collect()


def run_all_vs_all(base, methods, records, args, table):
    for n in args.matrix_sizes:
        spectra = base[:n]
        print(f"\nALL-VS-ALL: {n:,} spectra", flush=True)
        spectrum_list = list(spectra) if args.matrix_input == "native" else None
        for method in methods:
            representation = method.input_representation(args)
            matrix_input = spectrum_list if representation == "list_spectrum" else spectra
            for jobs in method.jobs(args.parallel_jobs):
                print(f"  {method.name}, n_jobs={jobs}, input={representation}", flush=True)
                for repeat in range(args.all_vs_all_repeats):
                    # A fresh scorer avoids retaining caches from previous calls.
                    scorer = method.create()
                    row = matrix_row(method.name, method.family, n, jobs, repeat, True, representation)
                    result = table.record(row, lambda: matrix_call(method, scorer, matrix_input, jobs),
                                          lambda result: validate_scores(result, (n, n), full=args.validate_values))
                    del result, scorer
        if records is not None:
            print(f"  {MSENTROPY_METHOD}", flush=True)
            subset = records[:n]
            for repeat in range(args.all_vs_all_repeats):
                result = table.record(matrix_row(MSENTROPY_METHOD, "entropy", n, 1, repeat, False, "msentropy_precleaned"),
                                      lambda: msentropy_all_vs_all(subset),
                                      lambda result: validate_scores(result, (n, n), full=args.validate_values))
                del result
            del subset
        del spectra, spectrum_list, matrix_input
        gc.collect()


def matrix_row(method, family, n, jobs, repeat, preparation_included, representation):
    return dict(benchmark="all_vs_all", family=family, method=method, n_spectra=n,
                matrix_cells=n*n, n_jobs=jobs, execution="serial" if jobs == 1 else "parallel",
                repeat=repeat, score_fields="score", representation_preparation_included=preparation_included,
                input_representation=representation)


def run_index_and_search(name, n, factor, build, search, args, index_table, search_table, *, preparation_included):
    """Keep only the final index. Never overlap two repeated large index builds."""
    common = dict(method=name, n_reference=n, duplication_factor=factor, n_jobs=1, execution="serial")
    index = None
    for repeat in range(args.query_index_repeats):
        # Release the previous index *before* calling the next builder.
        del index
        index = index_table.record(dict(common, benchmark="query_index_build", repeat=repeat,
                                       library_preparation_included=preparation_included), build)
    for repeat in range(args.query_search_repeats):
        row = dict(common, benchmark="query_search", repeat=repeat, n_queries=N_QUERY_SPECTRA,
                   query_preparation_included=True, index_build_included=False)
        if index is None:
            search_table.failure(row, "Skipped: final index build failed")
            continue
        result = search_table.record(row, lambda: search(index),
                                     lambda result: validate_scores(result, (N_QUERY_SPECTRA, n),
                                                                    full=args.validate_values))
        del result
    del index
    gc.collect()


def run_query(base, queries, methods, records, cleaner, args, index_table, search_table):
    for n in args.reference_sizes:
        factor = duplication_factor(n, len(base))
        print(f"\nQUERY: {n:,} references (duplication factor={factor})", flush=True)
        references = make_reference_collection(base, n)
        for name, method in query_methods(methods):
            print(f"  {name}", flush=True)
            scorer = method.create()
            run_index_and_search(name, n, factor, lambda: scorer.build_index(references),
                                 lambda index: indexed_search(scorer, queries, index),
                                 args, index_table, search_table, preparation_included=True)
            del scorer
        del references
        gc.collect()
        if records is not None:
            print(f"  {MSENTROPY_METHOD}", flush=True)
            subset = make_msentropy_reference_records(records, n)
            run_index_and_search(MSENTROPY_METHOD, n, factor, lambda: build_msentropy_core(subset),
                                 lambda index: msentropy_query(index, queries, cleaner),
                                 args, index_table, search_table, preparation_included=False)
            del subset
            gc.collect()


def parse_sizes(text):
    try:
        sizes = tuple(map(int, text.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.") from exc
    if not sizes or sizes[0] <= 0 or tuple(sorted(set(sizes))) != sizes:
        raise argparse.ArgumentTypeError("Sizes must be positive, unique and strictly increasing.")
    return sizes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Cleaned MGF (default: matchms_scoring_benchmark_set.mgf).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/similarity"))
    parser.add_argument("--seed", type=int, default=56789)
    parser.add_argument("--matrix-sizes", type=parse_sizes, help="Default: 100,200,500,1000,2000,5000.")
    parser.add_argument("--reference-sizes", type=parse_sizes, help="Default: 1000 through 2000000 (see README).")
    parser.add_argument("--base-reference-size", type=int, help="Default: 500000. Larger sizes repeat this base.")
    parser.add_argument("--all-vs-all-repeats", type=int, help="Default: 3.")
    parser.add_argument("--query-index-repeats", type=int, help="Default: 3.")
    parser.add_argument("--query-search-repeats", type=int, help="Default: 5.")
    parser.add_argument("--matrix-input", choices=("native", "collection"), default="native",
                        help="Native: pre-materialized Spectrum lists for non-indexed methods; collection: original all-collection design.")
    parser.add_argument("--parallel-jobs", type=int, help="Default: -1; use 1 to omit parallel runs.")
    parser.add_argument("--skip-all-vs-all", action="store_true")
    parser.add_argument("--skip-query", action="store_true")
    parser.add_argument("--skip-msentropy", action="store_true")
    parser.add_argument("--hash-dataset", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--validate-values", action="store_true", help="Inspect all returned scores outside timers (may copy large matrices).")
    parser.add_argument("--overwrite", action="store_true", help="Replace known scoring benchmark files; does not resume.")
    parser.add_argument("--smoke", action="store_true", help="Small workload; generate synthetic MGF if --dataset is absent.")
    args = parser.parse_args(argv)
    defaults = dict(matrix_sizes=(8, 16) if args.smoke else ALL_VS_ALL_SIZES,
                    reference_sizes=(16, 32, 64, 128) if args.smoke else QUERY_REFERENCE_SIZES,
                    base_reference_size=64 if args.smoke else BASE_REFERENCE_SIZE,
                    all_vs_all_repeats=1 if args.smoke else 3, query_index_repeats=1 if args.smoke else 3,
                    query_search_repeats=1 if args.smoke else 5, parallel_jobs=2 if args.smoke else -1)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.smoke:
        args.fail_fast = args.validate_values = True
    if args.seed < 0 or min(args.all_vs_all_repeats, args.query_index_repeats, args.query_search_repeats) < 1:
        parser.error("Seed must be nonnegative and repeats positive.")
    if args.skip_all_vs_all and args.skip_query:
        parser.error("Both experiments are disabled.")
    if args.parallel_jobs == 0 or args.parallel_jobs < -1:
        parser.error("--parallel-jobs must be -1 or a positive integer.")
    if args.base_reference_size < N_QUERY_SPECTRA:
        parser.error(f"--base-reference-size must be at least {N_QUERY_SPECTRA}.")
    if not args.skip_all_vs_all and max(args.matrix_sizes) > args.base_reference_size:
        parser.error("Matrix sizes must not exceed the base reference size.")
    if not args.skip_query:
        for n in args.reference_sizes:
            try:
                duplication_factor(n, args.base_reference_size)
            except ValueError as exc:
                parser.error(str(exc))
    if args.dataset is None and not args.smoke:
        args.dataset = Path("matchms_scoring_benchmark_set.mgf")
    return args


def write_smoke_mgf(path, n=64):
    """Deterministic nonempty, separated peaks with common and differing masses."""
    with path.open("w", encoding="utf-8") as handle:
        for i in range(n):
            handle.write(f"BEGIN IONS\nTITLE=smoke_{i}\nPEPMASS={300 + i*0.1:.5f}\nCHARGE=1+\n")
            for j in range(8):
                mz = 50 + 15*j + (i % 3)*0.004
                handle.write(f"{mz:.6f} {1 + (i+j)%9:.6f}\n")
            handle.write("END IONS\n\n")


def prepare_output_dir(path, overwrite):
    path.mkdir(parents=True, exist_ok=True)
    names = ["benchmark_config.json", "sample_selection.npz", "representation_preparation.csv", "smoke_input.mgf"]
    names += [f"{stage}_{kind}.csv" for stage in SUMMARY_GROUPS for kind in ("timings", "summary")]
    existing = [path / name for name in names if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Scoring results already exist in {path}; use a fresh directory or --overwrite.")
    config = path / "benchmark_config.json"
    if config.exists():
        previous = json.loads(config.read_text(encoding="utf-8"))
        if previous.get("benchmark") != "matchms_1.0_scoring_paper":
            raise ValueError("Output directory belongs to a different benchmark; choose another directory.")
    for file in existing:
        file.unlink()


def environment_info():
    import matchms
    packages = {name: _package_version(name) for name in
                ("matchms", "numpy", "pandas", "scipy", "numba", "ms_entropy", "threadpoolctl", "tqdm", "psutil")}
    cpu_model = platform.processor()
    if sys.platform.startswith("linux"):
        try:
            cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                              if line.startswith("model name")), cpu_model)
        except OSError:
            pass
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), machine=platform.machine(),
                processor=cpu_model, logical_cpu_count=os.cpu_count(),
                process_cpu_count=getattr(os, "process_cpu_count", os.cpu_count)(), packages=packages,
                matchms_path=str(Path(matchms.__file__).resolve()),
                thread_environment={key: os.environ.get(key) for key in (*THREAD_VARIABLES, "NUMBA_NUM_THREADS")})
    try:
        import psutil
        info.update(physical_cpu_count=psutil.cpu_count(logical=False), ram_bytes=psutil.virtual_memory().total)
    except ImportError:
        pass
    try:
        from threadpoolctl import threadpool_info
        info["native_threadpools"] = threadpool_info()
    except ImportError:
        pass
    return info


def method_manifest(methods, args):
    matrix = []
    if not args.skip_all_vs_all:
        for method in methods:
            for jobs in method.jobs(args.parallel_jobs):
                matrix.append(dict(method=method.name, family=method.family, n_jobs=jobs,
                                   constructor=method.factory.__name__, kwargs=method.kwargs,
                                   representation_preparation_included=True, input_representation=method.input_representation(args)))
        if not args.skip_msentropy:
            matrix.append(dict(method=MSENTROPY_METHOD, family="entropy", n_jobs=1,
                               constructor="FlashEntropySearchCore", kwargs=dict(max_ms2_tolerance_in_da=MZ_TOLERANCE_DA,
                               intensity_weight="entropy"), representation_preparation_included=False,
                               input_representation="msentropy_precleaned"))
    indexed = []
    if not args.skip_query:
        for name, method in query_methods(methods):
            indexed.append(dict(method=name, n_jobs=1, constructor=method.factory.__name__, kwargs=method.kwargs,
                                library_preparation_included=True))
        if not args.skip_msentropy:
            indexed.append(dict(method=MSENTROPY_METHOD, n_jobs=1, constructor="FlashEntropySearchCore",
                                kwargs=dict(max_ms2_tolerance_in_da=MZ_TOLERANCE_DA, intensity_weight="entropy"),
                                library_preparation_included=False))
    return dict(all_vs_all=matrix, indexed=indexed)


def main(argv=None):
    args = parse_args(argv)
    if args.dataset is not None and not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    version = _package_version("matchms")
    if version is None or version.split(".")[0] != "1":
        raise RuntimeError(f"This script requires matchms 1.x (found {version!r}); use 1.0.0 for the paper.")
    # Fail on missing classes / invalid constructor arguments before loading a large MGF.
    methods = build_matrix_methods()
    for method in methods if not args.skip_all_vs_all else [m for _, m in query_methods(methods)]:
        method.create()
    cleaner = None if args.skip_msentropy else make_msentropy_cleaner()
    output = args.output_dir
    if args.dataset is not None and args.dataset.resolve() == (output / "smoke_input.mgf").resolve():
        raise ValueError("Use a different output directory for a previously generated smoke input.")
    prepare_output_dir(output, args.overwrite)
    synthetic = args.dataset is None
    if synthetic:
        args.dataset = output / "smoke_input.mgf"
        write_smoke_mgf(args.dataset, args.base_reference_size)
    print(f"Running {Path(__file__).resolve()} (SHA256 {_sha256(Path(__file__))[:12]})", flush=True)
    print(f"matchms={version}; ms_entropy={_package_version('ms_entropy')}; Python={platform.python_version()}", flush=True)
    config_path = output / "benchmark_config.json"
    config = dict(schema_version=2, benchmark="matchms_1.0_scoring_paper", status="running",
                  started_utc=_utc_now(), arguments=vars(args), synthetic_dataset=synthetic,
                  scientific_settings=dict(mz_tolerance_da=MZ_TOLERANCE_DA, mz_precision=MZ_PRECISION,
                     matchms_noise_cutoff=0.0, matchms_remove_precursor=False, entropy_dtype="float32",
                     score_fields=["score"], all_vs_all_sizes=list(args.matrix_sizes),
                     query_reference_sizes=list(args.reference_sizes), n_query_spectra=N_QUERY_SPECTRA,
                     base_reference_size=args.base_reference_size, msentropy_cleaning=MSENTROPY_SETTINGS),
                  methods=method_manifest(methods, args), environment=environment_info(),
                  benchmark_script=dict(path=str(Path(__file__).resolve()), sha256=_sha256(Path(__file__))))
    stat = args.dataset.stat()
    config["dataset"] = dict(path=str(args.dataset.resolve()), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                             sha256=_sha256(args.dataset) if args.hash_dataset or synthetic else None)
    _atomic_json(config_path, config)
    tables = {stage: TimingTable(output, stage, args.fail_fast) for stage in SUMMARY_GROUPS}
    try:
        base, queries, indices, positions, loaded, seconds = load_benchmark_dataset(args.dataset, args)
        config["dataset"].update(n_loaded_spectra=loaded, load_seconds=seconds, selected=dataset_statistics(base))
        np.savez(output / "sample_selection.npz", reference_indices=indices, query_positions=positions,
                 query_source_indices=indices[positions])
        _atomic_json(config_path, config)
        warm_up(base, queries, methods, args, cleaner)
        records = None
        if cleaner is not None:
            # Convert once, retain only the required prefix; avoid making a full base copy.
            required = max(args.matrix_sizes) if args.skip_query else min(len(base), max(args.reference_sizes))
            if not args.skip_all_vs_all:
                required = max(required, max(args.matrix_sizes))
            source = base if required == len(base) else base[:required]
            records, stats = prepare_msentropy_records(source, cleaner)
            del source
            _atomic_csv(output / "representation_preparation.csv", pd.DataFrame([stats]))
            config["scientific_settings"]["msentropy_preparation"] = stats
            _atomic_json(config_path, config)
        if not args.skip_all_vs_all:
            run_all_vs_all(base, methods, records, args, tables["all_vs_all"])
        if not args.skip_query:
            # All queried sizes <= base can use a shorter cached MS-Entropy prefix.
            run_query(base, queries, methods, records, cleaner, args, tables["query_index"], tables["query_search"])
        if any(row["status"] != "ok" for table in tables.values() for row in table.rows):
            raise RuntimeError("One or more timing repeats failed; inspect the raw CSVs.")
        config["status"] = "complete"
    except BaseException as exc:
        config.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print("Benchmark stopped. Completed raw rows are retained.", file=sys.stderr, flush=True)
        raise
    finally:
        config["finished_utc"] = _utc_now()
        # Do not mask the original benchmark exception with a summary failure.
        try:
            write_summary_tables(output)
        except Exception as exc:
            config["summary_error"] = f"{type(exc).__name__}: {exc}"
            if config["status"] == "complete":
                config["status"] = "failed"
                raise
            print(f"Could not write summaries: {exc}", file=sys.stderr)
        finally:
            _atomic_json(config_path, config)
    print(f"\nFinished. Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
