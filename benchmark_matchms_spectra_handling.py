#!/usr/bin/env python3
"""Reproducible benchmark for spectrum handling in matchms 1.0.

This benchmark compares the traditional ``list[Spectrum]`` representation with
``SpectraCollection`` for common spectrum-handling operations. The benchmark is
intended for a software-paper evaluation of the collection-based data model; it
therefore focuses on basic, local operations and deliberately avoids network
lookups, structure-toolkit work, or other scientifically expensive annotation
steps.

Experiments
-----------
The default benchmark uses nested samples containing 100, 1,000, 10,000, and
100,000 spectra. A deterministic reservoir sample is drawn from the complete
input file without keeping the entire source library in memory for streaming
formats (some importers return an eager list). Uniform sampling
avoids selecting just the beginning of a sorted source file, but the exact sample
still depends on its record order and content.

For every size, the same Spectrum objects are used to construct a
SpectraCollection. The following operation groups are benchmarked.

1. Read-only metadata and fragment summaries
   * precursor m/z values
   * precursor m/z distribution (histogram)
   * peak counts per spectrum
   * peak-count distribution
   * total ion current per spectrum

2. Row selection and metadata-based filtering
   * contiguous row slice
   * deterministic random row subset
   * construction of a precursor-m/z range mask
   * filtering rows to the middle precursor-m/z range
   * materializing a 100-500 m/z fragment slice

3. Public matchms filters
   * require precursor m/z
   * normalize intensities
   * select peaks by relative intensity
   * remove peaks relative to precursor m/z
   * reduce spectra to at most 500 peaks
   * a small peak-cleaning pipeline containing the four peak-processing steps

The peak-cleaning pipeline is intentionally limited to simple operations that
are representative of the matchms default/training pipelines. It excludes
structure conversion, annotation repair, PubChem access, and other chemistry-
heavy steps.

Timing methodology
------------------
* Dataset loading, deterministic sampling, and construction of the benchmark
  inputs are outside the operation timers.
* SpectraCollection construction is measured separately as a one-time setup
  cost.
* Every operation is warmed and validated on both representations before its
  timed repetitions.
* Small operations are automatically repeated inside one timing block until a
  target duration is reached. The reported value is wall-clock seconds per
  operation call.
* Garbage collection is run immediately before each timed block but remains
  enabled during execution.
* Representation order alternates between repeats to reduce systematic order,
  cache, and thermal effects.
* BLAS/OpenMP thread pools are restricted to one native thread before importing
  NumPy/SciPy/matchms.
* Raw timing rows are written incrementally so completed measurements survive a
  later failure. Summary CSVs are rebuilt from completed rows on exit.
* Results from different runs must be stored in separate output directories;
  --overwrite deliberately replaces existing benchmark files.
* Input Spectrum peaks are discretized at the chosen collection m/z precision.
  Exact equality requires peaks to survive binning without merging; correctness
  checks fail rather than quietly recording non-equivalent benchmark cases.

Interpretation
--------------
Not every operation has identical object-ownership semantics. In particular,
Python list slicing normally returns references to existing Spectrum objects,
whereas SpectraCollection row selection materializes a synchronized collection
containing metadata and fragment rows. These cases are still useful user-level
benchmarks but should not be interpreted as pure kernel comparisons. The output
CSV contains a ``semantics`` column documenting this distinction.

The script is usable from the command line and as an importable module.
"""

from __future__ import annotations

# Restrict implicit native thread pools before importing numerical libraries.
import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import re
import platform
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

DEFAULT_SIZES = (100, 1_000, 10_000, 100_000)
DEFAULT_MZ_PRECISION = 1e-6
DEFAULT_REPEATS = 5
DEFAULT_TARGET_REPEAT_SECONDS = 0.20
DEFAULT_MAX_INNER_LOOPS = 10_000
DEFAULT_RELATIVE_INTENSITY = 0.001
DEFAULT_TOP_K = 500
MZ_SLICE_MIN = 100.0
MZ_SLICE_MAX = 500.0

# Use a fixed number of histogram bins so list and collection execute the same
# downstream distribution calculation.
PRECURSOR_HISTOGRAM_BINS = 100
PEAK_COUNT_HISTOGRAM_EDGES = np.array(
    [0, 5, 10, 20, 50, 100, 200, 500, 1_000, np.inf],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Benchmark case definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkCase:
    """One list-versus-collection operation."""

    name: str
    category: str
    list_operation: Callable[[list, Any], Any]
    collection_operation: Callable[[Any, Any], Any]
    validator: Callable[[Any, Any], None]
    summarizer: Callable[[Any], dict[str, Any]]
    semantics: str
    expensive: bool = False


@dataclass(frozen=True)
class SizeContext:
    """Size-specific deterministic selectors and metadata thresholds."""

    n_spectra: int
    contiguous_slice: slice
    random_indices: np.ndarray
    precursor_low: float
    precursor_high: float


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _physical_cpu_count() -> int | None:
    """Record physical CPU cores if psutil is installed (optional dependency)."""
    try:
        import psutil
        return psutil.cpu_count(logical=False)
    except ImportError:
        return None


def _total_memory_bytes() -> int | None:
    """Record total RAM if psutil is installed (optional dependency)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def _git_revision(path: Path) -> str | None:
    """Return the current Git revision for *path*, if available."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
    temporary.replace(path)


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def _parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(part.strip().replace("_", "")) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--sizes requires comma-separated integers.") from exc
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Sizes must be positive integers.")
    if tuple(sorted(set(sizes))) != sizes:
        raise argparse.ArgumentTypeError(
            "Sizes must be unique and supplied in increasing order."
        )
    return sizes


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} is not empty. Use --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for filename in (
            "construction_timings.csv",
            "construction_summary.csv",
            "operation_timings.csv",
            "operation_summary.csv",
            "sample_selection.json",
            "sample_selection.npz",
            "benchmark_config.json",
        ):
            target = path / filename
            if target.exists():
                target.unlink()


def _summary_table(raw_path: Path, output_path: Path, group_columns: list[str]) -> pd.DataFrame:
    if not raw_path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(raw_path)
    if "status" in raw.columns:
        raw = raw[raw["status"] == "ok"].copy()

    if raw.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            raw.groupby(group_columns, dropna=False, sort=False)["seconds_per_call"]
            .agg(
                n_repeats="count",
                seconds_median="median",
                seconds_mean="mean",
                seconds_std="std",
                seconds_min="min",
                seconds_max="max",
            )
            .reset_index()
        )

    summary.to_csv(output_path, index=False)
    return summary


def _consume_result(result: Any) -> None:
    """Touch a result after timing so eager outputs are retained until completion."""
    if result is None:
        return
    if isinstance(result, np.ndarray):
        _ = result.shape
        return
    if isinstance(result, tuple):
        _ = len(result)
        return
    try:
        _ = len(result)
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# Dataset sampling and representation setup
# ---------------------------------------------------------------------------


def _reservoir_sample_spectra(
    dataset_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list, np.ndarray, int, float]:
    """Uniformly sample spectra and retain their source-row indices."""
    from matchms.importing import load_spectra

    rng = np.random.default_rng(seed)
    reservoir: list = []
    source_indices = np.empty(sample_size, dtype=np.int64)
    n_seen = 0

    start = perf_counter()
    spectra = load_spectra(str(dataset_path), metadata_harmonization=True)

    for n_seen, spectrum in enumerate(
        tqdm(spectra, desc="Sampling spectra", unit="spectra"),
        start=1,
    ):
        if n_seen <= sample_size:
            reservoir.append(spectrum)
            source_indices[n_seen - 1] = n_seen - 1
            continue

        replacement = int(rng.integers(0, n_seen))
        if replacement < sample_size:
            reservoir[replacement] = spectrum
            source_indices[replacement] = n_seen - 1

    elapsed = perf_counter() - start

    if n_seen < sample_size:
        raise ValueError(
            f"Dataset contains only {n_seen:,} spectra, but the largest requested "
            f"benchmark size is {sample_size:,}."
        )

    # Reservoir positions are not guaranteed to be in random order. Shuffle once
    # so prefixes form nested random benchmark sets.
    permutation = rng.permutation(sample_size)
    reservoir = [reservoir[int(index)] for index in permutation]
    source_indices = source_indices[permutation]

    return reservoir, source_indices, n_seen, elapsed


def _list_precursor_values(spectra: list) -> np.ndarray:
    values = np.empty(len(spectra), dtype=np.float64)
    for index, spectrum in enumerate(spectra):
        value = spectrum.get("precursor_mz", None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = np.nan
        values[index] = value if np.isfinite(value) and value > 0 else np.nan
    return values


def _collection_precursor_values(collection) -> np.ndarray:
    metadata = collection.metadata
    if "precursor_mz" not in metadata.columns:
        return np.full(len(collection), np.nan, dtype=np.float64)

    values = pd.to_numeric(
        metadata["precursor_mz"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    # Match the Spectrum implementation without modifying a potentially
    # read-only pandas view or the underlying metadata.
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def _make_size_context(
    spectra: list,
    precursor_low: float,
    precursor_high: float,
    seed: int,
) -> SizeContext:
    n_spectra = len(spectra)
    selection_size = min(10_000, max(1, n_spectra // 10))
    contiguous_start = max(0, (n_spectra - selection_size) // 2)

    rng = np.random.default_rng(seed + n_spectra)
    random_indices = np.sort(
        rng.choice(n_spectra, size=selection_size, replace=False)
    ).astype(np.int64)

    return SizeContext(
        n_spectra=n_spectra,
        contiguous_slice=slice(
            contiguous_start,
            contiguous_start + selection_size,
        ),
        random_indices=random_indices,
        precursor_low=precursor_low,
        precursor_high=precursor_high,
    )


def _precursor_range_from_sample(spectra: list) -> tuple[float, float, int]:
    values = _list_precursor_values(spectra)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        raise ValueError(
            "At least two spectra with finite positive precursor_mz values are "
            "required for the metadata-filter benchmark."
        )

    low, high = np.quantile(finite, [0.25, 0.75])
    if not low < high:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if not low < high:
        raise ValueError("Precursor m/z values do not span a usable range.")

    return float(low), float(high), int(finite.size)


# ---------------------------------------------------------------------------
# Read-only operations
# ---------------------------------------------------------------------------


def _list_precursor_histogram(spectra: list, context: SizeContext):
    values = _list_precursor_values(spectra)
    finite = values[np.isfinite(values)]
    return np.histogram(
        finite,
        bins=PRECURSOR_HISTOGRAM_BINS,
        range=(context.precursor_low, context.precursor_high),
    )


def _collection_precursor_histogram(collection, context: SizeContext):
    values = _collection_precursor_values(collection)
    finite = values[np.isfinite(values)]
    return np.histogram(
        finite,
        bins=PRECURSOR_HISTOGRAM_BINS,
        range=(context.precursor_low, context.precursor_high),
    )


def _list_peak_counts(spectra: list, _context: SizeContext) -> np.ndarray:
    return np.fromiter(
        (len(spectrum.peaks) for spectrum in spectra),
        dtype=np.int64,
        count=len(spectra),
    )


def _collection_peak_counts(collection, _context: SizeContext) -> np.ndarray:
    return np.asarray(collection.fragments.count(axis=1), dtype=np.int64)


def _list_peak_count_histogram(spectra: list, context: SizeContext):
    counts = _list_peak_counts(spectra, context)
    return np.histogram(counts, bins=PEAK_COUNT_HISTOGRAM_EDGES)


def _collection_peak_count_histogram(collection, context: SizeContext):
    counts = _collection_peak_counts(collection, context)
    return np.histogram(counts, bins=PEAK_COUNT_HISTOGRAM_EDGES)


def _list_total_ion_current(spectra: list, _context: SizeContext) -> np.ndarray:
    return np.fromiter(
        (np.sum(spectrum.peaks.intensities) for spectrum in spectra),
        dtype=np.float64,
        count=len(spectra),
    )


def _collection_total_ion_current(collection, _context: SizeContext) -> np.ndarray:
    return np.asarray(collection.fragments.sum(axis=1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Row selection and metadata filtering
# ---------------------------------------------------------------------------


def _list_contiguous_slice(spectra: list, context: SizeContext) -> list:
    return spectra[context.contiguous_slice]


def _collection_contiguous_slice(collection, context: SizeContext):
    return collection[context.contiguous_slice]


def _list_random_subset(spectra: list, context: SizeContext) -> list:
    return [spectra[int(index)] for index in context.random_indices]


def _collection_random_subset(collection, context: SizeContext):
    return collection[context.random_indices]


def _list_precursor_range_mask(spectra: list, context: SizeContext) -> np.ndarray:
    values = _list_precursor_values(spectra)
    return (
        np.isfinite(values)
        & (values >= context.precursor_low)
        & (values <= context.precursor_high)
    )


def _collection_precursor_range_mask(collection, context: SizeContext) -> np.ndarray:
    values = _collection_precursor_values(collection)
    return (
        np.isfinite(values)
        & (values >= context.precursor_low)
        & (values <= context.precursor_high)
    )


def _list_filter_precursor_range(spectra: list, context: SizeContext) -> list:
    mask = _list_precursor_range_mask(spectra, context)
    return [spectrum for spectrum, keep in zip(spectra, mask, strict=True) if keep]


def _collection_filter_precursor_range(collection, context: SizeContext):
    mask = _collection_precursor_range_mask(collection, context)
    return collection.filter(mask)


def _list_fragment_mz_slice(spectra: list, _context: SizeContext) -> list:
    """Materialize spectra containing only peaks in a fixed m/z window."""
    from matchms import Spectrum

    sliced = []
    for spectrum in spectra:
        # The peak properties each return a copy; obtain them only once per
        # spectrum rather than adding unrelated repeated-copy overhead.
        mz = spectrum.peaks.mz
        intensities = spectrum.peaks.intensities
        keep = (mz >= MZ_SLICE_MIN) & (mz <= MZ_SLICE_MAX)
        sliced.append(
            Spectrum(
                mz=mz[keep],
                intensities=intensities[keep],
                metadata=spectrum.metadata,
                metadata_harmonization=False,
            )
        )
    return sliced


def _collection_fragment_mz_slice(collection, _context: SizeContext):
    return collection[:, slice(MZ_SLICE_MIN, MZ_SLICE_MAX)]


# ---------------------------------------------------------------------------
# Public matchms filter operations
# ---------------------------------------------------------------------------


def _apply_filter_to_list(spectra: list, filter_function: Callable, **kwargs) -> list:
    output = []
    for spectrum in spectra:
        filtered = filter_function(spectrum, **kwargs)
        if filtered is not None:
            output.append(filtered)
    return output


def _list_require_precursor(spectra: list, _context: SizeContext) -> list:
    from matchms import filtering as msfilters

    return _apply_filter_to_list(spectra, msfilters.require_precursor_mz)


def _collection_require_precursor(collection, _context: SizeContext):
    from matchms import filtering as msfilters

    return msfilters.require_precursor_mz(collection)


def _list_normalize_intensities(spectra: list, _context: SizeContext) -> list:
    from matchms import filtering as msfilters

    return _apply_filter_to_list(spectra, msfilters.normalize_intensities)


def _collection_normalize_intensities(collection, _context: SizeContext):
    from matchms import filtering as msfilters

    return msfilters.normalize_intensities(collection)


def _list_relative_intensity_filter(spectra: list, _context: SizeContext) -> list:
    from matchms import filtering as msfilters

    return _apply_filter_to_list(
        spectra,
        msfilters.select_by_relative_intensity,
        intensity_from=DEFAULT_RELATIVE_INTENSITY,
    )


def _collection_relative_intensity_filter(collection, _context: SizeContext):
    from matchms import filtering as msfilters

    return msfilters.select_by_relative_intensity(
        collection,
        intensity_from=DEFAULT_RELATIVE_INTENSITY,
    )


def _list_remove_precursor_region(spectra: list, _context: SizeContext) -> list:
    from matchms import filtering as msfilters

    return _apply_filter_to_list(
        spectra,
        msfilters.remove_peaks_relative_to_precursor_mz,
    )


def _collection_remove_precursor_region(collection, _context: SizeContext):
    from matchms import filtering as msfilters

    return msfilters.remove_peaks_relative_to_precursor_mz(collection)


def _list_reduce_to_top_k(spectra: list, _context: SizeContext) -> list:
    from matchms import filtering as msfilters

    return _apply_filter_to_list(
        spectra,
        msfilters.reduce_to_number_of_peaks,
        n_max=DEFAULT_TOP_K,
    )


def _collection_reduce_to_top_k(collection, _context: SizeContext):
    from matchms import filtering as msfilters

    return msfilters.reduce_to_number_of_peaks(
        collection,
        n_max=DEFAULT_TOP_K,
    )


def _list_basic_peak_pipeline(spectra: list, _context: SizeContext) -> list:
    """Run one-copy peak cleaning using a simple subset of default filters."""
    from matchms import filtering as msfilters

    current = [spectrum.clone() for spectrum in spectra]
    steps = (
        (msfilters.normalize_intensities, {}),
        (
            msfilters.select_by_relative_intensity,
            {"intensity_from": DEFAULT_RELATIVE_INTENSITY},
        ),
        (msfilters.remove_peaks_relative_to_precursor_mz, {}),
        (msfilters.reduce_to_number_of_peaks, {"n_max": DEFAULT_TOP_K}),
    )

    for filter_function, kwargs in steps:
        filtered_spectra = []
        for spectrum in current:
            filtered = filter_function(
                spectrum,
                clone=False,
                **kwargs,
            )
            if filtered is not None:
                filtered_spectra.append(filtered)
        current = filtered_spectra

    return current


def _collection_basic_peak_pipeline(collection, _context: SizeContext):
    """Run the same peak-cleaning sequence on one copied collection."""
    from matchms import filtering as msfilters

    current = collection.copy()
    steps = (
        (msfilters.normalize_intensities, {}),
        (
            msfilters.select_by_relative_intensity,
            {"intensity_from": DEFAULT_RELATIVE_INTENSITY},
        ),
        (msfilters.remove_peaks_relative_to_precursor_mz, {}),
        (msfilters.reduce_to_number_of_peaks, {"n_max": DEFAULT_TOP_K}),
    )

    for filter_function, kwargs in steps:
        current = filter_function(
            current,
            clone=False,
            **kwargs,
        )
        if current is None:
            break

    return current


# ---------------------------------------------------------------------------
# Validation and result summaries
# ---------------------------------------------------------------------------


def _validate_array_equal(list_result, collection_result) -> None:
    np.testing.assert_array_equal(collection_result, list_result)


def _validate_array_close(list_result, collection_result) -> None:
    np.testing.assert_allclose(
        collection_result,
        list_result,
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )


def _validate_histogram(list_result, collection_result) -> None:
    list_counts, list_edges = list_result
    collection_counts, collection_edges = collection_result
    np.testing.assert_array_equal(collection_counts, list_counts)
    np.testing.assert_allclose(collection_edges, list_edges, rtol=0.0, atol=0.0)


def _sequence_peak_counts(spectra: Iterable) -> np.ndarray:
    return np.fromiter(
        (len(spectrum.peaks) for spectrum in spectra),
        dtype=np.int64,
    )


def _validate_sequence_collection(list_result, collection_result) -> None:
    if collection_result is None:
        if len(list_result) != 0:
            raise AssertionError("Collection result is None but list result is nonempty.")
        return

    if len(list_result) != len(collection_result):
        raise AssertionError(
            f"Different number of spectra: list={len(list_result)}, "
            f"collection={len(collection_result)}"
        )

    expected_counts = _sequence_peak_counts(list_result)
    actual_counts = np.asarray(collection_result.fragments.count(axis=1), dtype=np.int64)
    np.testing.assert_array_equal(actual_counts, expected_counts)

    expected_precursor = _list_precursor_values(list_result)
    actual_precursor = _collection_precursor_values(collection_result)
    np.testing.assert_allclose(
        actual_precursor,
        expected_precursor,
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    )

    if not list_result:
        return

    # Check total ion current for every output row as well as peak counts.
    # This catches intensity differences outside the spot-checked fragment rows.
    expected_tic = _list_total_ion_current(list_result, None)
    actual_tic = _collection_total_ion_current(collection_result, None)
    np.testing.assert_allclose(
        actual_tic, expected_tic, rtol=1e-12, atol=1e-12, equal_nan=True
    )

    # Spot-check the actual m/z/intensity vectors at deterministic positions;
    # validate without converting all collection rows back to Spectrum objects.
    positions = sorted(set(np.linspace(0, len(list_result) - 1, min(21, len(list_result)), dtype=int)))
    for position in positions:
        expected = list_result[position]
        actual_mz, actual_intensity = collection_result.fragments.get_row(position)
        np.testing.assert_allclose(
            actual_mz,
            expected.peaks.mz,
            rtol=0.0,
            atol=max(getattr(collection_result, "mz_precision", 0.0), 1e-12),
        )
        np.testing.assert_allclose(
            actual_intensity,
            expected.peaks.intensities,
            rtol=1e-12,
            atol=1e-12,
        )


def _summary_array(result: np.ndarray) -> dict[str, Any]:
    finite = result[np.isfinite(result)] if np.issubdtype(result.dtype, np.floating) else result
    return {
        "output_size": int(result.size),
        "output_n_spectra": np.nan,
        "output_n_peaks": np.nan,
        "output_checksum": float(np.sum(finite, dtype=np.float64)) if finite.size else 0.0,
    }


def _summary_histogram(result: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    counts, _edges = result
    return {
        "output_size": int(counts.size),
        "output_n_spectra": np.nan,
        "output_n_peaks": np.nan,
        "output_checksum": float(np.sum(counts, dtype=np.float64)),
    }


def _summary_list(result: list) -> dict[str, Any]:
    return {
        "output_size": len(result),
        "output_n_spectra": len(result),
        "output_n_peaks": int(sum(len(spectrum.peaks) for spectrum in result)),
        "output_checksum": np.nan,
    }


def _summary_collection(result) -> dict[str, Any]:
    if result is None:
        return {
            "output_size": 0,
            "output_n_spectra": 0,
            "output_n_peaks": 0,
            "output_checksum": np.nan,
        }

    return {
        "output_size": len(result),
        "output_n_spectra": len(result),
        "output_n_peaks": int(np.sum(result.fragments.count(axis=1), dtype=np.int64)),
        "output_checksum": np.nan,
    }


def _summary_sequence_result(result) -> dict[str, Any]:
    if isinstance(result, list):
        return _summary_list(result)
    return _summary_collection(result)


# ---------------------------------------------------------------------------
# Benchmark case registry
# ---------------------------------------------------------------------------


def build_benchmark_cases() -> list[BenchmarkCase]:
    """Return the paper benchmark's common list/collection operations."""
    return [
        BenchmarkCase(
            name="precursor_values",
            category="metadata_query",
            list_operation=lambda spectra, _context: _list_precursor_values(spectra),
            collection_operation=lambda collection, _context: _collection_precursor_values(collection),
            validator=_validate_array_close,
            summarizer=_summary_array,
            semantics="Equivalent numeric precursor_mz array.",
        ),
        BenchmarkCase(
            name="precursor_histogram",
            category="metadata_query",
            list_operation=_list_precursor_histogram,
            collection_operation=_collection_precursor_histogram,
            validator=_validate_histogram,
            summarizer=_summary_histogram,
            semantics="Equivalent precursor_mz distribution using fixed histogram settings.",
        ),
        BenchmarkCase(
            name="peak_counts",
            category="fragment_query",
            list_operation=_list_peak_counts,
            collection_operation=_collection_peak_counts,
            validator=_validate_array_equal,
            summarizer=_summary_array,
            semantics=(
                "Equivalent peak counts when collection binning preserves each peak."
            ),
        ),
        BenchmarkCase(
            name="peak_count_histogram",
            category="fragment_query",
            list_operation=_list_peak_count_histogram,
            collection_operation=_collection_peak_count_histogram,
            validator=_validate_histogram,
            summarizer=_summary_histogram,
            semantics="Equivalent peak-count distribution using fixed bins.",
        ),
        BenchmarkCase(
            name="total_ion_current",
            category="fragment_query",
            list_operation=_list_total_ion_current,
            collection_operation=_collection_total_ion_current,
            validator=_validate_array_close,
            summarizer=_summary_array,
            semantics="Equivalent sum of fragment intensities per spectrum.",
        ),
        BenchmarkCase(
            name="contiguous_slice_10pct",
            category="row_selection",
            list_operation=_list_contiguous_slice,
            collection_operation=_collection_contiguous_slice,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                "Same selected spectra, but list slicing returns references while "
                "SpectraCollection materializes synchronized metadata/fragments."
            ),
        ),
        BenchmarkCase(
            name="random_subset_10pct",
            category="row_selection",
            list_operation=_list_random_subset,
            collection_operation=_collection_random_subset,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                "Same selected rows; list output contains existing Spectrum references, "
                "whereas SpectraCollection returns a new collection."
            ),
        ),
        BenchmarkCase(
            name="precursor_range_mask",
            category="metadata_filter",
            list_operation=_list_precursor_range_mask,
            collection_operation=_collection_precursor_range_mask,
            validator=_validate_array_equal,
            summarizer=_summary_array,
            semantics="Equivalent boolean mask using fixed precursor_mz quartile bounds.",
        ),
        BenchmarkCase(
            name="filter_precursor_range",
            category="metadata_filter",
            list_operation=_list_filter_precursor_range,
            collection_operation=_collection_filter_precursor_range,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                "Equivalent retained rows; list filtering keeps Spectrum references while "
                "SpectraCollection materializes synchronized rows."
            ),
        ),
        BenchmarkCase(
            name="fragment_mz_slice_100_500",
            category="fragment_slice",
            list_operation=_list_fragment_mz_slice,
            collection_operation=_collection_fragment_mz_slice,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                "Equivalent materialized spectra restricted to 100-500 m/z; list mode "
                "constructs Spectrum objects row by row, while SpectraCollection slices "
                "the packed fragment matrix and keeps metadata synchronized."
            ),
            expensive=True,
        ),
        BenchmarkCase(
            name="require_precursor_mz",
            category="public_metadata_filter",
            list_operation=_list_require_precursor,
            collection_operation=_collection_require_precursor,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics="Public matchms require_precursor_mz filter on both representations.",
            expensive=True,
        ),
        BenchmarkCase(
            name="normalize_intensities",
            category="public_fragment_filter",
            list_operation=_list_normalize_intensities,
            collection_operation=_collection_normalize_intensities,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics="Public matchms normalize_intensities filter with default clone behavior.",
            expensive=True,
        ),
        BenchmarkCase(
            name="select_relative_intensity",
            category="public_fragment_filter",
            list_operation=_list_relative_intensity_filter,
            collection_operation=_collection_relative_intensity_filter,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                f"Public select_by_relative_intensity with intensity_from="
                f"{DEFAULT_RELATIVE_INTENSITY}."
            ),
            expensive=True,
        ),
        BenchmarkCase(
            name="remove_precursor_region",
            category="public_fragment_filter",
            list_operation=_list_remove_precursor_region,
            collection_operation=_collection_remove_precursor_region,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics="Public remove_peaks_relative_to_precursor_mz filter.",
            expensive=True,
        ),
        BenchmarkCase(
            name="reduce_to_top_500",
            category="public_fragment_filter",
            list_operation=_list_reduce_to_top_k,
            collection_operation=_collection_reduce_to_top_k,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=f"Public reduce_to_number_of_peaks with n_max={DEFAULT_TOP_K}.",
            expensive=True,
        ),
        BenchmarkCase(
            name="basic_peak_cleaning_pipeline",
            category="public_fragment_pipeline",
            list_operation=_list_basic_peak_pipeline,
            collection_operation=_collection_basic_peak_pipeline,
            validator=_validate_sequence_collection,
            summarizer=_summary_sequence_result,
            semantics=(
                "One explicit copy followed by normalize_intensities, relative-intensity "
                "filtering, precursor-region removal, and top-500 reduction with clone=False."
            ),
            expensive=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _calibrate_inner_loops(
    operation: Callable[[], Any],
    target_seconds: float,
    max_inner_loops: int,
    expensive: bool,
) -> int:
    """Choose repeat count for one timing block from a single calibration call."""
    gc.collect()
    start = perf_counter()
    result = operation()
    elapsed = perf_counter() - start
    _consume_result(result)
    del result

    if elapsed <= 0:
        return 1

    loops = max(1, int(math.ceil(target_seconds / elapsed)))
    cap = min(max_inner_loops, 10 if expensive else max_inner_loops)
    return min(loops, cap)


def _time_operation(
    operation: Callable[[], Any],
    inner_loops: int,
) -> tuple[float, Any]:
    gc.collect()
    start = perf_counter()
    result = None
    for _ in range(inner_loops):
        result = operation()
    elapsed = perf_counter() - start
    # Results are eager; retaining the last one prevents its immediate removal.
    # Do not include result-inspection overhead inside the timing block because
    # ndarray/list/collection outputs would pay representation-dependent costs.
    _consume_result(result)
    return elapsed / inner_loops, result


def _run_case(
    case: BenchmarkCase,
    spectra: list,
    collection,
    context: SizeContext,
    repeats: int,
    target_repeat_seconds: float,
    max_inner_loops: int,
    raw_path: Path,
    fail_fast: bool,
) -> None:
    """Validate, calibrate, and time one operation for both representations."""
    list_callable = lambda: case.list_operation(spectra, context)
    collection_callable = lambda: case.collection_operation(collection, context)

    # Warm and validate before timing. Capture fallback warnings so the user can
    # see them during the run, while timing starts only after first-call setup.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        list_result = list_callable()
        collection_result = collection_callable()

    for warning in caught:
        print(
            f"WARNING during warmup [{case.name}]: {warning.message}",
            file=sys.stderr,
        )

    case.validator(list_result, collection_result)

    list_summary = case.summarizer(list_result)
    collection_summary = case.summarizer(collection_result)
    del list_result, collection_result
    gc.collect()

    list_loops = _calibrate_inner_loops(
        list_callable,
        target_repeat_seconds,
        max_inner_loops,
        case.expensive,
    )
    collection_loops = _calibrate_inner_loops(
        collection_callable,
        target_repeat_seconds,
        max_inner_loops,
        case.expensive,
    )

    summaries = {
        "list_spectrum": list_summary,
        "spectra_collection": collection_summary,
    }
    callables = {
        "list_spectrum": (list_callable, list_loops),
        "spectra_collection": (collection_callable, collection_loops),
    }

    for repeat in range(repeats):
        # Alternate execution order across repetitions.
        order = (
            ("list_spectrum", "spectra_collection")
            if repeat % 2 == 0
            else ("spectra_collection", "list_spectrum")
        )

        for representation in order:
            operation, inner_loops = callables[representation]
            base = {
                "benchmark": "spectrum_handling",
                "category": case.category,
                "operation": case.name,
                "representation": representation,
                "n_spectra": len(spectra),
                "repeat": repeat,
                "inner_loops": inner_loops,
                "semantics": case.semantics,
            }
            base.update(summaries[representation])

            try:
                seconds_per_call, result = _time_operation(
                    operation,
                    inner_loops,
                )
                row = {
                    **base,
                    "status": "ok",
                    "seconds_per_call": seconds_per_call,
                    "error": "",
                }
                del result
            except Exception as exc:  # noqa: BLE001 - benchmark should record failures.
                row = {
                    **base,
                    "status": "error",
                    "seconds_per_call": np.nan,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                _append_csv_row(raw_path, row)
                if fail_fast:
                    raise
                continue

            _append_csv_row(raw_path, row)


# ---------------------------------------------------------------------------
# Construction benchmark
# ---------------------------------------------------------------------------


def _benchmark_collection_construction(
    spectra: list,
    mz_precision: float,
    repeats: int,
    target_repeat_seconds: float,
    max_inner_loops: int,
    raw_path: Path,
) -> None:
    from matchms import SpectraCollection

    operation = lambda: SpectraCollection(spectra, mz_precision=mz_precision)

    # Warm and ensure the output has the expected size.
    warm = operation()
    if len(warm) != len(spectra):
        raise AssertionError("SpectraCollection construction changed row count.")
    del warm

    inner_loops = _calibrate_inner_loops(
        operation,
        target_repeat_seconds,
        max_inner_loops=min(max_inner_loops, 10),
        expensive=True,
    )

    for repeat in range(repeats):
        seconds_per_call, result = _time_operation(operation, inner_loops)
        summary = _summary_collection(result)
        del result

        _append_csv_row(
            raw_path,
            {
                "benchmark": "spectra_collection_construction",
                "category": "construction",
                "operation": "list_to_spectra_collection",
                "representation": "spectra_collection",
                "n_spectra": len(spectra),
                "repeat": repeat,
                "inner_loops": inner_loops,
                "semantics": (
                    "One-time packing of existing Spectrum objects into synchronized "
                    "metadata and sparse fragment storage."
                ),
                **summary,
                "status": "ok",
                "seconds_per_call": seconds_per_call,
                "error": "",
            },
        )


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------


def run_benchmark(
    dataset_path: Path,
    output_dir: Path,
    sizes: tuple[int, ...],
    seed: int,
    repeats: int,
    target_repeat_seconds: float,
    max_inner_loops: int,
    mz_precision: float,
    include_expensive_filters: bool,
    fail_fast: bool,
    hash_dataset: bool,
) -> None:
    from matchms import SpectraCollection
    import matchms

    version = _package_version("matchms")
    if not re.match(r"^1\.", version):
        raise RuntimeError(
            f"This benchmark is for matchms 1.x, but the installed matchms is {version!r}. "
            "For the paper's matchms 1.0 baseline, install matchms==1.0.0."
        )
    max_size = max(sizes)

    cases = build_benchmark_cases()
    if not include_expensive_filters:
        cases = [case for case in cases if not case.expensive]

    sampled_spectra, source_indices, n_source_spectra, sampling_seconds = _reservoir_sample_spectra(
        dataset_path,
        max_size,
        seed,
    )

    precursor_low, precursor_high, n_finite_precursors = _precursor_range_from_sample(
        sampled_spectra
    )

    script_path = Path(__file__).resolve()
    config = {
        "run": {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": sys.argv,
            "benchmark_script": str(script_path),
            "benchmark_script_sha256": _sha256(script_path),
        },
        "dataset": {
            "path": str(dataset_path.resolve()),
            "size_bytes": dataset_path.stat().st_size,
            "mtime_ns": dataset_path.stat().st_mtime_ns,
            "sha256": _sha256(dataset_path) if hash_dataset else None,
            "n_source_spectra": n_source_spectra,
            "sampling_seconds": sampling_seconds,
            "sampling_method": "deterministic reservoir sample followed by permutation",
            "sample_size": max_size,
            "n_finite_precursors_in_sample": n_finite_precursors,
        },
        "benchmark": {
            "sizes": list(sizes),
            "operations": [case.name for case in cases],
            "operation_categories": {case.name: case.category for case in cases},
            "seed": seed,
            "repeats": repeats,
            "target_repeat_seconds": target_repeat_seconds,
            "max_inner_loops": max_inner_loops,
            "mz_precision": mz_precision,
            "relative_intensity_threshold": DEFAULT_RELATIVE_INTENSITY,
            "top_k": DEFAULT_TOP_K,
            "fragment_mz_slice": [MZ_SLICE_MIN, MZ_SLICE_MAX],
            "precursor_filter_low": precursor_low,
            "precursor_filter_high": precursor_high,
            "include_expensive_filters": include_expensive_filters,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "matchms_version": _package_version("matchms"),
            "matchms_path": str(Path(matchms.__file__).resolve()),
            "matchms_git_revision": _git_revision(Path(matchms.__file__).resolve().parent),
            "numpy_version": _package_version("numpy"),
            "pandas_version": _package_version("pandas"),
            "scipy_version": _package_version("scipy"),
            "matplotlib_version": _package_version("matplotlib"),
            "tqdm_version": _package_version("tqdm"),
            "physical_cpu_count": _physical_cpu_count(),
            "total_memory_bytes": _total_memory_bytes(),
        },
        "timing_notes": {
            "loading_and_sampling": "excluded",
            "input_construction": "excluded from operation timings and measured separately",
            "warmup_and_validation": "excluded",
            "garbage_collection": "gc.collect immediately before each timed block; GC remains enabled",
            "reported_unit": "wall-clock seconds per operation call",
            "timed_filter_clone": (
                "default (True) for individual filters; "
                "explicit one-copy and clone=False for pipelines"
            ),
            "sample_order_dependent": True,
            "collection_mz_binning_may_merge_peaks": True,
            "row_selection_semantics": (
                "Python lists retain references; SpectraCollection selection materializes "
                "synchronized metadata and fragments."
            ),
        },
    }
    _write_json(output_dir / "benchmark_config.json", config)

    np.savez_compressed(
        output_dir / "sample_selection.npz",
        source_indices=source_indices,
        nested_sizes=np.asarray(sizes, dtype=np.int64),
    )
    _write_json(
        output_dir / "sample_selection.json",
        {
            "seed": seed,
            "method": "reservoir sample + permutation",
            "n_source_spectra": n_source_spectra,
            "sample_size": max_size,
            "nested_sizes": list(sizes),
            "precursor_filter_low": precursor_low,
            "precursor_filter_high": precursor_high,
            "source_indices_file": "sample_selection.npz",
        },
    )

    construction_path = output_dir / "construction_timings.csv"
    operation_path = output_dir / "operation_timings.csv"

    try:
        _run_sizes(
            sizes=sizes,
            sampled_spectra=sampled_spectra,
            cases=cases,
            precursor_low=precursor_low,
            precursor_high=precursor_high,
            seed=seed,
            mz_precision=mz_precision,
            repeats=repeats,
            target_repeat_seconds=target_repeat_seconds,
            max_inner_loops=max_inner_loops,
            fail_fast=fail_fast,
            construction_path=construction_path,
            operation_path=operation_path,
        )
    finally:
        # These summaries reflect the raw rows even when a run is interrupted.
        for raw_path, filename in (
            (construction_path, "construction_summary.csv"),
            (operation_path, "operation_summary.csv"),
        ):
            _summary_table(raw_path, output_dir / filename, [
                "benchmark", "category", "operation", "representation", "n_spectra", "semantics",
            ])

    config["run"]["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(output_dir / "benchmark_config.json", config)
    print("\nBenchmark complete.")
    print(f"Results written to: {output_dir.resolve()}")


def _run_sizes(
    *, sizes, sampled_spectra, cases, precursor_low, precursor_high,
    seed, mz_precision, repeats,
    target_repeat_seconds, max_inner_loops, fail_fast,
    construction_path, operation_path,
) -> None:
    """Measure each requested size; all input representations are constructed outside case timers."""
    from matchms import SpectraCollection

    for size in sizes:
        print(f"\n=== {size:,} spectra ===")
        spectra = sampled_spectra[:size]
        context = _make_size_context(
            spectra,
            precursor_low,
            precursor_high,
            seed,
        )

        print("Benchmarking SpectraCollection construction...")
        _benchmark_collection_construction(
            spectra,
            mz_precision,
            repeats,
            target_repeat_seconds,
            max_inner_loops,
            construction_path,
        )

        # Build the persistent input representation once outside operation timers.
        collection = SpectraCollection(spectra, mz_precision=mz_precision)

        # Basic input validation before running any benchmark cases.
        np.testing.assert_array_equal(
            _collection_peak_counts(collection, context),
            _list_peak_counts(spectra, context),
        )
        np.testing.assert_allclose(
            _collection_precursor_values(collection),
            _list_precursor_values(spectra),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        )

        for case in cases:
            print(f"  {case.category:24s} {case.name}")
            _run_case(
                case,
                spectra,
                collection,
                context,
                repeats,
                target_repeat_seconds,
                max_inner_loops,
                operation_path,
                fail_fast,
            )

        del collection
        gc.collect()

        # Flush summaries after every size so partial runs are immediately useful.
        _summary_table(
            construction_path,
            construction_path.parent / "construction_summary.csv",
            [
                "benchmark",
                "category",
                "operation",
                "representation",
                "n_spectra",
                "semantics",
            ],
        )
        _summary_table(
            operation_path,
            operation_path.parent / "operation_summary.csv",
            [
                "benchmark",
                "category",
                "operation",
                "representation",
                "n_spectra",
                "semantics",
            ],
        )



# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark list[Spectrum] versus SpectraCollection for common matchms "
            "spectrum-handling operations."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("matchms_scoring_benchmark_set.mgf"),
        help="Input spectrum file (default: matchms_scoring_benchmark_set.mgf).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_spectra_handling"),
        help="Directory for benchmark CSV/JSON output.",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_SIZES,
        help="Comma-separated nested sample sizes (default: 100,1000,10000,100000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56789,
        help="Random seed for reservoir sampling and row subsets.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of timed repetitions per operation and representation.",
    )
    parser.add_argument(
        "--target-repeat-seconds",
        type=float,
        default=DEFAULT_TARGET_REPEAT_SECONDS,
        help=(
            "Target duration used to choose inner loops for fast operations "
            f"(default: {DEFAULT_TARGET_REPEAT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--max-inner-loops",
        type=int,
        default=DEFAULT_MAX_INNER_LOOPS,
        help="Maximum operations executed inside one timing block.",
    )
    parser.add_argument(
        "--mz-precision",
        type=float,
        default=DEFAULT_MZ_PRECISION,
        help="SpectraCollection m/z precision used for fragment packing.",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help=(
            "Skip materializing fragment slices and public peak/metadata filters; "
            "run only cheap queries, row selection, and direct metadata filtering."
        ),
    )
    parser.add_argument(
        "--hash-dataset",
        action="store_true",
        help="Calculate and record the SHA256 of the input dataset.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed benchmark operation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow reuse of a non-empty output directory and replace benchmark files.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(f"Dataset does not exist: {args.dataset}")
    if args.repeats <= 0:
        parser.error("--repeats must be positive.")
    if args.target_repeat_seconds <= 0:
        parser.error("--target-repeat-seconds must be positive.")
    if args.max_inner_loops <= 0:
        parser.error("--max-inner-loops must be positive.")
    if not math.isfinite(args.mz_precision) or args.mz_precision <= 0:
        parser.error("--mz-precision must be finite and positive.")
    if not math.isfinite(args.target_repeat_seconds):
        parser.error("--target-repeat-seconds must be finite.")
    if not args.dataset.is_file():
        parser.error(f"Dataset must be a regular file: {args.dataset}")

    _prepare_output_directory(args.output_dir, args.overwrite)

    try:
        run_benchmark(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            sizes=args.sizes,
            seed=args.seed,
            repeats=args.repeats,
            target_repeat_seconds=args.target_repeat_seconds,
            max_inner_loops=args.max_inner_loops,
            mz_precision=args.mz_precision,
            include_expensive_filters=not args.basic_only,
            fail_fast=args.fail_fast,
            hash_dataset=args.hash_dataset,
        )
    except Exception:
        print(
            "Benchmark stopped. Completed raw timing rows and summaries are retained.",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    main()
