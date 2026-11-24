"""Fast checks that do not require importing matchms."""

from pathlib import Path
import sys
import argparse

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_matchms_spectra_handling as benchmark


class PeaksStub:
    def __init__(self, mz, intensities):
        self.mz = np.asarray(mz, dtype=float)
        self.intensities = np.asarray(intensities, dtype=float)

    def __len__(self):
        return len(self.mz)


class SpectrumStub:
    def __init__(self, precursor_mz, mz=None, intensities=None):
        self._precursor_mz = precursor_mz
        self.peaks = PeaksStub(
            [100.0, 110.0] if mz is None else mz,
            [1.0, 2.0] if intensities is None else intensities,
        )

    def get(self, key, default=None):
        assert key == "precursor_mz"
        return self._precursor_mz


class FragmentStub:
    def __init__(self, spectra):
        self.spectra = spectra

    def count(self, axis=1):
        return np.asarray([len(s.peaks.mz) for s in self.spectra])

    def sum(self, axis=1):
        return np.asarray([s.peaks.intensities.sum() for s in self.spectra])

    def get_row(self, index):
        peaks = self.spectra[index].peaks
        return peaks.mz.copy(), peaks.intensities.copy()


class CollectionStub:
    mz_precision = 1e-6

    def __init__(self, spectra):
        self.spectra = spectra
        self.fragments = FragmentStub(spectra)
        self.metadata = pd.DataFrame({"precursor_mz": [s.get("precursor_mz") for s in spectra]})

    def __len__(self):
        return len(self.spectra)


def test_parse_sizes():
    assert benchmark._parse_sizes("100, 1_000,10000") == (100, 1000, 10000)
    for value in ("100,100", "1000,100", "0,100", "", "abc,100"):
        with pytest.raises(argparse.ArgumentTypeError):
            benchmark._parse_sizes(value)


def test_precursor_value_implementations_agree_on_missing_and_invalid():
    spectra = [SpectrumStub(x) for x in [25, None, -5, 0, "45.5", float("inf"), "abc"]]
    listed = benchmark._list_precursor_values(spectra)
    collected = benchmark._collection_precursor_values(CollectionStub(spectra))
    np.testing.assert_allclose(listed, collected, equal_nan=True)
    assert np.isnan(listed[[1, 2, 3, 5, 6]]).all()
    np.testing.assert_array_equal(listed[[0, 4]], [25.0, 45.5])


def test_context_is_deterministic_and_nested_selection():
    spectra = [SpectrumStub(i + 100) for i in range(50)]
    first = benchmark._make_size_context(spectra, 120.0, 130.0, 56789)
    second = benchmark._make_size_context(spectra, 120.0, 130.0, 56789)
    np.testing.assert_array_equal(first.random_indices, second.random_indices)
    assert np.unique(first.random_indices).size == first.random_indices.size
    assert first.random_indices.min() >= 0
    assert first.random_indices.max() < len(spectra)


def test_sequence_validation_detects_intensity_error_in_unchecked_row():
    spectra = [SpectrumStub(i + 100) for i in range(50)]
    collection = CollectionStub(spectra)
    benchmark._validate_sequence_collection(spectra, collection)
    wrong = [SpectrumStub(i + 100) for i in range(50)]
    wrong[13].peaks.intensities[0] += 0.5
    with pytest.raises(AssertionError):
        benchmark._validate_sequence_collection(spectra, CollectionStub(wrong))


def test_summary_skips_failed_timings(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "summary.csv"
    pd.DataFrame([
        {"operation": "x", "representation": "list", "seconds_per_call": 1.0, "status": "ok"},
        {"operation": "x", "representation": "list", "seconds_per_call": 3.0, "status": "ok"},
        {"operation": "x", "representation": "list", "seconds_per_call": np.nan, "status": "error"},
    ]).to_csv(raw, index=False)
    summary = benchmark._summary_table(raw, out, ["operation", "representation"])
    assert out.is_file()
    assert summary.iloc[0]["n_repeats"] == 2
    assert summary.iloc[0]["seconds_median"] == 2.0
