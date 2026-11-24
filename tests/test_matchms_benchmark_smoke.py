"""Run every operation's correctness validator against an installed matchms 1.x.

This is deliberately small; it validates API compatibility, not performance.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

matchms = pytest.importorskip("matchms")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchmark_matchms_spectra_handling as benchmark


def test_matchms_1_smoke_all_cases():
    from matchms import Spectrum, SpectraCollection

    spectra = []
    for i in range(8):
        mz = np.arange(100.0, 651.0, 1.0)
        intensities = np.linspace(0.1, 1.0, mz.size)
        intensities[0] = 1e-5  # Exercise relative-intensity filtering.
        spectra.append(Spectrum(
            mz=mz, intensities=intensities,
            metadata={"precursor_mz": float(480 + 15 * i), "id": f"benchmark-test-{i}"},
        ))
    collection = SpectraCollection(spectra, mz_precision=1e-6)
    context = benchmark._make_size_context(spectra, 495.0, 555.0, 56789)

    for case in benchmark.build_benchmark_cases():
        list_result = case.list_operation(spectra, context)
        collection_result = case.collection_operation(collection, context)
        case.validator(list_result, collection_result)
        assert case.summarizer(list_result)["output_size"] == case.summarizer(collection_result)["output_size"]
