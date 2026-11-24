"""Optional integration smoke tests.
"""
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_matchms_similarity_computation.py"


@pytest.mark.parametrize("external_baseline", [False, True])
def test_real_scoring_smoke(tmp_path, external_baseline):
    pytest.importorskip("matchms")
    if external_baseline:
        pytest.importorskip("ms_entropy")
    flags = [] if external_baseline else ["--skip-msentropy"]
    subprocess.run([sys.executable, str(SCRIPT), "--smoke", "--output-dir", str(tmp_path), *flags],
                   check=True, timeout=600)
