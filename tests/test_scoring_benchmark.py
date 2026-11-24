"""Regression tests for benchmark orchestration (no mass-spectrometry packages)."""
import importlib.util
import json
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_matchms_similarity_computation.py"
spec = importlib.util.spec_from_file_location("scoring_benchmark_under_test", SCRIPT)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


def test_default_and_smoke_settings():
    args = b.parse_args([])
    assert args.matrix_sizes == (100, 200, 500, 1000, 2000, 5000)
    assert args.reference_sizes[-2:] == (1000000, 2000000)
    assert args.base_reference_size == 500000
    assert (args.all_vs_all_repeats, args.query_index_repeats, args.query_search_repeats) == (3, 3, 5)
    assert args.parallel_jobs == -1
    smoke = b.parse_args(["--smoke"])
    assert smoke.base_reference_size == 64 and smoke.reference_sizes[-1] == 128
    assert smoke.dataset is None and smoke.fail_fast and smoke.validate_values
    assert smoke.parallel_jobs == 2


@pytest.mark.parametrize("argv", [
    ["--matrix-sizes", "10,10"], ["--reference-sizes", "500000,750000"],
    ["--skip-query", "--skip-all-vs-all"], ["--parallel-jobs", "0"],
    ["--all-vs-all-repeats", "0"], ["--base-reference-size", "5"],
])
def test_bad_options(argv):
    with pytest.raises(SystemExit):
        b.parse_args(argv)


def test_precursor_values_readonly():
    frame = pd.DataFrame({"precursor_mz": [100.0, 200.0]})
    with pd.option_context("mode.copy_on_write", True):
        values = b.precursor_values(SimpleNamespace(metadata=frame))
        np.testing.assert_array_equal(values, [100, 200])
    pd.testing.assert_frame_equal(frame, pd.DataFrame({"precursor_mz": [100., 200.]}))
    for invalid in (pd.NA, np.inf, 0.0, -5.0, "bad"):
        with pytest.raises(ValueError, match="nonfinite/nonpositive"):
            b.precursor_values(SimpleNamespace(metadata=pd.DataFrame({"precursor_mz": [100, invalid]})))


def test_reference_replication():
    assert b.duplication_factor(1, 4) == 1
    assert b.duplication_factor(8, 4) == 2
    assert b.make_msentropy_reference_records([1, 2, 3, 4], 8) == [1, 2, 3, 4]*2
    with pytest.raises(ValueError):
        b.duplication_factor(6, 4)


def test_dense_score_validation():
    with pytest.raises(RuntimeError, match="shape"):
        b.validate_scores(np.zeros((3, 2)), (2, 3))
    for value in (np.nan, np.inf, 1.1, -.1):
        with pytest.raises(RuntimeError, match="out-of-range"):
            b.validate_scores(np.full((2, 2), value), (2, 2), full=True)
    score = SimpleNamespace(shape=(2, 2), score_fields=("score", "matches"), is_sparse=False)
    with pytest.raises(RuntimeError, match="score-only"):
        b.validate_scores(score, (2, 2))


def test_msentropy_rejects_broadcast():
    core = SimpleNamespace(total_spectra_num=3, search=lambda **kw: np.array(0.5))
    with pytest.raises(RuntimeError, match="score-vector"):
        b.msentropy_open_search(core, [{"peaks": np.ones((2, 2))}])


def test_index_released_before_next_build(tmp_path):
    class Index:
        pass
    old = []
    def build():
        assert not old or old[-1]() is None, "Previous index is still alive"
        obj = Index()
        old.append(weakref.ref(obj))
        return obj
    args = b.parse_args(["--smoke", "--query-index-repeats", "3", "--query-search-repeats", "2"])
    idx = b.TimingTable(tmp_path, "query_index", True)
    search = b.TimingTable(tmp_path, "query_search", True)
    b.run_index_and_search("test", 64, 2, build, lambda _: np.ones((10, 64)),
                           args, idx, search, preparation_included=True)
    assert len(idx.rows) == 3 and len(search.rows) == 2
    assert all(r["duplication_factor"] == 2 for r in idx.rows + search.rows)
    assert all(ref() is None for ref in old)
    b.write_summary_tables(tmp_path)
    assert pd.read_csv(tmp_path / "query_index_summary.csv").n_repeats.iloc[0] == 3


def test_failed_final_build_checkpoints_skipped_searches(tmp_path):
    count = 0
    def build():
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("simulated failure")
        return object()
    args = b.parse_args(["--query-index-repeats", "2", "--query-search-repeats", "2"])
    idx = b.TimingTable(tmp_path, "query_index", False)
    search = b.TimingTable(tmp_path, "query_search", False)
    b.run_index_and_search("test", 64, 1, build, lambda _: pytest.fail("No index"),
                           args, idx, search, preparation_included=True)
    assert [r["status"] for r in idx.rows] == ["ok", "ValueError: simulated failure"]
    assert all(r["status"].startswith("Skipped") for r in search.rows)
    b.write_summary_tables(tmp_path)
    assert pd.read_csv(tmp_path / "query_search_summary.csv").empty


def test_fail_fast_writes_row(tmp_path):
    table = b.TimingTable(tmp_path, "test", True)
    def fail():
        raise RuntimeError("expected")
    with pytest.raises(RuntimeError, match="expected"):
        table.record({"repeat": 0}, fail)
    raw = pd.read_csv(table.path)
    assert raw.status.iloc[0] == "RuntimeError: expected"
    assert pd.isna(raw.seconds.iloc[0])


class MockCollection:
    """Minimal deterministic stand-in used only for orchestration tests."""
    def __init__(self, ids):
        self.ids = np.asarray(ids)
        self.metadata = pd.DataFrame({"precursor_mz": 300. + self.ids * .1})
        self.fragments = self
    def __len__(self):
        return len(self.ids)
    def __iter__(self):
        return iter(self.ids)
    def __getitem__(self, sel):
        return MockCollection(self.ids[sel])
    def count(self, axis):
        return np.full(len(self), 3)
    def sum(self, axis):
        return np.full(len(self), 6.0)
    def get_row(self, i):
        return np.array([50., 100., 150.]), np.array([1., 2., 3.])
    @classmethod
    def concat(cls, collections):
        return cls(np.concatenate([c.ids for c in collections]))


class MockScorer:
    def __init__(self, **kwargs):
        pass
    def matrix(self, spectra, **kwargs):
        n = len(spectra)
        return np.eye(n, dtype=np.float32)
    def build_index(self, spectra):
        return len(spectra)
    def search(self, query, index, **kwargs):
        return np.full((len(query), index), .5, dtype=np.float32)


class MockCleaner:
    def __init__(self, **kwargs):
        pass
    def clean_spectrum_for_search(self, *, precursor_mz, peaks, **kwargs):
        result = peaks.copy()
        result[:, 1] /= result[:, 1].sum()
        return result


class MockCore:
    def __init__(self, **kwargs):
        pass
    def build_index(self, records):
        self.total_spectra_num = len(records)
    def search(self, **kwargs):
        return np.full(self.total_spectra_num, 1., dtype=np.float32)


def install_mock_apis(monkeypatch):
    """Do not claim this verifies the real packages."""
    from types import ModuleType
    pkg = ModuleType("matchms")
    pkg.__file__ = str(SCRIPT)
    pkg.SpectraCollection = MockCollection
    importer = ModuleType("matchms.importing")
    importer.load_ms2_dataset = lambda path, **kw: MockCollection(np.arange(64))
    similarity = ModuleType("matchms.similarity")
    for name in ("Cosine", "CosineBlink", "CosineGreedy", "CosineHungarian", "CosineLinear",
                 "ModifiedCosine", "ModifiedCosineGreedy", "ModifiedCosineHungarian",
                 "ModifiedCosineLinear", "Entropy", "EntropySearch"):
        setattr(similarity, name, type(name, (MockScorer,), {}))
    monkeypatch.setitem(sys.modules, "matchms", pkg)
    monkeypatch.setitem(sys.modules, "matchms.importing", importer)
    monkeypatch.setitem(sys.modules, "matchms.similarity", similarity)
    monkeypatch.setattr(b, "import_msentropy", lambda: (MockCleaner, MockCore))
    monkeypatch.setattr(b, "_package_version", lambda name: "1.0.0" if name == "matchms" else "mock")
    # The test itself must not time repeated expensive collection of Python objects.
    monkeypatch.setattr(b.gc, "collect", lambda: None)


@pytest.mark.parametrize("flags,expected", [
    ([], (36, 12, 12)),
    (["--skip-msentropy", "--parallel-jobs", "1"], (24, 8, 8)),
    (["--skip-query"], (36, 0, 0)),
    (["--skip-all-vs-all", "--skip-msentropy"], (0, 8, 8)),
])
def test_full_mock_workflow(tmp_path, monkeypatch, flags, expected):
    install_mock_apis(monkeypatch)
    b.main(["--smoke", "--output-dir", str(tmp_path), *flags])
    config = json.loads((tmp_path / "benchmark_config.json").read_text())
    assert config["status"] == "complete"
    assert config["synthetic_dataset"]
    for stage, n in zip(b.SUMMARY_GROUPS, expected, strict=True):
        raw = tmp_path / f"{stage}_timings.csv"
        if n:
            frame = pd.read_csv(raw)
            assert len(frame) == n and frame.status.eq("ok").all()
            assert (tmp_path / f"{stage}_summary.csv").exists()
        else:
            assert not raw.exists()
    if expected[0]:
        raw = pd.read_csv(tmp_path / "all_vs_all_timings.csv")
        assert raw.loc[raw.method.eq("cosine_greedy"), "input_representation"].eq("list_spectrum").all()
        assert raw.loc[raw.method.eq("cosine"), "input_representation"].eq("spectra_collection").all()
    with pytest.raises(FileExistsError):
        b.main(["--smoke", "--output-dir", str(tmp_path), *flags])


def test_matrix_input_original_policy():
    args = b.parse_args(["--matrix-input", "collection"])
    method = b.MatrixMethod("test", "cosine", MockScorer, {})
    assert method.input_representation(args) == "spectra_collection"


def test_overwrite_protects_other_benchmark(tmp_path):
    path = tmp_path / "benchmark_config.json"
    path.write_text('{"benchmark": "other_benchmark"}', encoding="utf-8")
    with pytest.raises(ValueError, match="different benchmark"):
        b.prepare_output_dir(tmp_path, overwrite=True)
    assert path.exists()
