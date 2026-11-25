"""CPU regressions and opt-in CPU-oracle checks for RDNA2 training.

Run explicitly with pytest. Set LIGHTGBM_RUN_RDNA2_CERTIFICATION=1 only on a
suitable gfx1030 host. LIGHTGBM_FROZEN_QUANTILE_DIR optionally supplies the
exported input.json, features.f32le, labels.f32le and CPU booster.txt.
"""

import contextlib
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import lightgbm as lgb
from lightgbm.basic import _LIB

_RUN_RDNA2 = os.environ.get("LIGHTGBM_RUN_RDNA2_CERTIFICATION") == "1"
_FROZEN_DIR = os.environ.get("LIGHTGBM_FROZEN_QUANTILE_DIR")


def _data():
    # Odd number of rows and columns exercises the tail of both tilings.
    rng = np.random.default_rng(193)
    x = rng.normal(size=(4099, 9)).astype(np.float32)
    x[:, 8] = (x[:, 8] > 0).astype(np.float32)
    y = (3 * x[:, 0] + 2 * x[:, 1] + x[:, 8]).astype(np.float32)
    return x, y


def _params():
    return {
        "objective": "quantile",
        "alpha": 0.2,
        "num_leaves": 15,
        "max_bin": 127,
        "min_data_in_bin": 1,
        "min_data_in_leaf": 10,
        "force_col_wise": True,
        "enable_bundle": False,
        "is_enable_sparse": False,
        "seed": 23,
        "num_threads": 2,
        "verbosity": -1,
    }


def _fit(x, y, params, rounds):
    return lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=rounds)


def _split_structure(node):
    if "split_feature" not in node:
        return (node.get("leaf_index"), node["leaf_count"])
    return (
        node["split_feature"],
        node["threshold"],
        node["decision_type"],
        node["default_left"],
        node["missing_type"],
        _split_structure(node["left_child"]),
        _split_structure(node["right_child"]),
    )


def _check_oracle(cpu, other, x, rounds):
    assert cpu.num_trees() == other.num_trees() == rounds
    assert [_split_structure(tree["tree_structure"]) for tree in other.dump_model()["tree_info"]] == [
        _split_structure(tree["tree_structure"]) for tree in cpu.dump_model()["tree_info"]
    ]
    expected, actual = cpu.predict(x), other.predict(x)
    assert np.isfinite(expected).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    print(f"CPU oracle: max_abs_diff={np.max(np.abs(actual - expected)):.17g}", flush=True)


@pytest.mark.parametrize("feature", [-1, 9, 8])
def test_invalid_forced_split_during_create_and_reset_is_rejected(tmp_path, feature):
    x, y = _data()
    x[:, 8] = 1  # A valid original column index with no usable inner feature.
    path = tmp_path / "invalid-split.json"
    path.write_text(
        json.dumps(
            {
                "feature": 0,
                "threshold": 0.1,
                "left": {
                    "feature": feature,
                    "threshold": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    params = _params()
    with pytest.raises(lgb.basic.LightGBMError, match=f"Forced splits file includes feature index {feature}"):
        _fit(x, y, dict(params, forcedsplits_filename=path), 1)
    booster = lgb.Booster(params, train_set=lgb.Dataset(x, label=y))
    with pytest.raises(lgb.basic.LightGBMError, match=f"Forced splits file includes feature index {feature}"):
        booster.reset_parameter({"forcedsplits_filename": str(path)})
    booster.update()  # A rejected reset must leave the previous learner usable.
    assert booster.num_trees() == 1


@pytest.mark.parametrize("limits", [{"num_leaves": 2}, {"num_leaves": 15, "max_depth": 1}])
def test_forced_splits_stop_at_tree_limits(tmp_path, limits):
    x, y = _data()
    path = tmp_path / "deep-split.json"
    path.write_text(
        json.dumps(
            {
                "feature": 0,
                "threshold": 0.1,
                "left": {
                    "feature": 1,
                    "threshold": 0.1,
                },
            }
        ),
        encoding="utf-8",
    )
    booster = _fit(x, y, dict(_params(), **limits, forcedsplits_filename=path), 1)
    tree = booster.dump_model()["tree_info"][0]
    assert tree["num_leaves"] == 2
    assert tree["tree_structure"]["split_feature"] == 0
    assert np.isfinite(booster.predict(x)).all()


@pytest.mark.parametrize("rows", [(-1, 2), (2**31 - 1, 1), (0, 0)])
def test_dense_c_api_rejects_invalid_dimensions_before_reading_matrix(rows):
    x = (ctypes.c_float * 1)(0.0)
    pointer = ctypes.cast(x, ctypes.c_void_p)
    data = (ctypes.c_void_p * 2)(pointer, pointer)
    nrow = (ctypes.c_int32 * 2)(*rows)
    row_major = (ctypes.c_int * 2)(1, 1)
    handle = ctypes.c_void_p()
    result = _LIB.LGBM_DatasetCreateFromMats(
        2,
        data,
        0,
        nrow,
        1,
        row_major,
        b"device_type=cpu num_threads=1",
        None,
        ctypes.byref(handle),
    )
    assert result == -1
    assert handle.value is None
    assert b"Check failed" in _LIB.LGBM_GetLastError()


def test_cpu_dense_matrix_layout_matches_generic_loader():
    x, y = _data()
    x[::19, 2] = np.nan
    x[::7, 0] = 0.0
    params = _params()
    canonical = _fit(x.astype(np.float64), y, params, 5)
    packed = _fit(x, y, params, 5)
    _check_oracle(canonical, packed, x, 5)
    np.testing.assert_array_equal(packed.predict(x).view(np.uint64), canonical.predict(x).view(np.uint64))


@pytest.mark.skipif(not _RUN_RDNA2, reason="requires explicit AMD gfx1030 opt-in")
@pytest.mark.parametrize("case", ["h64", "h64_supertile", "h128_masked", "limited_pool", "wide_bins", "forced_splits"])
def test_rdna2_matches_cpu_oracle(case, tmp_path, capfd):
    x, y = _data()
    params = _params()
    if case in {"h64", "h64_supertile"}:
        params["max_bin"] = 63
        if case == "h64_supertile":
            x, y = np.tile(x, (3, 1)), np.tile(y, 3)  # Just above the 12,288-row threshold.
    elif case == "h128_masked":
        params["feature_fraction"] = 0.75
    elif case == "limited_pool":
        params["histogram_pool_size"] = 0.01
    elif case == "wide_bins":
        params["max_bin"] = 255
    elif case == "forced_splits":
        path = tmp_path / "force.json"
        path.write_text(json.dumps({"feature": 0, "threshold": 0.1}), encoding="utf-8")
        params["forcedsplits_filename"] = path
    cpu = _fit(x, y, dict(params, device_type="cpu"), 5)
    rdna2 = _fit(x, y, dict(params, device_type="rdna2", verbosity=1), 5)
    log = capfd.readouterr().out
    if case == "wide_bins":
        assert "RDNA2 dataset population:" not in log
        assert "RDNA2 packed dataset:" not in log
    else:
        assert "RDNA2 dataset population:" in log
        assert "RDNA2 packed dataset:" in log
        assert f"h64={'yes' if case.startswith('h64') else 'h128'}" in log
    _check_oracle(cpu, rdna2, x, 5)


@pytest.mark.skipif(not _RUN_RDNA2, reason="requires explicit AMD gfx1030 opt-in")
@pytest.mark.parametrize("case", ["binary", "binary-sampled", "quantile-q90"])
def test_h128_preserves_serial_feature_pruning(case, capfd):
    # Deep trees with many weak features exercise pruning at the parent, not
    # just agreement between the winning root splits. Before the fix, H128
    # could mark a pruned feature as eligible and choose it in a descendant.
    train_rows = 16385
    quantile = case == "quantile-q90"
    valid_start = train_rows + (64 if quantile else 0)
    total = valid_start + 2501
    seed = 20260809
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((total, 257), dtype=np.float32)
    nonlinear = np.tanh(x[:, 4]) if quantile else (x[:, 4] > 0.25).astype(np.float32)
    target = (
        2.8 * x[:, 0]
        - 2.1 * x[:, 1]
        + 1.6 * x[:, 2] * x[:, 3]
        + 1.2 * nonlinear
        - 0.8 * (x[:, 5] < -0.5).astype(np.float32)
        + 0.15 * rng.standard_normal(total).astype(np.float32)
    )
    y = target if quantile else (target > 0.0).astype(np.float32)
    params = dict(
        _params(),
        objective="quantile" if quantile else "binary",
        alpha=0.9,
        learning_rate=0.05,
        num_leaves=40,
        max_depth=6,
        min_data_in_leaf=50,
        min_data_in_bin=3,
        bagging_fraction=1.0 if quantile else 0.9,
        bagging_freq=1,
        feature_fraction=0.9 if case == "binary-sampled" else 1.0,
        lambda_l1=0.1,
        lambda_l2=0.1,
        seed=seed,
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        num_threads=8,
    )
    rounds = 20
    cpu = _fit(x[:train_rows], y[:train_rows], dict(params, device_type="cpu"), rounds)
    rdna2_params = dict(params, device_type="rdna2", gpu_device_id=0, verbosity=1)
    rdna2 = _fit(x[:train_rows], y[:train_rows], rdna2_params, rounds)
    log = capfd.readouterr().out
    assert "RDNA2 packed dataset:" in log
    assert "h64=h128" in log
    _check_oracle(cpu, rdna2, x[valid_start:], rounds)


def _frozen_inputs():
    directory = Path(_FROZEN_DIR)
    spec = json.loads((directory / "input.json").read_bytes())
    features, labels = (directory / "features.f32le").read_bytes(), (directory / "labels.f32le").read_bytes()
    assert hashlib.sha256(features).hexdigest() == spec["feature_sha256"]
    assert hashlib.sha256(labels).hexdigest() == spec["label_sha256"]
    x = np.frombuffer(features, dtype="<f4").reshape(spec["rows"], spec["columns"])
    y = np.frombuffer(labels, dtype="<f4")
    config = spec["original_head_config"]
    seed = config["seed"]
    params = {
        k: config[k]
        for k in (
            "learning_rate",
            "max_depth",
            "num_leaves",
            "min_data_in_leaf",
            "max_bin",
            "lambda_l1",
            "lambda_l2",
            "feature_fraction",
            "seed",
        )
    }
    assert not config["categorical_feature_indices"]
    params.update(
        objective="quantile",
        alpha=np.format_float_positional(np.float32(config["objective"]["alpha"]), trim="-"),
        gpu_device_id=config["execution"]["gpu_device_id"],
        num_threads=config["execution"]["num_threads"],
        feature_fraction_seed=seed,
        bagging_seed=seed,
        data_random_seed=seed,
        force_col_wise=True,
        verbosity=-1,
    )
    return directory, x, y, params, config["rounds"]


@pytest.mark.skipif(_FROZEN_DIR is None, reason="requires exported frozen ETH q20 inputs")
def test_frozen_eth_cpu_matches_saved_oracle():
    directory, x, y, params, rounds = _frozen_inputs()
    reference = lgb.Booster(model_file=directory / "booster.txt")
    cpu = _fit(x, y, dict(params, device_type="cpu"), rounds)
    _check_oracle(reference, cpu, x, rounds)
    np.testing.assert_array_equal(cpu.predict(x).view(np.uint64), reference.predict(x).view(np.uint64))


@pytest.mark.skipif(
    not _RUN_RDNA2 or _FROZEN_DIR is None, reason="requires explicit gfx1030 opt-in and frozen ETH data"
)
@pytest.mark.parametrize("dense_layout", [False, True], ids=["saved-config", "require-gpu-histogram"])
def test_frozen_eth_rdna2_matches_cpu_oracle(dense_layout):
    _, x, y, params, rounds = _frozen_inputs()
    if dense_layout:
        # Keep values, labels and training settings; isolate dense packing from
        # feature grouping, which can send the original configuration to Serial.
        params.update(enable_bundle=False, is_enable_sparse=False)
    cpu = _fit(x, y, dict(params, device_type="cpu"), rounds)
    # Use pytest -s and redirect to disk for the frozen replay. A watchdog may
    # stop the process before pytest can print or collect a result.
    params = dict(params, device_type="rdna2", num_iterations=rounds, verbosity=1)
    stdout = sys.stdout
    log_lines = []

    class PhaseLog:
        def write(self, message):
            log_lines.append(message)
            return stdout.write(message)

        def flush(self):
            stdout.flush()

    with contextlib.redirect_stdout(PhaseLog()):
        print("RDNA2 dataset preparation", flush=True)
        dataset = lgb.Dataset(x, label=y, params=params).construct()
        print("RDNA2 booster creation", flush=True)
        rdna2 = lgb.Booster(params=params, train_set=dataset)
        for step in range(rounds):
            print(f"RDNA2 training iteration {step + 1}/{rounds}", flush=True)
            rdna2.update()
    print("RDNA2 training finished; comparing with CPU", flush=True)
    log = "".join(log_lines)
    assert "[device_type: rdna2]" in rdna2.model_to_string()
    gpu_packing = "RDNA2 packed dataset:" in log
    print(
        f"RDNA2 dataset population={'RDNA2 dataset population:' in log}, GPU histogram eligible={gpu_packing}",
        flush=True,
    )
    if dense_layout:
        assert "RDNA2 dataset population:" in log
        assert gpu_packing
        assert "h64=yes" in log
    _check_oracle(cpu, rdna2, x, rounds)
