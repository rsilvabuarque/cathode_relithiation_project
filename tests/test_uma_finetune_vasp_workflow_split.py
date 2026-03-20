from __future__ import annotations

from hydrorelith.pipelines.uma_finetune_vasp_workflow import (
    _ensure_diversity_by_swap,
    _split_records_stratified,
)


def _synthetic_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    idx = 0
    for temperature in [393, 433, 473, 493]:
        for delith in [0.0, 4.17, 8.33, 12.5, 16.67, 20.83, 25.0]:
            for _ in range(4):
                records.append(
                    {
                        "case_id": f"case_{idx}",
                        "case_dir": f"/tmp/case_{idx}",
                        "natoms": 96,
                        "temperature_k": temperature,
                        "pressure_mpa": 0.08,
                        "lithiation_pct": 100.0 - delith,
                        "delithiation_pct": delith,
                        "extxyz_path": f"/tmp/case_{idx}.extxyz",
                    }
                )
                idx += 1
    return records


def test_split_is_90_5_5_with_expected_sizes() -> None:
    records = _synthetic_records()
    splits = _split_records_stratified(records, train_frac=0.90, val_frac=0.05, test_frac=0.05, seed=7)
    total = len(records)

    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == total
    assert len(splits["train"]) == 101
    assert len(splits["val"]) == 6
    assert len(splits["test"]) == 5


def test_diversity_swap_preserves_split_sizes_and_increases_variety() -> None:
    records = _synthetic_records()
    splits = _split_records_stratified(records, train_frac=0.90, val_frac=0.05, test_frac=0.05, seed=11)

    sizes_before = {k: len(v) for k, v in splits.items()}
    splits = _ensure_diversity_by_swap(splits, field="temperature_k", seed=11)
    splits = _ensure_diversity_by_swap(splits, field="delithiation_pct", seed=11)

    for split_name in ["train", "val", "test"]:
        assert len(splits[split_name]) == sizes_before[split_name]

    for split_name in ["train", "val", "test"]:
        temps = {row["temperature_k"] for row in splits[split_name] if row.get("temperature_k") is not None}
        delith = {row["delithiation_pct"] for row in splits[split_name] if row.get("delithiation_pct") is not None}
        assert len(temps) >= 2
        assert len(delith) >= 2