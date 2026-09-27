"""合成データでの単体テスト。実データの結果は見ない。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

nib = pytest.importorskip("nibabel")

from src.stage_curve.features import Thresholds, assert_same_grid, component_labels, lesion_rows
from src.stage_curve.timecourse import adc_ratio_to_day, rising_phase


def _case(mask: np.ndarray, prediction: np.ndarray, adc_inside: float = 400.0):
    adc = np.full(mask.shape, 800.0)
    adc[mask] = adc_inside
    brain = np.ones(mask.shape, dtype=bool)
    b1000 = np.ones(mask.shape)
    return lesion_rows(
        adc, b1000, brain, mask, prediction, voxel_ml=0.001, thresholds=Thresholds(),
        dataset="TEST", subject="case", mask_sequence="", clinical_stage="",
        scale_unknown=False,
    )


def test_two_components_have_exact_volumes():
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    mask[25:30, 25:30, 25:30] = True
    labels, count = component_labels(mask)
    assert count == 2
    rows = _case(mask, np.zeros(mask.shape, dtype=bool))
    volumes = sorted(row["vol_ml"] for row in rows)
    assert volumes == pytest.approx([0.125, 1.0])


def test_adc_ratio_is_half_when_lesion_is_400_and_parenchyma_is_800():
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    rows = _case(mask, np.zeros(mask.shape, dtype=bool), adc_inside=400.0)
    assert rows[0]["adc_ratio_median"] == pytest.approx(0.5)
    assert rows[0]["adc_normal"] == pytest.approx(800.0)


def test_overlap_of_20_percent_is_detected_at_the_preregistered_threshold():
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[5:15, 5:15, 5:15] = True
    prediction = np.zeros(mask.shape, dtype=bool)
    voxels = np.argwhere(mask)
    chosen = voxels[: int(0.2 * len(voxels))]
    prediction[tuple(chosen.T)] = True
    rows = _case(mask, prediction)
    assert rows[0]["overlap_frac"] == pytest.approx(0.2)
    assert rows[0]["detected"] == 1
    assert rows[0]["detected_half"] == 0


def test_shape_or_affine_mismatch_raises():
    data = np.zeros((4, 4, 4), dtype=np.float32)
    reference = nib.Nifti1Image(data, np.eye(4))
    affine = np.eye(4)
    affine[0, 0] = 2.0
    other = nib.Nifti1Image(data, affine)
    with pytest.raises(ValueError):
        assert_same_grid(reference, other)


def test_empty_literature_table_is_not_filled_in():
    empty = pd.DataFrame(columns=["day", "adc_ratio", "source"])
    with pytest.raises(ValueError):
        rising_phase(empty)


def test_day_conversion_uses_only_the_rising_limb():
    table = pd.DataFrame({
        "day": [1, 3, 10],
        "adc_ratio": [0.6, 0.5, 1.2],
        "source": ["test", "test", "test"],
    })
    assert adc_ratio_to_day(0.85, table) == pytest.approx(6.5)
