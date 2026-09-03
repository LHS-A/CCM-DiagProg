import torch
from src.models.unified_causal_ccm import OUTPUT_DIMS, TASKS


def test_six_task_contract():
    assert len(TASKS) == 6
    assert OUTPUT_DIMS == (3, 3, 4, 1, 4, 2)
    assert sum(OUTPUT_DIMS) == 17

