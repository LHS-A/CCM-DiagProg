from src.models.unified_causal_ccm import OUTPUT_DIMS, TASKS


def test_six_task_contract():
    assert TASKS == ("oc_diag", "sys_diag", "oc_reg", "hba1c", "short", "long")
    assert OUTPUT_DIMS == (3, 3, 4, 1, 4, 1)
    assert sum(OUTPUT_DIMS) == 16

