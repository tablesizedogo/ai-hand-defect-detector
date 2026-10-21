"""The packaged HADM backend must stay identical to the ComfyUI node pack's copy (when the pack is checked out next to it)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "comfyui_hand_fixer" / "custom_nodes" / "comfyui-hand-fixer" / "hand_fixer"


@pytest.mark.parametrize("name", ["hadm_backend", "geometry", "detector"])
def test_same_as_node_pack(name):
    if not PACK.exists():
        pytest.skip("node pack not checked out")
    assert (ROOT / "hand_defects" / f"_{name}.py").read_bytes() == (PACK / f"{name}.py").read_bytes()


def test_calibration_same_as_node_pack():
    if not PACK.exists():
        pytest.skip("node pack not checked out")
    assert (ROOT / "models" / "hadm_calibration.json").read_bytes() == (PACK.parent / "models" / "hadm_calibration.json").read_bytes()
