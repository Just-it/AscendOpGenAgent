import sys
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn

try:
    import ascend_kernel
except ImportError:
    pass

# Load SCENARIO_BY_SHAPE from sibling model.py
_model_path = Path(__file__).resolve().parent / "model.py"
_spec = importlib.util.spec_from_file_location("_model", _model_path)
_model_module = importlib.util.module_from_spec(_spec)
sys.modules["_model"] = _model_module
_spec.loader.exec_module(_model_module)
SCENARIO_BY_SHAPE = _model_module.SCENARIO_BY_SHAPE


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _resolve_scenario(self, x: torch.Tensor):
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (N, C, D, H, W), got shape={tuple(x.shape)}")

        shape = tuple(int(dim) for dim in x.shape)
        scenario = SCENARIO_BY_SHAPE.get(shape)
        if scenario is None:
            supported = ", ".join(str(case_shape) for case_shape in SCENARIO_BY_SHAPE.keys())
            raise ValueError(f"Unsupported avg_pool3_d input shape {shape}. Supported shapes: {supported}")
        return scenario

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scenario = self._resolve_scenario(x)
        divisor_override = scenario["divisor_override"] or None
        return torch.ops.npu.avg_pool3d(
            x,
            kernel_size=list(scenario["kernel_size"]),
            stride=list(scenario["stride"]),
            padding=list(scenario["padding"]),
            ceil_mode=scenario["ceil_mode"],
            count_include_pad=scenario["count_include_pad"],
            divisor_override=divisor_override,
        )
