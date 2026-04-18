from __future__ import annotations

from pathlib import Path


MODEL_PAIRS = [
    ("app/models/garbege.pt", "app/models/garbege.onnx"),
    ("app/models/fire_recall2.pt", "app/models/fire_recall2.onnx"),
    ("app/models/fire_smoke.pt", "app/models/fire_smoke.onnx"),
]


def register_spd_compat() -> None:
    try:
        import torch
        import torch.nn as nn
        import ultralytics.nn.modules.block as block
    except Exception:
        return

    if hasattr(block, "space_to_depth"):
        return

    class space_to_depth(nn.Module):  # noqa: N801 - keep checkpoint class name
        def __init__(self, dimension: int = 1):
            super().__init__()
            self.d = dimension

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]],
                1,
            )

    setattr(block, "space_to_depth", space_to_depth)


def export_model(pt_path: Path, onnx_path: Path) -> None:
    register_spd_compat()
    from ultralytics import YOLO

    if not pt_path.exists():
        print(f"[skip] missing model: {pt_path}")
        return

    print(f"[export] {pt_path} -> {onnx_path}")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", opset=12, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported_path.read_bytes())
    print(f"[ok] {onnx_path}")


if __name__ == "__main__":
    for pt_file, onnx_file in MODEL_PAIRS:
        export_model(Path(pt_file), Path(onnx_file))
