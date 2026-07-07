import inspect

import torch
from mamba_ssm import Mamba


def main() -> None:
    signature = inspect.signature(Mamba)
    for name in ("bimamba_type", "nslices"):
        if name not in signature.parameters:
            raise RuntimeError(f"Installed Mamba is missing `{name}` support.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Mamba(
        d_model=8,
        d_state=4,
        d_conv=2,
        expand=2,
        bimamba_type="v3",
        nslices=2,
    ).to(device)
    x = torch.randn(1, 8, 8, device=device)
    outputs = model(x)

    if not isinstance(outputs, tuple) or len(outputs) != 4:
        raise RuntimeError("TDR-Mamba requires Mamba to return `(out, fwd, bwd, slc)`.")

    shapes = [tuple(item.shape) for item in outputs]
    print(f"TDR-Mamba patched Mamba check passed on {device}: {shapes}")


if __name__ == "__main__":
    main()
