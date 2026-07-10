from __future__ import annotations

from pathlib import Path

import h5py


SOURCE = Path("Static_3.5D__b128f.nc")
OUTPUT = Path("tests/data/tiny_flow.nc")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(SOURCE, "r") as src, h5py.File(OUTPUT, "w") as dst:
        dst.attrs["source_file"] = str(SOURCE)
        dst.attrs["description"] = "Small test fixture extracted from b128f data."

        dst.create_dataset("t", data=src["t"][:4])
        dst.create_dataset("z", data=src["z"][13:16])
        dst.create_dataset("y", data=src["y"][:5])
        dst.create_dataset("x", data=src["x"][:6])

        for name in ("u", "v", "w"):
            data = src[name][:4, 13:16, :5, :6]
            dst.create_dataset(name, data=data)

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
