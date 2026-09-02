from __future__ import annotations

import numpy as np


INVALID_SAMPLE_MODES = ("zero", "nan", "zero-or-nan", "none")


def validate_invalid_samples(invalid_samples: str) -> None:
    if invalid_samples not in INVALID_SAMPLE_MODES:
        raise ValueError(
            "invalid_samples must be one of "
            f"{', '.join(repr(mode) for mode in INVALID_SAMPLE_MODES)}"
        )


def valid_component_samples(data: np.ndarray, invalid_samples: str = "nan") -> np.ndarray:
    validate_invalid_samples(invalid_samples)
    if invalid_samples == "zero":
        return data != 0.0
    if invalid_samples == "nan":
        return np.isfinite(data)
    if invalid_samples == "zero-or-nan":
        return (data != 0.0) & np.isfinite(data)
    return np.ones(data.shape, dtype=bool)


def valid_vector_samples(
    components: dict[str, np.ndarray],
    invalid_samples: str = "nan",
) -> np.ndarray:
    validate_invalid_samples(invalid_samples)
    values = list(components.values())
    if invalid_samples == "none":
        return np.ones(values[0].shape, dtype=bool)
    if invalid_samples == "zero":
        return np.logical_or.reduce([data != 0.0 for data in values])
    finite = np.logical_and.reduce([np.isfinite(data) for data in values])
    if invalid_samples == "nan":
        return finite
    nonzero = np.logical_or.reduce([data != 0.0 for data in values])
    return finite & nonzero
