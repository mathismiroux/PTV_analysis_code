from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CASES_FILE = Path("cases.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs")


@dataclass(frozen=True)
class CaseFiles:
    velocity: Path | None = None
    vorticity: Path | None = None
    q_criterion: Path | None = None
    phase_signal: Path | None = None


@dataclass(frozen=True)
class FlowCase:
    case_id: str
    label: str
    motion_type: str
    downstream_distance: float | str
    frequency_hz: float | None
    reduced_frequency: float | None
    amplitude: float | str | None
    u_inf: float | None
    rotor_diameter: float | None
    rotor_frequency_hz: float | None
    blade_passing_frequency_hz: float | None
    files: CaseFiles
    registry_path: Path

    @property
    def base_processing_id(self) -> str:
        return self.case_id

    def default_output_path(
        self,
        product_name: str,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
        unique: bool = False,
    ) -> Path:
        output_dir = output_root / self.base_processing_id
        if unique:
            output_dir = _next_available_output_dir(output_dir, product_name)
        return output_dir / product_name

    def processing_output_dir(
        self,
        processing_id: str | None = None,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> Path:
        return output_root / (processing_id or self.base_processing_id)

    def resolve_existing_product(
        self,
        product_name: str,
        processing_id: str | None = None,
        output_root: Path = DEFAULT_OUTPUT_ROOT,
    ) -> Path:
        if processing_id is not None:
            product = self.processing_output_dir(processing_id, output_root) / product_name
            if not product.exists():
                raise FileNotFoundError(
                    f"Could not find {product_name!r} for processing_id "
                    f"{processing_id!r}: {product}"
                )
            return product

        candidates = sorted(output_root.glob(f"{self.base_processing_id}*/{product_name}"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(
                f"Could not find {product_name!r} for case {self.case_id!r}. "
                f"Expected something like "
                f"{output_root / self.base_processing_id / product_name}."
            )
        processing_ids = ", ".join(candidate.parent.name for candidate in candidates)
        raise ValueError(
            f"Found multiple {product_name!r} products for case {self.case_id!r}: "
            f"{processing_ids}. Pass --processing-id to choose one."
        )

    def metadata_attributes(self) -> dict[str, str | float]:
        attrs: dict[str, str | float] = {
            "case_id": self.case_id,
            "processing_id": self.base_processing_id,
            "label": self.label,
            "motion_type": self.motion_type,
            "downstream_distance": self.downstream_distance,
        }
        optional = {
            "frequency_hz": self.frequency_hz,
            "reduced_frequency": self.reduced_frequency,
            "u_inf": self.u_inf,
            "rotor_diameter": self.rotor_diameter,
            "rotor_frequency_hz": self.rotor_frequency_hz,
            "blade_passing_frequency_hz": self.blade_passing_frequency_hz,
        }
        for key, value in optional.items():
            if value is not None:
                attrs[key] = float(value)
        if self.amplitude is not None:
            attrs["amplitude"] = self.amplitude
        return attrs

    def require_velocity(self) -> Path:
        if self.files.velocity is None:
            raise ValueError(
                f"Case {self.case_id!r} does not define files.velocity in "
                f"{self.registry_path}"
            )
        if not self.files.velocity.exists():
            raise FileNotFoundError(
                f"Velocity file for case {self.case_id!r} does not exist: "
                f"{self.files.velocity}"
            )
        return self.files.velocity

    def require_u_inf(self) -> float:
        if self.u_inf is None:
            raise ValueError(
                f"Case {self.case_id!r} does not define u_inf, which is required "
                "for wake-deficit output."
            )
        return float(self.u_inf)

    def validate_for_temporal_average(self) -> None:
        self.require_velocity()
        self.require_u_inf()

    def validate_for_phase_average(self) -> None:
        self.require_velocity()
        self.require_u_inf()
        if self.frequency_hz is None and self.files.phase_signal is None:
            raise ValueError(
                f"Case {self.case_id!r} must define frequency_hz or "
                "files.phase_signal for phase averaging."
            )


def _next_available_output_dir(base_output_dir: Path, product_name: str) -> Path:
    candidate = base_output_dir
    counter = 2
    while (candidate / product_name).exists():
        candidate = base_output_dir.with_name(f"{base_output_dir.name}_{counter:02d}")
        counter += 1
    return candidate


def _resolve_optional_path(registry_path: Path, value: object) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = registry_path.parent / path
    return path


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _required_downstream_distance(case_id: str, data: dict[str, Any]) -> float | str:
    value = data.get("downstream_distance")
    if value in (None, ""):
        raise ValueError(f"Case {case_id!r} must define downstream_distance")
    return value


def _case_from_mapping(
    registry_path: Path,
    case_id: str,
    data: dict[str, Any],
) -> FlowCase:
    files = data.get("files") or {}
    if not isinstance(files, dict):
        raise ValueError(f"Case {case_id!r} files entry must be a mapping")

    return FlowCase(
        case_id=case_id,
        label=str(data.get("label", case_id)),
        motion_type=str(data.get("motion_type", "unknown")),
        downstream_distance=_required_downstream_distance(case_id, data),
        frequency_hz=_optional_float(data.get("frequency_hz")),
        reduced_frequency=_optional_float(data.get("reduced_frequency")),
        amplitude=data.get("amplitude"),
        u_inf=_optional_float(data.get("u_inf")),
        rotor_diameter=_optional_float(data.get("rotor_diameter")),
        rotor_frequency_hz=_optional_float(data.get("rotor_frequency_hz")),
        blade_passing_frequency_hz=_optional_float(
            data.get("blade_passing_frequency_hz")
        ),
        files=CaseFiles(
            velocity=_resolve_optional_path(registry_path, files.get("velocity")),
            vorticity=_resolve_optional_path(registry_path, files.get("vorticity")),
            q_criterion=_resolve_optional_path(
                registry_path, files.get("q_criterion")
            ),
            phase_signal=_resolve_optional_path(
                registry_path, files.get("phase_signal")
            ),
        ),
        registry_path=registry_path,
    )


def load_cases(path: str | Path = DEFAULT_CASES_FILE) -> dict[str, FlowCase]:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Could not find case registry: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Case registry must contain a mapping: {registry_path}")

    cases_raw = raw.get("cases", raw)
    if not isinstance(cases_raw, dict):
        raise ValueError(f"Case registry 'cases' entry must be a mapping: {registry_path}")

    cases = {}
    for case_id, data in cases_raw.items():
        if not isinstance(data, dict):
            raise ValueError(f"Case {case_id!r} must be a mapping")
        cases[str(case_id)] = _case_from_mapping(registry_path, str(case_id), data)
    return cases


def load_case(
    case_id: str,
    path: str | Path = DEFAULT_CASES_FILE,
) -> FlowCase:
    cases = load_cases(path)
    try:
        return cases[case_id]
    except KeyError as exc:
        available = ", ".join(sorted(cases))
        raise KeyError(
            f"Case {case_id!r} not found in {path}. Available cases: {available}"
        ) from exc
