from __future__ import annotations

import pytest

from ptv_flow.cases import load_case, load_cases


def test_load_cases_resolves_paths_relative_to_registry():
    cases = load_cases("tests/data/cases.yaml")

    case = cases["tiny_static_x3p5d"]

    assert case.case_id == "tiny_static_x3p5d"
    assert case.base_processing_id == "tiny_static_x3p5d"
    assert case.label == "Tiny static fixture"
    assert case.u_inf == 4.0
    assert case.files.velocity is not None
    assert case.files.velocity.name == "tiny_flow.nc"
    assert case.files.velocity.exists()
    assert case.default_output_path("mean.nc").as_posix() == (
        "outputs/tiny_static_x3p5d/mean.nc"
    )


def test_case_default_output_path_adds_integer_when_product_exists(tmp_path):
    case = load_case("tiny_static_x3p5d", "tests/data/cases.yaml")
    existing = tmp_path / "tiny_static_x3p5d" / "mean.nc"
    existing.parent.mkdir()
    existing.write_text("already here")

    output = case.default_output_path("mean.nc", output_root=tmp_path, unique=True)

    assert output == tmp_path / "tiny_static_x3p5d_02" / "mean.nc"


def test_case_resolves_existing_product(tmp_path):
    case = load_case("tiny_static_x3p5d", "tests/data/cases.yaml")
    mean = tmp_path / "tiny_static_x3p5d" / "mean.nc"
    mean.parent.mkdir()
    mean.write_text("mean")

    assert case.resolve_existing_product("mean.nc", output_root=tmp_path) == mean
    assert (
        case.resolve_existing_product(
            "mean.nc",
            processing_id="tiny_static_x3p5d",
            output_root=tmp_path,
        )
        == mean
    )


def test_case_resolve_existing_product_rejects_ambiguous_outputs(tmp_path):
    case = load_case("tiny_static_x3p5d", "tests/data/cases.yaml")
    for processing_id in ("tiny_static_x3p5d", "tiny_static_x3p5d_02"):
        mean = tmp_path / processing_id / "mean.nc"
        mean.parent.mkdir()
        mean.write_text("mean")

    with pytest.raises(ValueError, match="--processing-id"):
        case.resolve_existing_product("mean.nc", output_root=tmp_path)


def test_case_validation_rejects_missing_velocity():
    case = load_case("missing_velocity_x3p5d", "tests/data/cases.yaml")

    with pytest.raises(ValueError, match="files.velocity"):
        case.validate_for_temporal_average()


def test_case_validation_rejects_missing_u_inf():
    case = load_case("missing_u_inf_x3p5d", "tests/data/cases.yaml")

    with pytest.raises(ValueError, match="u_inf"):
        case.validate_for_temporal_average()


def test_case_phase_validation_requires_frequency_or_phase_signal(tmp_path):
    registry = tmp_path / "cases.yaml"
    velocity = (tmp_path / "tiny_flow.nc").resolve()
    velocity.write_bytes(b"not opened")
    registry.write_text(
        f"""
cases:
  bad_phase_case:
    label: Bad phase case
    motion_type: surge
    downstream_distance: 3.5D
    frequency_hz: null
    u_inf: 4.0
    files:
      velocity: "{velocity.as_posix()}"
      phase_signal: null
""",
        encoding="utf-8",
    )
    case = load_case("bad_phase_case", registry)

    with pytest.raises(ValueError, match="frequency_hz"):
        case.validate_for_phase_average()


def test_load_cases_rejects_missing_downstream_distance(tmp_path):
    registry = tmp_path / "cases.yaml"
    registry.write_text(
        """
cases:
  bad_case:
    label: Bad case
    motion_type: static
    u_inf: 4.0
    files:
      velocity: tiny_flow.nc
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="downstream_distance"):
        load_cases(registry)
