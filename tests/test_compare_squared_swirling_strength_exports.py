from __future__ import annotations

import numpy as np
import h5py

from scripts.compare_squared_swirling_strength_exports import (
    compare_swirl_fields,
    infer_columns,
    inspect_grid,
    interpolate_to_common_grid,
    label_plane_components,
    label_vortex_components,
    load_field_file,
    reshape_structured_grid,
)


def _write_table(path, x_values, y_values, z_values, scale=1.0):
    lines = ["Position_X,Position_Y,Position_Z,LambdaCi2,u"]
    for x in x_values:
        for y in y_values:
            for z in z_values:
                swirl = scale * (10.0 + x + 2.0 * y + 3.0 * z)
                lines.append(f"{x},{y},{z},{swirl},{4.0 + x}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_infer_columns_accepts_aliases():
    columns = ["Position_X", "Position_Y", "Position_Z", "LambdaCi2", "velocity_u"]

    inferred = infer_columns(columns)

    assert inferred["x"] == "Position_X"
    assert inferred["y"] == "Position_Y"
    assert inferred["z"] == "Position_Z"
    assert inferred["lambda_ci_squared"] == "LambdaCi2"
    assert inferred["u"] == "velocity_u"


def test_load_inspect_and_reshape_structured_grid(tmp_path):
    path = tmp_path / "field.csv"
    _write_table(path, [0.0, 100.0], [-50.0, 50.0], [0.0, 25.0])

    field = load_field_file(path)
    inspection = inspect_grid(
        field,
        label="test",
        rotor_diameter=1.2,
        coordinate_unit="mm",
    )
    grid = reshape_structured_grid(
        field,
        label="test",
        rotor_diameter=1.2,
        u_inf=4.0,
        coordinate_unit="mm",
    )

    assert inspection.structured
    assert inspection.nx == 2
    assert inspection.ny == 2
    assert inspection.nz == 2
    assert grid.swirl_star.shape == (2, 2, 2)
    assert grid.u is not None
    np.testing.assert_allclose(grid.u[:, 0, 0], [1.0, 26.0])


def test_interpolate_and_compare_on_common_grid(tmp_path):
    fine_path = tmp_path / "fine.csv"
    coarse_path = tmp_path / "coarse.csv"
    _write_table(fine_path, [0.0, 50.0, 100.0], [0.0, 50.0], [0.0, 50.0])
    _write_table(coarse_path, [0.0, 100.0], [0.0, 50.0], [0.0, 50.0])
    fine = reshape_structured_grid(
        load_field_file(fine_path), "fine", 1.2, 4.0, "mm"
    )
    coarse = reshape_structured_grid(
        load_field_file(coarse_path), "coarse", 1.2, 4.0, "mm"
    )

    fine_on_coarse, coarse_overlap = interpolate_to_common_grid(fine, coarse)
    metrics = compare_swirl_fields(fine_on_coarse, coarse_overlap)

    assert fine_on_coarse.shape == coarse_overlap.shape
    assert metrics["rms_error"] == 0.0
    assert metrics["pearson_correlation"] == 1.0


def test_label_vortex_components_filters_small_regions(tmp_path):
    path = tmp_path / "field.csv"
    _write_table(path, [0.0, 50.0, 100.0], [0.0, 50.0], [0.0, 50.0])
    grid = reshape_structured_grid(load_field_file(path), "field", 1.2, 4.0, "mm")

    components = label_vortex_components(
        grid,
        threshold=grid.swirl_star.max() - 1e-9,
        threshold_label="peak",
        min_component_size=1,
        remove_boundary_components=False,
    )

    assert len(components) == 1
    assert components[0].size_cells == 1
    assert components[0].peak == grid.swirl_star.max()


def test_load_hdf5_uses_davis_values_without_hole_mask(tmp_path):
    path = tmp_path / "field.nc"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([0.0, 1.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("swir", data=np.array([[[[0.0, -4.0]]]]))
        h5.create_dataset("u", data=np.array([[[[0.0, 1.0]]]]))
        h5.create_dataset("v", data=np.array([[[[0.0, 1.0]]]]))
        h5.create_dataset("w", data=np.array([[[[0.0, 1.0]]]]))

    field = load_field_file(
        path,
        swirl_variable="swir",
        swirl_sign="negative",
        hole_mask="none",
    )

    np.testing.assert_allclose(field.swirl_squared, [0.0, 4.0])


def test_load_hdf5_can_mask_velocity_zero_holes(tmp_path):
    path = tmp_path / "field.nc"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("x", data=np.array([0.0, 1.0]))
        h5.create_dataset("y", data=np.array([0.0]))
        h5.create_dataset("z", data=np.array([0.0]))
        h5.create_dataset("swir", data=np.array([[[[-2.0, -4.0]]]]))
        h5.create_dataset("u", data=np.array([[[[0.0, 1.0]]]]))
        h5.create_dataset("v", data=np.array([[[[0.0, 1.0]]]]))
        h5.create_dataset("w", data=np.array([[[[0.0, 1.0]]]]))

    field = load_field_file(
        path,
        swirl_variable="swir",
        swirl_sign="negative",
        hole_mask="velocity-zero",
    )

    assert np.isnan(field.swirl_squared[0])
    assert field.swirl_squared[1] == 4.0


def test_label_plane_components_marks_centroid_and_peak():
    values = np.array(
        [
            [0.0, 5.0, 0.0],
            [0.0, 7.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    components = label_plane_components(
        values,
        x=np.array([10.0, 20.0, 30.0]),
        y=np.array([100.0, 200.0, 300.0]),
        threshold=4.0,
        min_component_size=1,
        remove_boundary_components=False,
    )

    assert len(components) == 1
    assert components[0].size_cells == 2
    assert components[0].peak == 7.0
    assert components[0].peak_x == 20.0
    assert components[0].peak_y == 200.0
