import h5py
import numpy as np
import pytest

from scripts.visualize_q_criterion import read_frame, surface_mesh, select_files, build_parser, build_figure


def test_gif_contains_changing_frames_and_requested_timing(tmp_path):
    from PIL import Image, ImageChops
    from scripts.visualize_q_criterion import main

    path = tmp_path / "Static_1D__a.nc"
    with h5py.File(path, "w") as f:
        for name in ("x", "y", "z"):
            f[name] = np.arange(3.) * 1000
        f["t"] = [0., 0.01]
        q = np.broadcast_to(np.arange(3.), (3, 3, 3)).copy()
        f["qcri"] = np.stack([q, q * 2])
    output = tmp_path / "preview.gif"
    main(["--file", str(path), "--gif", str(output), "--stop-frame", "2",
          "--threshold", "0.05", "--fps", "10", "--hole-mask", "none", "--no-open"])
    with Image.open(output) as gif:
        assert gif.n_frames == 2
        assert gif.info["duration"] == 100
        first = gif.convert("RGB")
        gif.seek(1)
        assert ImageChops.difference(first, gif.convert("RGB")).getbbox() is not None


def test_gif_end_stations_are_not_clipped(tmp_path):
    from PIL import Image
    from scripts.visualize_q_criterion import main

    for index in range(5):
        with h5py.File(tmp_path / f"Static_{index + 1}D__a.nc", "w") as f:
            f["x"] = np.linspace(index * 1100, index * 1100 + 1000, 3)
            f["y"] = np.linspace(0, 1000, 3)
            f["z"] = np.linspace(0, 30, 3)
            # Surface spans the thin slab's broad face, making clipping visible.
            f["qcri"] = np.broadcast_to(np.arange(3.)[:, None, None], (3, 3, 3))
    output = tmp_path / "ends.gif"
    main(["--folder", str(tmp_path), "--all-distances", "--gif", str(output),
          "--threshold", "0.05", "--hole-mask", "none", "--no-open"])
    with Image.open(output) as gif:
        # Exclude the legend: each endpoint color must occur in the plot itself.
        pixels = np.asarray(gif.convert("RGB"))[100:480].astype(float)
        for color in ([22, 138, 173], [128, 90, 203]):
            assert (np.linalg.norm(pixels - color, axis=2) < 15).sum() > 50


def test_all_distances_selection_sorted_and_case_specific(tmp_path):
    for name in ("Static_10D__a.nc", "Static_2D__a.nc", "SurgeLF_1D__a.nc"):
        (tmp_path / name).touch()
    args = build_parser().parse_args(["--folder", str(tmp_path), "--case", "static", "--all-distances"])
    assert [p.name for p in select_files(args)] == ["Static_2D__a.nc", "Static_10D__a.nc"]
    (tmp_path / "Static_2D__duplicate.nc").touch()
    with pytest.raises(ValueError, match="Multiple exports"):
        select_files(args)


def test_combined_surfaces_keep_station_coordinates_and_handle_empty_station(tmp_path):
    axes = [np.arange(3.)] * 3
    q = np.broadcast_to(np.arange(3.), (3, 3, 3)).copy()
    shifted = [axes[0] + 10, axes[1], axes[2]]
    fig = build_figure([(tmp_path / "Static_1D__a.nc", axes, q),
                        (tmp_path / "Static_2D__a.nc", shifted, q),
                        (tmp_path / "Static_3D__a.nc", shifted, q * 0)], 0.5, 1, "test")
    assert len(fig.data) == 6
    np.testing.assert_allclose(fig.data[0].x, 0.5)
    np.testing.assert_allclose(fig.data[2].x, 10.5)
    assert len(fig.data[4].i) == 0
    buttons = fig.layout.updatemenus[0].buttons
    assert list(buttons[1].args[0]['visible']) == [False, True] * 3


def test_surface_mesh_interpolates_coordinates_and_excludes_missing_cells():
    axes = [np.array([0., 2., 4.]), np.array([0., 3., 6.]), np.array([0., 5., 10.])]
    q = np.broadcast_to(np.array([0., 1., 2.]), (3, 3, 3)).copy()
    vertices, faces = surface_mesh(axes, q, 0.5)
    assert len(faces) > 0
    np.testing.assert_allclose(vertices[faces, 0], 1.)
    q[:, :, 0] = np.nan
    _, faces = surface_mesh(axes, q, 0.5)
    assert len(faces) == 0


def test_read_export_masks_holes_preserves_sign_and_converts_coordinates(tmp_path):
    path = tmp_path / "field.nc"
    with h5py.File(path, "w") as f:
        for key in ("x", "y", "z"):
            f[key] = [0., 1000.]
        values = np.arange(16, dtype=float).reshape(2, 2, 2, 2) - 10
        f["qcri"] = values
        for key in ("u", "v", "w"):
            velocity = np.ones_like(values)
            velocity[1, 0, 0, 0] = 0
            f[key] = velocity
    axes, q = read_frame(path, "qcri", 1, "mm", "velocity-zero-or-nan")
    np.testing.assert_equal(axes[0], [0, 1])
    assert np.isnan(q[0, 0, 0])
    assert q[0, 0, 1] == -1
    assert q[1, 1, 1] == 5
    with pytest.raises(ValueError, match="Frame"):
        read_frame(path, "qcri", 2, "mm", "none")
