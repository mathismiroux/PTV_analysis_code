from __future__ import annotations

import subprocess
import sys

from ptv_flow.postprocess import phase_average_volume
from ptv_flow.reader import FlowDataset


def test_assess_phase_average_quality_writes_outputs(tiny_flow_path, tmp_path):
    phase_average = tmp_path / "phase_average.nc"
    output = tmp_path / "phase_quality.png"
    with FlowDataset(tiny_flow_path) as flow:
        phase_average_volume(
            flow,
            phase_average,
            n_phase_bins=4,
            frequency_hz=1.0,
            chunk_size=2,
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/assess_phase_average_quality.py",
            str(phase_average),
            "--raw-file",
            str(tiny_flow_path),
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--min-valid-fraction",
            "0.5",
            "--output",
            str(output),
        ],
        check=True,
    )

    assert output.exists()
    assert output.with_suffix(".csv").exists()
