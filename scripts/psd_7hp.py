"""Run the 7HP PSD command without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ptv_flow.probe_psd import main

if __name__ == "__main__":
    main()
