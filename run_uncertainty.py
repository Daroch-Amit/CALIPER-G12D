#!/usr/bin/env python
"""Entry point for conformal calibration and the applicability domain."""

# Run this after training; it edits the bundle in place.
from src.uncertainty import main

if __name__ == "__main__":
    main()
