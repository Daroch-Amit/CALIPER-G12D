#!/usr/bin/env python
"""Entry point for the featurisation step.

Run this from the project root rather than calling the module with -m, so that
objects pickled here keep the src.* import path the app expects.
"""

from src.featurization import main

if __name__ == "__main__":
    main()
