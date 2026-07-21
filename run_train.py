#!/usr/bin/env python
"""Entry point for model comparison, tuning and bundle export."""

# Use --model together with --no-compare to retrain a known family quickly.
from src.train import main

if __name__ == "__main__":
    main()
