#!/usr/bin/env python3
"""SymAgent entry point — delegates to src/run.py.

Usage:
    python run.py <mode> [options]
    python -m src.run <mode> [options]   # equivalent

Modes: plan, execute, explore, train, evaluate, full_pipeline
"""

import sys
import os

# Ensure project root is in path so `src` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.run import main

if __name__ == "__main__":
    main()
