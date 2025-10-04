#!/usr/bin/env python3
# Debug version of main.py to show full tracebacks

import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import main

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Full traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)