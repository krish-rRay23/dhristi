"""Allow running Dṛṣṭi CLI via `python -m drishti`."""

import sys
from drishti.cli import main

if __name__ == "__main__":
    sys.exit(main())
