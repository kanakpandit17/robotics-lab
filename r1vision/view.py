"""Entry point so `python -m r1vision.view` works."""

import sys

from .viewer import main

if __name__ == "__main__":
    sys.exit(main())
