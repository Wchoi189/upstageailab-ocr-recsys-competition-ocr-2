"""
Vendor directory for third-party code.

This directory contains vendored versions of external libraries.
The vendor directory is automatically added to sys.path to allow
imports like 'from strhub...' to work.
"""

import sys
from pathlib import Path

# Add vendor directory to Python path so imports like 'from strhub...' work
vendor_dir = Path(__file__).parent
if str(vendor_dir) not in sys.path:
    sys.path.insert(0, str(vendor_dir))
