import sys
from pathlib import Path

# partner_modeling/ sits at the repo root and is not part of the installed jaxmarl package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
