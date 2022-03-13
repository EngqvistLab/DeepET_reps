# Quick hack to import deepet since singularity doesn't seem to pick it up
import inspect
from pathlib import Path
import sys

_this_filename = inspect.getframeinfo(inspect.currentframe()).filename
_this_path = Path(_this_filename).parent.resolve()
sys.path.append(str(_this_path.parent.parent / 'DeepET_reps'))
