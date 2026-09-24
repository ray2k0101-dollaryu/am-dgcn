"""Where the released code finds its data and writes its outputs.

The internal working copies of these scripts used absolute D:/ paths.  They are
replaced here by one resolver so that a checkout runs from any directory.

    AMDGCN_DATA      datasets root            (default ./data)     retraining only
    AMDGCN_RESULTS   result checkpoints       (default ./results)
    AMDGCN_FIGURES   figure output directory  (default ./figures)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("AMDGCN_DATA", os.path.join(REPO, "data"))
RESULTS = os.environ.get("AMDGCN_RESULTS", os.path.join(REPO, "results"))
FIGURES = os.environ.get("AMDGCN_FIGURES", os.path.join(REPO, "figures"))


def dataset_dir(name):
    """SEED, SEED_IV or DEAP, as they are expected to appear under AMDGCN_DATA."""
    return os.path.join(DATA, name)
