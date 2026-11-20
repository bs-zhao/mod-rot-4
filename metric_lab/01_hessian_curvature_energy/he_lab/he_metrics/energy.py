"""Each HE energy metric is a standalone callable."""
import numpy as np
from .scopes import valid_energy

def total_energy(v,*,top_k=1):
    return float(np.maximum(v,0).sum()) if valid_energy(v) else float("nan")
