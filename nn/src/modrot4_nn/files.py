"""Minimal JSON writer for NN artifacts."""
from pathlib import Path
import json
import numpy as np

def _serial(obj):
    if isinstance(obj,dict):
        return {str(k):_serial(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [_serial(v) for v in obj]
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    return obj

def save_json(value,path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(_serial(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")
