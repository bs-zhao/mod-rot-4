"""Family B: W,Y plus frozen trained MLP -> HE, separate from blind W,Y."""
from .registry import METHODS, infer_he, register_method
from .adapter import FrozenWMLP
