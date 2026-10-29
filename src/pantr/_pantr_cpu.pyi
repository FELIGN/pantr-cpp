"""Which x86-64 levels this CPU and its OS support (private: used by pantr._isa)."""

arch: str = ...

def supports(tag: str) -> bool:
    """Return True if the CPU and its OS support the x86-64 level `tag` ('v2', 'v3', 'v4')."""
