"""Which x86-64 levels this CPU and its OS support (private: used by pantr._isa)."""

arch: str = ...

def _levels_from_registers(
    leaf1_ecx: int, leaf7_ebx: int, ext1_ecx: int, xcr0: int
) -> tuple[bool, bool, bool]:
    """Return (v2, v3, v4) as decided from these register values (for tests)."""

def supports(tag: str) -> bool:
    """Return True if the CPU and its OS support the x86-64 level `tag` ('v2', 'v3', 'v4')."""
