# Python wrappers over private bindings

Decision record for how pantr's public Python modules reach the C++ library. Expensive to reverse
because every ported module, every test that reaches an internal and the per-call cost of the
public API follow it. Architecture: `design/port_design.md` §3.1, §3.8 and §8.1.

## The decision

- **The public Python types and functions stay Python.** Each public module keeps its names,
  signatures, defaults, docstrings and `@overload`s, and calls a private submodule of the compiled
  extension (`pantr._pantr.quad`, `.basis`, `.change_basis`, `.tolerance`, ...). A public class
  holds its C++ value in one attribute, `_h`. The bindings are flat: already-normalised arguments
  (C-contiguous arrays of the exact dtype, integers for enumerations), one function per return
  type, and one function or overload per dtype (`float` and `double` instantiations).
- **What stays in the wrapper:** dtype dispatch (`_is_float32`), `np.asarray` and contiguity,
  union-typed arguments (`PointsLattice` or an array, a scalar or a sequence), `out=` handling
  with the existing Python messages (a non-contiguous `out` is filled from a temporary), the
  Python-specific checks (an `isinstance` test, a Python type's repr in a message), and pickling
  through `__reduce__`. **Validation of values is C++'s**, once, with the messages the suite
  matches; where the wrapper must look at an argument before C++ does (to size an `out`), it
  hands an inconsistent argument to the C++ call first so the C++ message is the one raised.
- **Immutable values cache what their properties return at construction** (DR-25). A wrapper of
  an immutable C++ value (`PointsLattice`, `QuadratureRule`) reads its scalar properties and the
  read-only views of its arrays once, when it is built, and its properties return those Python
  objects: a property read costs what main's pure-Python property cost, not a binding call. The
  views are NumPy arrays over the C++ value's shared representation, kept alive by a capsule
  holding a reference to it, so they never go stale and never copy.
- **Results cross without a copy.** A C++ result (`NdArray`) moves its buffer into a capsule the
  NumPy array owns.
- **`pantr._testing` is the only private surface a test imports** (Di2). Internals worth testing
  directly and exposed by no public function (one-point recurrences, the generic Gram
  projection, the process tables' bound and reset) are bound in `_pantr._testing` and re-exported
  by `pantr._testing`. No shim keeps a retired private path alive; a test of a retired Numba
  detail (a serial twin, a dispatch threshold) is replaced by the property it protected.
- **`_pantr._kernels` is the interim surface of the coexistence rule** (DR-2): a Python caller not
  yet ported reaches a ported kernel through it, with the argument conventions of the Numba kernel
  it replaces (any strides, output written in place). It shrinks as callers are ported and is
  deleted with the last one.
- **Typing:** the stubs are generated from the bindings (`make stubs`) and committed. A call whose
  dtype is only known at run time passes its array arguments untyped (`cast(Any, ...)` or an
  `Any`-typed module alias) rather than satisfying one overload per branch.

## Alternatives not taken

- nanobind classes as the public types: one Python class must front two C++ instantiations, and
  docstrings, autodoc, doctests, overloads and union-typed arguments live badly in binding code.
- Python keeps validating and C++ checks nothing: the C++ API is public too, and validation in two
  places drifts.
- Properties forwarding to the binding on every read: a binding call per read, measurably slower
  than main's pure-Python properties (the stage-0 measurement), for a value that cannot change.
- Tests importing the private wrapper modules or the bindings directly: every port would move the
  paths they import, and a test could not tell an internal from an accident of the layout.
