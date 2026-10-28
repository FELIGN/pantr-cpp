# Error model

Decision record for how pantr's C++ code reports failure (`include/pantr/core/errors.hpp`).
Expensive to reverse because every public C++ function, the binding's translator and the Python
suite's message assertions depend on it. Architecture: `design/port_design.md` §2.5 and §2.7.

## The decision

- **Errors are exceptions; incompleteness is a status.** A caller error (a wrong value, type,
  index or state, an exceeded implementation limit) throws. A computation that ran but could not
  finish its work (a spent iteration budget, a dropped search region, a fired depth cap) returns
  normally with a status field in a `[[nodiscard]]` result, which the Python wrapper turns into the
  `RuntimeWarning` it emits today.
- **One hierarchy, rooted twice.** Each pantr exception derives from a standard exception and from
  the empty mixin `pantr::error`:

  | C++ | standard base | Python |
  |---|---|---|
  | `value_error` | `std::invalid_argument` | `ValueError` |
  | `type_error` | `std::invalid_argument` | `TypeError` |
  | `index_error` | `std::out_of_range` | `IndexError` |
  | `state_error` | `std::logic_error` | `RuntimeError` |
  | `not_implemented` | `std::logic_error` | `NotImplementedError` |
  | `capacity_error` | `std::length_error` | `ValueError` |

  A generic `catch (const std::exception&)` keeps working, and `catch (const pantr::error&)` catches
  exactly pantr's own. `capacity_error` marks an implementation limit, as distinct from an argument
  that is wrong in itself. `std::overflow_error` and `std::bad_alloc` keep their standard meaning
  (`OverflowError`, `MemoryError`).
- **The binding registers one explicit translator** for the whole hierarchy rather than relying on
  nanobind's default mapping. Python-only errors (`KeyError`, `AttributeError`, a `TypeError` for a
  wrong Python type, `ImportError`) stay in the Python wrappers.
- **Messages are part of the contract.** The Python suite matches error messages with regular
  expressions, so the C++ message is the single source and carries every fragment an existing
  `match=` expects (DR-13). Values interpolated into a message are spelled as Python spells them,
  through `pantr::py_repr` (`core/format.hpp`): `1e-05`, `3.0`, `(3, 4)`. Python-specific reprs
  (`dtype('float64')`) stay in the wrapper checks.
- **Validation lives in the public C++ API, in every build type** (QB1). Internal kernels validate
  nothing and say so in their documentation.

## Alternatives not taken

- Error codes or `std::expected` at the public API: every caller would have to check, and the
  binding would rebuild exceptions from codes anyway; kept for the C ABI between extension modules
  only, where exceptions cannot cross.
- Assertions (`assert`) as the validation channel: release builds would check nothing, leaving a C++
  caller with undefined behaviour on bad input.
- Raising on incompleteness: a partial result with a status is more useful to a caller than none,
  and it is what the Python API does today (a warning, not an exception).
