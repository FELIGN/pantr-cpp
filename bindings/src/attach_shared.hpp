// How a pantr extension module other than _pantr joins the process's one pool (DR-20).
#pragma once

#include "pantr/core/errors.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>

#include <cstring>
#include <string>

namespace pantr::bindings {

/// Attach this module's parallel state to the table of the loaded _pantr rung.
///
/// Call it first in the module's `NB_MODULE` body: from then on this module's parallel loops run
/// on _pantr's pool and its thread settings are _pantr's.
///
/// Raises:
///     state_error: The capsule is not pantr's shared-parallelism table, or the table's layout
///         differs from this module's headers (see `attach_shared_parallelism`).
///     nanobind::python_error: pantr._ext cannot be imported.
inline void attach_shared_parallelism_from_pantr() {
  namespace nb = nanobind;
  const nb::object capsule_obj =
      nb::module_::import_("pantr._ext").attr("shared_parallelism_capsule")();
  const auto capsule = nb::cast<nb::capsule>(capsule_obj);
  const char* name = capsule.name();
  if (name == nullptr || std::strcmp(name, pantr::parallel::kSharedParallelismCapsule) != 0) {
    throw pantr::state_error(std::string("expected a capsule named '") +
                             pantr::parallel::kSharedParallelismCapsule + "', got '" +
                             (name == nullptr ? "" : name) + "'");
  }
  pantr::parallel::attach_shared_parallelism(
      static_cast<const pantr::parallel::SharedParallelism*>(capsule.data()));
}

}  // namespace pantr::bindings
