/// @file tags.hpp
/// @brief Sparse named tags for grid cells and facets.
///
/// Two immutable registries attach integer labels to a sparse subset of a grid's entities,
/// following the model of `dolfinx.mesh.MeshTags`: each named tag is a pair of parallel arrays
/// `(ids, values)` (or `(keys, values)` for facets), sorted by id; an entity not listed is
/// untagged. Memory is proportional to the number of *tagged* entities.
///
/// A registry is a value: `set` and `remove` return a new registry and leave the receiver as it
/// was. Tags are kept in insertion order; setting a name that exists replaces the entry in place,
/// so the order of `names()` is that of a Python `dict` under reassignment. Entries are shared
/// between a registry and the ones derived from it. The dense scatter (`to_dense`) stays in the
/// Python wrapper, where the numpy dtype logic lives.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/detail/py_text.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pantr::grid {

namespace detail {

/// One named tag.
///
/// Attributes:
///     name (std::string): The tag name.
///     keys (std::vector<std::int64_t>): Cell ids sorted ascending, or facet keys as flattened
///         `(cell id, local facet id)` pairs sorted lexicographically.
///     values (std::vector<std::int64_t>): One value per id (per key pair).
struct TagEntry {
  std::string name;
  std::vector<std::int64_t> keys;
  std::vector<std::int64_t> values;
};

/// The ordered tag entries of a registry; an entry is shared by the registries derived from it.
using TagEntries = std::vector<std::shared_ptr<const TagEntry>>;

/// Find a tag by name.
///
/// Args:
///     entries (const TagEntries&): The entries.
///     name (std::string_view): The tag name.
///
/// Returns:
///     std::size_t: The index of the entry, or `entries.size()` when there is none.
[[nodiscard]] inline std::size_t find_tag(const TagEntries& entries, std::string_view name) {
  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (entries[i]->name == name) {
      return i;
    }
  }
  return entries.size();
}

/// Get the entries after setting `name` (in place when it exists, appended otherwise).
[[nodiscard]] inline TagEntries with_entry(const TagEntries& entries,
                                           std::shared_ptr<const TagEntry> entry) {
  TagEntries out = entries;
  const std::size_t at = find_tag(out, entry->name);
  if (at == out.size()) {
    out.push_back(std::move(entry));
  } else {
    out[at] = std::move(entry);
  }
  return out;
}

/// Get the entries without `name`.
///
/// Raises:
///     index_error: There is no tag named `name`.
[[nodiscard]] inline TagEntries without_entry(const TagEntries& entries, std::string_view name) {
  const std::size_t at = find_tag(entries, name);
  if (at == entries.size()) {
    throw index_error("no tag named " + py_repr_str(name) + ".");
  }
  TagEntries out = entries;
  out.erase(out.begin() + static_cast<std::ptrdiff_t>(at));
  return out;
}

/// Get the names of the entries, in insertion order.
[[nodiscard]] inline std::vector<std::string> entry_names(const TagEntries& entries) {
  std::vector<std::string> out;
  out.reserve(entries.size());
  for (const auto& e : entries) {
    out.push_back(e->name);
  }
  return out;
}

/// Check the length of `values` against the number of keys.
///
/// Raises:
///     value_error: The lengths differ.
inline void check_values_length(std::size_t n_keys, std::size_t n_values) {
  if (n_values != n_keys) {
    throw value_error("values must be a scalar or have length " + std::to_string(n_keys) +
                      "; got length " + std::to_string(n_values) + ".");
  }
}

/// The shared representation of a `CellTags`.
///
/// Attributes:
///     num_cells (std::int64_t): Number of cells of the owning grid.
///     entries (TagEntries): The tags, in insertion order.
struct CellTagsRep {
  std::int64_t num_cells = 0;
  TagEntries entries;
};

/// The shared representation of a `FacetTags`.
///
/// Attributes:
///     num_cells (std::int64_t): Number of cells of the owning grid.
///     facets_per_cell (std::int64_t): Number of local facets per cell.
///     entries (TagEntries): The tags, in insertion order.
struct FacetTagsRep {
  std::int64_t num_cells = 0;
  std::int64_t facets_per_cell = 0;
  TagEntries entries;
};

}  // namespace detail

/// Sparse named integer tags over a grid's cells.
///
/// Each tag named `name` is a pair of parallel `int64` arrays `(ids, values)` sorted by `ids`; a
/// cell not listed in `ids` is untagged under `name`. Distinct tag names are independent.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::CellTagsRep>): The cell count and the entries.
class CellTags {
 public:
  /// Create an empty registry.
  ///
  /// Args:
  ///     num_cells (std::int64_t): Number of cells of the owning grid, at least 0.
  ///
  /// Raises:
  ///     value_error: `num_cells` is negative.
  explicit CellTags(std::int64_t num_cells) {
    if (num_cells < 0) {
      throw value_error("num_cells must be >= 0; got " + std::to_string(num_cells) + ".");
    }
    auto rep = std::make_shared<detail::CellTagsRep>();
    rep->num_cells = num_cells;
    rep_ = std::move(rep);
  }

  /// Get the number of cells of the owning grid.
  ///
  /// Returns:
  ///     std::int64_t: The cell count; valid cell ids are `[0, num_cells())`.
  [[nodiscard]] std::int64_t num_cells() const noexcept { return rep_->num_cells; }

  /// Get a registry with the tag `name` created or replaced by the association `ids -> values`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name; an existing tag of that name is replaced in place.
  ///     ids (std::span<const std::int64_t>): Cell ids, each in `[0, num_cells())`, unique.
  ///     values (std::span<const std::int64_t>): One value per id (the Python wrapper broadcasts a
  ///         scalar).
  ///
  /// Returns:
  ///     CellTags: The new registry; this one is unchanged.
  ///
  /// Raises:
  ///     value_error: An id is out of range, the ids contain a duplicate, or `values` has a
  ///         length other than `ids.size()`.
  [[nodiscard]] CellTags set(std::string_view name, std::span<const std::int64_t> ids,
                             std::span<const std::int64_t> values) const {
    if (!ids.empty()) {
      const auto [mn, mx] = std::minmax_element(ids.begin(), ids.end());
      if (*mn < 0 || *mx >= rep_->num_cells) {
        throw value_error("cell ids must be in [0, " + std::to_string(rep_->num_cells) +
                          "); got range [" + std::to_string(*mn) + ", " + std::to_string(*mx) +
                          "].");
      }
    }
    std::vector<std::size_t> order(ids.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
    for (std::size_t i = 1; i < order.size(); ++i) {
      if (ids[order[i]] == ids[order[i - 1]]) {
        throw value_error("cell ids for tag " + detail::py_repr_str(name) +
                          " must be unique; got duplicates.");
      }
    }
    detail::check_values_length(ids.size(), values.size());
    auto entry = std::make_shared<detail::TagEntry>();
    entry->name = std::string(name);
    entry->keys.reserve(ids.size());
    entry->values.reserve(ids.size());
    for (const std::size_t i : order) {
      entry->keys.push_back(ids[i]);
      entry->values.push_back(values[i]);
    }
    auto rep = std::make_shared<detail::CellTagsRep>();
    rep->num_cells = rep_->num_cells;
    rep->entries = detail::with_entry(rep_->entries, std::move(entry));
    return CellTags(std::move(rep));
  }

  /// Get a registry without the tag `name`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name.
  ///
  /// Returns:
  ///     CellTags: The new registry; this one is unchanged.
  ///
  /// Raises:
  ///     index_error: There is no tag named `name` (the Python wrapper raises `KeyError`).
  [[nodiscard]] CellTags remove(std::string_view name) const {
    auto rep = std::make_shared<detail::CellTagsRep>();
    rep->num_cells = rep_->num_cells;
    rep->entries = detail::without_entry(rep_->entries, name);
    return CellTags(std::move(rep));
  }

  /// Report whether a tag named `name` exists.
  ///
  /// Args:
  ///     name (std::string_view): Candidate tag name.
  ///
  /// Returns:
  ///     bool: Whether `name` is a registered tag.
  [[nodiscard]] bool contains(std::string_view name) const noexcept {
    return detail::find_tag(rep_->entries, name) != rep_->entries.size();
  }

  /// Get the registered tag names.
  ///
  /// Returns:
  ///     std::vector<std::string>: The names in insertion order.
  [[nodiscard]] std::vector<std::string> names() const {
    return detail::entry_names(rep_->entries);
  }

  /// Get the number of registered tags.
  ///
  /// Returns:
  ///     std::size_t: The count of distinct tag names.
  [[nodiscard]] std::size_t size() const noexcept { return rep_->entries.size(); }

  /// Get the `(ids, values)` arrays of the tag `name`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name.
  ///
  /// Returns:
  ///     std::pair<std::span<const std::int64_t>, std::span<const std::int64_t>>: The ids sorted
  ///     ascending and their values, views valid while this registry (or any sharing the entry)
  ///     lives.
  ///
  /// Raises:
  ///     index_error: There is no tag named `name`.
  [[nodiscard]] std::pair<std::span<const std::int64_t>, std::span<const std::int64_t>> get(
      std::string_view name) const {
    const std::size_t at = detail::find_tag(rep_->entries, name);
    if (at == rep_->entries.size()) {
      throw index_error("no tag named " + detail::py_repr_str(name) + ".");
    }
    const detail::TagEntry& e = *rep_->entries[at];
    return {e.keys, e.values};
  }

  /// Get the shared representation.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::CellTagsRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::CellTagsRep>& rep() const noexcept {
    return rep_;
  }

 private:
  explicit CellTags(std::shared_ptr<const detail::CellTagsRep> rep) : rep_(std::move(rep)) {}

  std::shared_ptr<const detail::CellTagsRep> rep_;
};

/// Sparse named integer tags over a grid's local facets.
///
/// Each facet is addressed by a `(cell id, local facet id)` key, with the local facet id in
/// `[0, facets_per_cell())`. Each tag named `name` is a pair `(keys, values)`, `keys` being the
/// flattened `(M, 2)` row-major array of keys sorted lexicographically and `values` one value per
/// key.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::FacetTagsRep>): The counts and the entries.
class FacetTags {
 public:
  /// Create an empty registry.
  ///
  /// Args:
  ///     num_cells (std::int64_t): Number of cells of the owning grid, at least 0.
  ///     facets_per_cell (std::int64_t): Number of local facets per cell, at least 1.
  ///
  /// Raises:
  ///     value_error: `num_cells` is negative or `facets_per_cell < 1`.
  FacetTags(std::int64_t num_cells, std::int64_t facets_per_cell) {
    if (num_cells < 0) {
      throw value_error("num_cells must be >= 0; got " + std::to_string(num_cells) + ".");
    }
    if (facets_per_cell < 1) {
      throw value_error("facets_per_cell must be >= 1; got " + std::to_string(facets_per_cell) +
                        ".");
    }
    auto rep = std::make_shared<detail::FacetTagsRep>();
    rep->num_cells = num_cells;
    rep->facets_per_cell = facets_per_cell;
    rep_ = std::move(rep);
  }

  /// Get the number of cells of the owning grid.
  ///
  /// Returns:
  ///     std::int64_t: The cell count.
  [[nodiscard]] std::int64_t num_cells() const noexcept { return rep_->num_cells; }

  /// Get the number of local facets per cell.
  ///
  /// Returns:
  ///     std::int64_t: `2 * ndim` for an axis-aligned box grid.
  [[nodiscard]] std::int64_t facets_per_cell() const noexcept { return rep_->facets_per_cell; }

  /// Get a registry with the tag `name` created or replaced by the association `keys -> values`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name; an existing tag of that name is replaced in place.
  ///     keys (std::span<const std::int64_t>): `M` keys as the flattened `(M, 2)` row-major array
  ///         of `(cell id, local facet id)` rows, unique, with the cell id in
  ///         `[0, num_cells())` and the local facet id in `[0, facets_per_cell())`. The `(M, 2)`
  ///         shape check lives in the Python wrapper, which spells numpy shapes.
  ///     values (std::span<const std::int64_t>): One value per key (the Python wrapper broadcasts
  ///         a scalar).
  ///
  /// Returns:
  ///     FacetTags: The new registry; this one is unchanged.
  ///
  /// Raises:
  ///     value_error: `keys` holds an odd number of entries, a cell id or a local facet id is out
  ///         of range, a key is duplicated, or `values` has a length other than `M`.
  [[nodiscard]] FacetTags set(std::string_view name, std::span<const std::int64_t> keys,
                              std::span<const std::int64_t> values) const {
    if (keys.size() % 2 != 0) {
      throw value_error("keys must hold (cell id, local facet id) pairs; got " +
                        std::to_string(keys.size()) + " entries.");
    }
    const std::size_t m = keys.size() / 2;
    if (m > 0) {
      std::int64_t cmin = keys[0];
      std::int64_t cmax = keys[0];
      std::int64_t fmin = keys[1];
      std::int64_t fmax = keys[1];
      for (std::size_t i = 1; i < m; ++i) {
        cmin = (std::min)(cmin, keys[2 * i]);
        cmax = (std::max)(cmax, keys[2 * i]);
        fmin = (std::min)(fmin, keys[2 * i + 1]);
        fmax = (std::max)(fmax, keys[2 * i + 1]);
      }
      if (cmin < 0 || cmax >= rep_->num_cells) {
        throw value_error("facet cell ids must be in [0, " + std::to_string(rep_->num_cells) +
                          "); got range [" + std::to_string(cmin) + ", " + std::to_string(cmax) +
                          "].");
      }
      if (fmin < 0 || fmax >= rep_->facets_per_cell) {
        throw value_error("local facet ids must be in [0, " +
                          std::to_string(rep_->facets_per_cell) + "); got range [" +
                          std::to_string(fmin) + ", " + std::to_string(fmax) + "].");
      }
    }
    // Lexicographic sort by (cell id, local facet id), stable as numpy's lexsort is.
    std::vector<std::size_t> order(m);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return keys[2 * a] != keys[2 * b] ? keys[2 * a] < keys[2 * b]
                                        : keys[2 * a + 1] < keys[2 * b + 1];
    });
    for (std::size_t i = 1; i < m; ++i) {
      if (keys[2 * order[i]] == keys[2 * order[i - 1]] &&
          keys[2 * order[i] + 1] == keys[2 * order[i - 1] + 1]) {
        throw value_error("facet keys for tag " + detail::py_repr_str(name) +
                          " must be unique; got duplicates.");
      }
    }
    detail::check_values_length(m, values.size());
    auto entry = std::make_shared<detail::TagEntry>();
    entry->name = std::string(name);
    entry->keys.reserve(2 * m);
    entry->values.reserve(m);
    for (const std::size_t i : order) {
      entry->keys.push_back(keys[2 * i]);
      entry->keys.push_back(keys[2 * i + 1]);
      entry->values.push_back(values[i]);
    }
    auto rep = std::make_shared<detail::FacetTagsRep>();
    rep->num_cells = rep_->num_cells;
    rep->facets_per_cell = rep_->facets_per_cell;
    rep->entries = detail::with_entry(rep_->entries, std::move(entry));
    return FacetTags(std::move(rep));
  }

  /// Get a registry without the tag `name`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name.
  ///
  /// Returns:
  ///     FacetTags: The new registry; this one is unchanged.
  ///
  /// Raises:
  ///     index_error: There is no tag named `name` (the Python wrapper raises `KeyError`).
  [[nodiscard]] FacetTags remove(std::string_view name) const {
    auto rep = std::make_shared<detail::FacetTagsRep>();
    rep->num_cells = rep_->num_cells;
    rep->facets_per_cell = rep_->facets_per_cell;
    rep->entries = detail::without_entry(rep_->entries, name);
    return FacetTags(std::move(rep));
  }

  /// Report whether a tag named `name` exists.
  ///
  /// Args:
  ///     name (std::string_view): Candidate tag name.
  ///
  /// Returns:
  ///     bool: Whether `name` is a registered tag.
  [[nodiscard]] bool contains(std::string_view name) const noexcept {
    return detail::find_tag(rep_->entries, name) != rep_->entries.size();
  }

  /// Get the registered tag names.
  ///
  /// Returns:
  ///     std::vector<std::string>: The names in insertion order.
  [[nodiscard]] std::vector<std::string> names() const {
    return detail::entry_names(rep_->entries);
  }

  /// Get the number of registered tags.
  ///
  /// Returns:
  ///     std::size_t: The count of distinct tag names.
  [[nodiscard]] std::size_t size() const noexcept { return rep_->entries.size(); }

  /// Get the `(keys, values)` arrays of the tag `name`.
  ///
  /// Args:
  ///     name (std::string_view): Tag name.
  ///
  /// Returns:
  ///     std::pair<std::span<const std::int64_t>, std::span<const std::int64_t>>: The keys as the
  ///     flattened `(M, 2)` array sorted lexicographically, and the `M` values; views valid while
  ///     this registry (or any sharing the entry) lives.
  ///
  /// Raises:
  ///     index_error: There is no tag named `name`.
  [[nodiscard]] std::pair<std::span<const std::int64_t>, std::span<const std::int64_t>> get(
      std::string_view name) const {
    const std::size_t at = detail::find_tag(rep_->entries, name);
    if (at == rep_->entries.size()) {
      throw index_error("no tag named " + detail::py_repr_str(name) + ".");
    }
    const detail::TagEntry& e = *rep_->entries[at];
    return {e.keys, e.values};
  }

  /// Get the shared representation.
  ///
  /// Returns:
  ///     const std::shared_ptr<const detail::FacetTagsRep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const detail::FacetTagsRep>& rep() const noexcept {
    return rep_;
  }

 private:
  explicit FacetTags(std::shared_ptr<const detail::FacetTagsRep> rep) : rep_(std::move(rep)) {}

  std::shared_ptr<const detail::FacetTagsRep> rep_;
};

}  // namespace pantr::grid
