// Fake communicators for the distributed algorithms' C++ tests, MPI-free.
//
// `ThreadHub` runs `n` ranks as `n` threads of one process: each rank's `HubComm` deposits its
// payload in the hub, waits at a barrier for every rank, reads every rank's payload, and waits
// again before the slots are reused. That is a real collective (every rank must call it, in the
// same order), so a test can run an algorithm on every rank at once and compare the ranks' results.
// `PresetComm` answers every gather with fixed blocks, as the Python suite's fakes do, for the
// cases a real collective cannot produce (another rank's malformed payload).
#pragma once

#include "pantr/core/ndarray.hpp"
#include "pantr/mpi/communicator.hpp"

#include <barrier>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::mpi::testing {

/// The shared state of `n` simulated ranks.
class ThreadHub {
 public:
  explicit ThreadHub(int n)
      : n_(n),
        barrier_(n),
        i64_(static_cast<std::size_t>(n)),
        f64_(static_cast<std::size_t>(n)),
        f32_(static_cast<std::size_t>(n)),
        shapes_(static_cast<std::size_t>(n)) {}

  [[nodiscard]] int size() const noexcept { return n_; }

  template <class T>
  [[nodiscard]] Gathered<T> gather(int rank, const Block<T>& b) {
    check_block(b);
    slots<T>()[static_cast<std::size_t>(rank)].assign(b.data.begin(), b.data.end());
    shapes_[static_cast<std::size_t>(rank)] = b.shape;
    barrier_.arrive_and_wait();
    Gathered<T> out;
    for (int r = 0; r < n_; ++r) {
      const std::vector<T>& s = slots<T>()[static_cast<std::size_t>(r)];
      out.data.insert(out.data.end(), s.begin(), s.end());
      out.shapes.push_back(shapes_[static_cast<std::size_t>(r)]);
    }
    barrier_.arrive_and_wait();
    return out;
  }

  template <class T>
  [[nodiscard]] NdArray<T> reduce(int rank, std::span<const T> data,
                                  std::span<const std::size_t> shape) {
    slots<T>()[static_cast<std::size_t>(rank)].assign(data.begin(), data.end());
    barrier_.arrive_and_wait();
    // Every rank sees the same sizes, so on a mismatch every rank throws (after the second wait),
    // as comm_capsule.h asks of a real communicator.
    bool same = true;
    NdArray<T> out(std::vector<std::size_t>(shape.begin(), shape.end()));
    for (int r = 0; r < n_; ++r) {
      const std::vector<T>& s = slots<T>()[static_cast<std::size_t>(r)];
      same = same && s.size() == slots<T>()[0].size();
      for (std::size_t i = 0; i < out.size() && i < s.size(); ++i) {
        out.data()[i] += s[i];
      }
    }
    barrier_.arrive_and_wait();
    if (!same) {
      throw value_error("allreduce_sum: the ranks passed tensors of different sizes.");
    }
    return out;
  }

 private:
  template <class T>
  std::vector<std::vector<T>>& slots() {
    if constexpr (std::is_same_v<T, std::int64_t>) {
      return i64_;
    } else if constexpr (std::is_same_v<T, double>) {
      return f64_;
    } else {
      return f32_;
    }
  }

  int n_;
  std::barrier<> barrier_;
  std::vector<std::vector<std::int64_t>> i64_;
  std::vector<std::vector<double>> f64_;
  std::vector<std::vector<float>> f32_;
  std::vector<std::vector<std::size_t>> shapes_;
};

/// One rank of a `ThreadHub`.
class HubComm {
 public:
  HubComm(ThreadHub& hub, int rank) : hub_(&hub), rank_(rank) {}
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return hub_->size(); }
  template <class T>
  [[nodiscard]] Gathered<T> allgatherv(const Block<T>& b) const {
    return hub_->gather(rank_, b);
  }
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const {
    Gathered<std::int64_t> gi = hub_->gather(rank_, index);
    return {std::move(gi), hub_->gather(rank_, value)};
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return hub_->reduce(rank_, data, shape);
  }

 private:
  ThreadHub* hub_;
  int rank_;
};

static_assert(Communicator<HubComm>);

/// Run `body(comm)` on `n` ranks, one thread each, and return each rank's result in rank order.
/// An exception on a rank is rethrown on the caller (the first rank's first); a test must not
/// throw on one rank before a collective the others enter.
template <class R, class F>
[[nodiscard]] std::vector<R> run_ranks(int n, F body) {
  ThreadHub hub(n);
  std::vector<std::optional<R>> results(static_cast<std::size_t>(n));
  std::vector<std::exception_ptr> errors(static_cast<std::size_t>(n));
  std::vector<std::thread> threads;
  for (int r = 0; r < n; ++r) {
    threads.emplace_back([&, r] {
      try {
        results[static_cast<std::size_t>(r)].emplace(body(HubComm(hub, r)));
      } catch (...) {
        errors[static_cast<std::size_t>(r)] = std::current_exception();
      }
    });
  }
  for (std::thread& t : threads) {
    t.join();
  }
  for (const std::exception_ptr& e : errors) {
    if (e) {
      std::rethrow_exception(e);
    }
  }
  std::vector<R> out;
  for (std::optional<R>& r : results) {
    out.push_back(std::move(*r));
  }
  return out;
}

/// A communicator whose gathers return fixed blocks whatever is sent (the Python fakes' preset
/// `allgather_results`); its reductions return the caller's own tensor.
class PresetComm {
 public:
  PresetComm(int rank, int size, Gathered<std::int64_t> i64, Gathered<double> f64)
      : rank_(rank),
        size_(size),
        i64_(std::make_shared<Gathered<std::int64_t>>(std::move(i64))),
        f64_(std::make_shared<Gathered<double>>(std::move(f64))) {}
  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int size() const noexcept { return size_; }
  [[nodiscard]] Gathered<std::int64_t> allgatherv(const Block<std::int64_t>& /*b*/) const {
    return *i64_;
  }
  [[nodiscard]] Gathered<double> allgatherv(const Block<double>& /*b*/) const { return *f64_; }
  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& /*index*/, const Block<double>& /*value*/) const {
    return {*i64_, *f64_};
  }
  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    return SelfCommunicator{}.allreduce_sum(data, shape);
  }

 private:
  int rank_;
  int size_;
  std::shared_ptr<Gathered<std::int64_t>> i64_;
  std::shared_ptr<Gathered<double>> f64_;
};

static_assert(Communicator<PresetComm>);

}  // namespace pantr::mpi::testing
