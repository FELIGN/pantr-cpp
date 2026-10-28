// NdArray: shape bookkeeping, row-major views, value semantics, buffer hand-off, limits.
#include <doctest/doctest.h>

#include <pantr/core/ndarray.hpp>

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

TEST_CASE("NdArray: shape, size, zero initialization") {
  pantr::NdArray<double> a({2, 3, 4});
  CHECK(a.rank() == 3);
  CHECK(a.size() == 24);
  CHECK(a.shape()[0] == 2);
  CHECK(a.shape()[2] == 4);
  for (const double x : a.flat()) {
    CHECK(x == 0.0);
  }
  pantr::NdArray<float> scalar(std::vector<std::size_t>{});
  CHECK(scalar.rank() == 0);
  CHECK(scalar.size() == 1);
  pantr::NdArray<double> empty({4, 0});
  CHECK(empty.size() == 0);
}

TEST_CASE("NdArray: views are row-major") {
  pantr::NdArray<double> a({2, 3});
  auto v = a.view<2>();
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      v(i, j) = static_cast<double>(10 * i + j);
    }
  }
  // Row-major: element (i, j) sits at flat index i * 3 + j.
  CHECK(a.data()[0 * 3 + 2] == 2.0);
  CHECK(a.data()[1 * 3 + 0] == 10.0);
  CHECK(a.data()[1 * 3 + 2] == 12.0);
  const auto& ca = a;
  CHECK(ca.view<2>()(1, 1) == 11.0);
  CHECK(v.extent(0) == 2);
  CHECK(v.extent(1) == 3);
}

TEST_CASE("NdArray: a view of the wrong rank is a value_error") {
  pantr::NdArray<double> a({2, 3});
  CHECK_THROWS_WITH_AS(static_cast<void>(a.view<3>()), "cannot view an array of rank 2 as rank 3",
                       pantr::value_error);
}

TEST_CASE("NdArray: copies do not alias, moves leave an empty array") {
  pantr::NdArray<double> a({3});
  a.data()[1] = 7.0;
  pantr::NdArray<double> b = a;
  b.data()[1] = 8.0;
  CHECK(a.data()[1] == 7.0);
  CHECK(b.data()[1] == 8.0);
  pantr::NdArray<double> c = std::move(a);
  CHECK(c.data()[1] == 7.0);
  // NOLINTBEGIN(bugprone-use-after-move): the moved-from state is part of the contract under test
  CHECK(a.size() == 0);
  CHECK(a.data() == nullptr);
  // NOLINTEND(bugprone-use-after-move)
  a = b;  // a moved-from array can be assigned to
  CHECK(a.data()[1] == 8.0);
  b = c;
  CHECK(b.data()[1] == 7.0);
}

TEST_CASE("NdArray: into_buffer hands over the elements without copying") {
  pantr::NdArray<double> a({4});
  a.data()[3] = 1.5;
  const double* before = a.data();
  auto buffer = std::move(a).into_buffer();
  CHECK(buffer.get() == before);
  CHECK(buffer[3] == 1.5);
  // NOLINTBEGIN(bugprone-use-after-move): into_buffer's consumed state is part of the contract
  CHECK(a.size() == 0);
  CHECK(a.data() == nullptr);
  // NOLINTEND(bugprone-use-after-move)
}

TEST_CASE("NdArray: an element count that overflows is a capacity_error") {
  const std::size_t big = std::numeric_limits<std::size_t>::max() / 4;
  CHECK_THROWS_AS(pantr::NdArray<double>({big, 8}), pantr::capacity_error);
  // One element too many in bytes, though the count itself fits.
  const std::size_t over_bytes = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
  CHECK_THROWS_AS(pantr::NdArray<double>({over_bytes}), pantr::capacity_error);
  // Each extent fits alone; the product overflows only at the third.
  const std::size_t e = std::size_t{1} << 21;
  CHECK_THROWS_AS(pantr::NdArray<double>({e, e, e}), pantr::capacity_error);
}

TEST_CASE("NdArray: a zero extent gives an empty array, wherever it sits") {
  const std::size_t huge = std::numeric_limits<std::size_t>::max() / 2;
  CHECK(pantr::NdArray<double>({huge, 0}).size() == 0);
  CHECK(pantr::NdArray<double>({0, huge}).size() == 0);
}
