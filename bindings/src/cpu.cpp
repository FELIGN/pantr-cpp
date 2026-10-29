// _pantr_cpu: which x86-64 microarchitecture levels this CPU and its OS support.
//
// The ISA ladder (pantr/_isa.py) asks it before importing a tuned rung. It is compiled at the
// toolchain baseline, includes no pantr header and links no pantr code, so importing it cannot
// fault on any CPU of the architecture.
//
// A level is supported when CPUID reports every feature of the level (x86-64 psABI, the
// microarchitecture-level table) and, for the vector levels, the OS has enabled saving the vector
// registers it needs (XCR0, read with XGETBV after checking OSXSAVE). Anything not confirmed counts
// as absent: a feature advertised by CPUID whose register state the OS does not save would fault
// at the first context switch. The same code path serves every compiler; only the two intrinsics
// differ.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#if defined(__x86_64__) || defined(_M_X64)
#define PANTR_CPU_X86_64 1
#if defined(_MSC_VER) && !defined(__clang__)
#include <immintrin.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace nb = nanobind;

namespace {

constexpr const char* kArch =
#if defined(PANTR_CPU_X86_64)
    "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    "arm64";
#else
    "other";
#endif

constexpr bool bit(std::uint32_t reg, unsigned n) { return ((reg >> n) & 1U) != 0; }

/// The levels a CPU and its OS support.
struct Features {
  bool v2 = false;  ///< CMPXCHG16B, LAHF/SAHF, POPCNT, SSE3, SSE4.1, SSE4.2, SSSE3.
  bool v3 = false;  ///< v2 and AVX, AVX2, BMI1, BMI2, F16C, FMA, LZCNT, MOVBE, OS AVX state.
  bool v4 = false;  ///< v3 and AVX512F/BW/CD/DQ/VL, OS opmask and ZMM state.
};

constexpr unsigned kOsxsaveBit = 27;  // CPUID.1:ECX

/// Decide the levels from the registers that carry them: CPUID leaf 1 ECX, leaf 7 (subleaf 0) EBX,
/// leaf 0x80000001 ECX, and XCR0 (ignored unless leaf 1 reports OSXSAVE). A pure function, so the
/// fail-closed rules can be tested on register values no test machine has.
constexpr Features decide(std::uint32_t l1ecx, std::uint32_t l7ebx, std::uint32_t e1ecx,
                          std::uint64_t xcr) {
  Features f;
  f.v2 = bit(l1ecx, 0) && bit(l1ecx, 9) && bit(l1ecx, 13) && bit(l1ecx, 19) && bit(l1ecx, 20) &&
         bit(l1ecx, 23) && bit(e1ecx, 0);
  const bool osxsave = bit(l1ecx, kOsxsaveBit);
  constexpr std::uint64_t kSseAvxState = 0x6;   // XMM, YMM upper halves
  constexpr std::uint64_t kAvx512State = 0xE0;  // opmask, ZMM_Hi256, Hi16_ZMM
  const bool os_avx = osxsave && (xcr & kSseAvxState) == kSseAvxState;
  const bool os_avx512 = os_avx && (xcr & kAvx512State) == kAvx512State;
  f.v3 = f.v2 && os_avx && bit(l1ecx, 28) /*AVX*/ && bit(l7ebx, 5) /*AVX2*/ &&
         bit(l7ebx, 3) /*BMI1*/ && bit(l7ebx, 8) /*BMI2*/ && bit(l1ecx, 29) /*F16C*/ &&
         bit(l1ecx, 12) /*FMA*/ && bit(e1ecx, 5) /*LZCNT*/ && bit(l1ecx, 22) /*MOVBE*/;
  f.v4 = f.v3 && os_avx512 && bit(l7ebx, 16) /*AVX512F*/ && bit(l7ebx, 30) /*BW*/ &&
         bit(l7ebx, 28) /*CD*/ && bit(l7ebx, 17) /*DQ*/ && bit(l7ebx, 31) /*VL*/;
  return f;
}

#if defined(PANTR_CPU_X86_64)

/// The four registers CPUID returns.
struct CpuidRegs {
  std::uint32_t eax = 0;
  std::uint32_t ebx = 0;
  std::uint32_t ecx = 0;
  std::uint32_t edx = 0;
};

CpuidRegs cpuid(std::uint32_t leaf, std::uint32_t subleaf) {
  CpuidRegs r;
#if defined(_MSC_VER) && !defined(__clang__)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  r.eax = static_cast<std::uint32_t>(regs[0]);
  r.ebx = static_cast<std::uint32_t>(regs[1]);
  r.ecx = static_cast<std::uint32_t>(regs[2]);
  r.edx = static_cast<std::uint32_t>(regs[3]);
#else
  __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
  return r;
}

/// XCR0, the vector state the OS saves. Only legal once OSXSAVE is confirmed.
std::uint64_t xcr0() {
#if defined(_MSC_VER) && !defined(__clang__)
  return static_cast<std::uint64_t>(_xgetbv(0));
#else
  std::uint32_t lo = 0;
  std::uint32_t hi = 0;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

Features read_features() {
  const std::uint32_t max_leaf = cpuid(0, 0).eax;
  const std::uint32_t max_ext = cpuid(0x80000000U, 0).eax;
  if (max_leaf < 1) {
    return Features{};
  }
  const CpuidRegs l1 = cpuid(1, 0);
  const CpuidRegs l7 = max_leaf >= 7 ? cpuid(7, 0) : CpuidRegs{};
  const CpuidRegs e1 = max_ext >= 0x80000001U ? cpuid(0x80000001U, 0) : CpuidRegs{};
  // XGETBV is legal only once OSXSAVE is confirmed.
  const std::uint64_t xcr = bit(l1.ecx, kOsxsaveBit) ? xcr0() : 0;
  return decide(l1.ecx, l7.ebx, e1.ecx, xcr);
}

#endif  // PANTR_CPU_X86_64

/// Report whether this CPU and its OS support the level `tag`.
///
/// Args:
///     tag (std::string_view): "v2", "v3" or "v4" (x86-64 microarchitecture levels).
///
/// Returns:
///     bool: True only when every feature and the OS vector state are confirmed; always false off
///     x86-64.
///
/// Raises:
///     std::invalid_argument: `tag` names no level (a misspelt rung must not read as unsupported).
bool supports(std::string_view tag) {
  if (tag != "v2" && tag != "v3" && tag != "v4") {
    throw std::invalid_argument("unknown x86-64 level '" + std::string(tag) +
                                "'; expected 'v2', 'v3' or 'v4'");
  }
#if defined(PANTR_CPU_X86_64)
  static const Features features = read_features();
  if (tag == "v2") {
    return features.v2;
  }
  return tag == "v3" ? features.v3 : features.v4;
#else
  return false;
#endif
}

}  // namespace

NB_MODULE(_pantr_cpu, m) {
  m.doc() = "Which x86-64 levels this CPU and its OS support (private: used by pantr._isa).";
  m.attr("arch") = kArch;
  m.def(
      "_levels_from_registers",
      [](std::uint32_t l1ecx, std::uint32_t l7ebx, std::uint32_t e1ecx, std::uint64_t xcr) {
        const Features f = decide(l1ecx, l7ebx, e1ecx, xcr);
        return std::tuple<bool, bool, bool>(f.v2, f.v3, f.v4);
      },
      nb::arg("leaf1_ecx"), nb::arg("leaf7_ebx"), nb::arg("ext1_ecx"), nb::arg("xcr0"),
      "Return (v2, v3, v4) as decided from these register values (for tests).");
  m.def("supports", &supports, nb::arg("tag"),
        "Return True if the CPU and its OS support the x86-64 level `tag` ('v2', 'v3', 'v4').");
}
