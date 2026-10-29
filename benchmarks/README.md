# Benchmarks

The harness behind the port's no-slower criterion (`design/port_design.md`, section 9; the plan's
DR-3 protocol). `cases.py` holds the cases, `run.py` times them in one process, and `compare.py`
runs several trees interleaved and applies the statistic. No number is quoted here: every result
lives under `results/` beside the command that produced it.

## Running

Every side is a source tree's `src` directory, given as an absolute path; `run.py` records which
`pantr` it imported and `compare.py` refuses a side that imported another one. A tree with the
extension needs `make ext` first. Measure `main` from a detached worktree of it, so that its Numba
cache stays out of any checkout you work in.

```bash
git worktree add --detach <scratch>/main <main-commit>
python benchmarks/compare.py --out-dir benchmarks/results/<date>-<what> \
    --side main=<scratch>/main/src --side branch="$PWD/src" \
    --k 10 --threads 1 8 --cpus 0-7 --filter '^S1\.'
```

Run nothing else heavy while measuring. `--cpus` pins every process to the same CPUs, BLAS and
OpenMP run on one thread, and `NUMBA_NUM_THREADS` is the thread setting.

## Results

| Directory | Command (`<scratch>` is a detached worktree of the commit named) |
|---|---|
| `results/2026-10-29-s1-baseline` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 1e0a66c>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S1\.'` |
| `results/2026-10-29-r6-abi3` | `compare.py --side main=<scratch>/main-11ab436/src --side abi3=<this tree at 1e0a66c>/src --side vs=<scratch>/branch-vs/src --k 10 --threads 1 8 --cpus 0-7 --filter '^r6\.' --r6 abi3,vs`, where `branch-vs` is a worktree of 1e0a66c built with `make ext EXT_STABLE_ABI=OFF` |
| `results/2026-10-29-r6-abi3-rerun` | the same, a fresh interleaved re-run (a flag counts only when a re-run flags it again) |
| `results/2026-10-29-s1-ac6-at-9b0f2c8` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 9b0f2c8>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S1\.'`: the quadrature/basis stage's first gate run, whose flags the following commits addressed |
| `results/2026-10-29-s1-ac6` | the same at 365ef21 with `--filter '^(S1\|r6)\.'` (the r6 cases now time the public operations on both trees) |
| `results/2026-10-29-s1-ac6-rerun` | the same at 365ef21 with `--filter '^S1\.tabulate_legendre_1d\.large'`: the fresh interleaved re-run of the one case the previous run flagged |
| `results/2026-10-29-s1-ac6-dtype-cache` | the same at 781f8c5 with `--filter '^(S1\.tolerance\|S1\.get_\|r6\.)'`: the cases on the dtype-cache path after its keys were narrowed (5a3d0c6) |
| `results/2026-10-29-s2a-ac6-at-048a709` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 048a709>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S2\.'`: the value-type stage's first gate run (AABB, AffineTransform, Bezier), whose one flag (`bezier_split.medium@t1`) the following commits addressed |
| `results/2026-10-29-s2a-ac6` | the same at 7392cde; it flags `bezier_split.medium@t1` again |
| `results/2026-10-29-s2a-ac6-rerun` | the same at 78d13b9 with `--filter '^S2\.bezier_split\.'`, after the split's chunks were widened |
| `results/2026-10-29-s2a-ac6-rerun2` | the same at 042003f with `--filter '^S2\.bezier_(split\|evaluate_lattice\|slice\|restrict)\.'`, after the contracted loops moved out of the chunk bodies: every case the change reaches, measured fresh |

Each holds `summary.md` (the statistic and range per side and case, and the flags), `summary.json`
(the same with every process median, the protocol, the commits and the machine) and `raw/` (each
process's record, with every repeat).
