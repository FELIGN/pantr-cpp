# Benchmarks

The harness behind the port's no-slower criterion (`design/port_design.md`, section 9; the plan's
DR-3 protocol). `cases.py` holds the cases, `run.py` times them in one process, and `compare.py`
runs several trees interleaved and applies the statistic. No number is quoted here: every result
lives under `results/` beside the command that produced it.

## Running

Every side is a source tree's `src` directory, given as an absolute path; `run.py` records which
`pantr` it imported and `compare.py` refuses a side that imported another one. A tree with the
extension needs `make ext` first. Measure `main`, which still runs Numba, from a detached worktree
of it, so that its Numba cache stays out of any checkout you work in.

```bash
git worktree add --detach <scratch>/main <main-commit>
python benchmarks/compare.py --out-dir benchmarks/results/<date>-<what> \
    --side main=<scratch>/main/src --side branch="$PWD/src" \
    --k 10 --threads 1 8 --cpus 0-7 --filter '^S1\.'
```

Run nothing else heavy while measuring. `--cpus` pins every process to the same CPUs, BLAS and
OpenMP run on one thread, and the thread setting goes to both sides in two variables, each side
reading one: `NUMBA_NUM_THREADS` for `main` (Numba's pool) and `PANTR_NUM_THREADS` for the branch
(the C++ pool's capacity); `run.py` then sets it with `pantr.set_num_threads` on both.

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
| `results/2026-10-29-s2b-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 3396699>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S2\.(bezier_reduce\|bezier_minimize\|interpolate\|fit_\|find_)'`: the reduction, fitting and root-finding cases (C2.3, C2.4) |
| `results/2026-10-29-s2b-ac6-at-d7cf952` | the same at d7cf952, after the Python review round changed the wrappers; the shared machine was heavily loaded by other users during this run, so both sides read slower than in the run above |
| `results/2026-10-30-s4a-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 4fda1ac>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S4\.'`: the B-spline space cases of C4.1 (construction, 1D, 2D, 3D and lattice tabulation); no case flagged, so no re-run was needed. Only `summary.md` and `summary.json` are kept: the latter holds every process median, from which the statistic is recomputed |
| `results/2026-10-31-s4b-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at f717a88>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S4\.(bspline_(construct\|evaluate\|insert\|remove\|elevate\|to_beziers\|to_periodic)\|spanwise)'`: the `Bspline` and spanwise-extraction cases of C4.2 and C5.2; only the construction cases are flagged (`summary.md` and `summary.json` only) |
| `results/2026-10-31-s4b-construct-rerun` | the confirming re-run of the flagged construction cases, `--filter '^S4\.bspline_construct'`, at c6b2921 (one binding call per construction): still flagged. Construction minus a NumPy copy of the same control points (made read-only), timed in one process, exceeds main's construction by at most 0.12 µs at `tiny`, 0.3 µs at `medium` and 1.4 to 3.2 µs (under 1% of the construction) at `large`. Whether the plan's exemption of the constructor's copy term covers the fixed cost of that copy at `tiny` is a ruling still to make |
| `results/2026-10-31-s4c-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 68e308b>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S4\.bspline_(find_roots\|multiply)'`: the root-finding and product cases of the S4 C++ port; `bspline_multiply.large` is flagged at both thread counts (`summary.md` and `summary.json` only) |
| `results/2026-10-31-s4c-ac6-rerun` | the confirming re-run, the same at 68e308b with `--filter '^S4\.bspline_multiply\.large'`: flagged again. The product's least-squares projections were a dense SVD, cubic in the number of intervals |
| `results/2026-10-31-s4c-ac6-fix` | the same at c77a658 with `--filter '^S4\.bspline_multiply'`, after the projections moved to a banded QR (7f91f42): no case flagged, so no re-run was needed (`summary.md` and `summary.json` only) |
| `results/2026-11-01-s5-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 3401390>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^(S5\.\|S4\.spanwise)'`: the fitting cases of C5.1 (interpolation, tensor and scattered fits, L2 projection, quasi-interpolation) and the spanwise extraction cases; no case flagged (`summary.md` and `summary.json` only). A sweep outside the harness then found interpolation slower than `main` beyond the `large` size (2000 and 4000 intervals), which the next run's tree addresses |
| `results/2026-11-01-s5-ac6-band-lu` | the same at 521509d with `--filter '^S5\.(interpolate\|l2_project)'`, after square systems moved to a banded LU: the cases that change reaches, measured fresh; no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-01-s6-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 6d38e22>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S6\.'`: the grid and point-location cases (grid location, hierarchical refinement, BVH query, `Bspline.locate`, `partition_grid`); `partition_grid_block.large` is flagged at both thread counts (`summary.md` and `summary.json` only) |
| `results/2026-11-01-s6-ac6-block-fix` | the same at 9d487e8 with `--filter '^S6\.partition_grid'`, after the block backend's per-cell divisions moved to per-axis tables: no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-01-s6-ac6-final` | the same at 41a0fc0 with `--filter '^S6\.'`: every grid and point-location case, fresh, after the stage's review fixes; no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-01-s6-locate-ac6` | the same at 3801e98 with `--filter '^S6\.'`, after `Bspline.locate` moved to C++ and int64 arrays to `numpy.int64`: every grid and point-location case; no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-02-s6-locate-ac6-final` | the same at 5523ce6 with `--filter '^S6\.bspline_locate'`, after the loop review changed `locate`'s work estimate: no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-02-s6-graph-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 35d8aa6>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S6\.(coupling_graph\|partition_graph)'`: the coupling-graph and graph-partition cases (C6.3, C6.4); `partition_graph_metis.large` is flagged at both thread counts, the extraction of the active subgraph sorting every row (`summary.md` and `summary.json` only) |
| `results/2026-11-02-s6-graph-ac6-final` | the same at ef3a483, after rows are sorted only when out of order and the dense eigensolver route finds one eigenvector: every case of the previous run, fresh; no case flagged (`summary.md` and `summary.json` only) |
| `results/2026-11-03-s6-thb-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 2bfa0d9>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S6\.(thb_\|quasi_interpolate_thb\|multilevel_extraction\|coupling_graph_thb\|build_local_thb\|compute_halo\|dof_owner)'`: the THB space, THB spline, THB quasi-interpolation, multilevel extraction and local-space cases (C6.3); no case flagged, so no re-run was needed (`summary.md` and `summary.json` only) |
| `results/2026-11-03-s7-ac6` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at e15c457>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S7\.'`: the CAD and multipatch cases (C7.1: Coons patch, revolution, join, interface detection); `create_coons_surface.large` is flagged at both thread counts (`summary.md` and `summary.json` only) |
| `results/2026-11-03-s7-ac6-rerun` | the confirming re-run, the same at e15c457 with `--filter '^S7\.create_coons_surface\.large'`: flagged again. Knot insertion moved each non-leading axis to the front and back, two passes and one allocation of the refined net's size, which the patch's compatibility step pays on nets of millions of points |
| `results/2026-11-03-s7-ac6-fix` | the same at 1a209e7 with `--filter '^(S7\.\|S4\.bspline_insert)'`, after knot insertion works along the axis: no S7 case flagged; `S4.bspline_insert_knots.large` is flagged at both thread counts (`summary.md` and `summary.json` only) |
| `results/2026-11-03-s7-ac6-final` | the same at 7800ec9, after the Oslo bands of a first-axis insertion are again formed per row: every case of the previous run, fresh; no case flagged, so the insertion flag was not confirmed (`summary.md` and `summary.json` only) |
| `results/2026-11-03-s7-ac6-tip` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 42d2cef>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S7\.'`: every CAD and multipatch case at the stage's last tree, after the cad binding stopped releasing the GIL; no case flagged. The shared machine was heavily loaded by other users during this run, so both sides read slower than in the runs above (`summary.md` and `summary.json` only) |

| `results/2026-11-03-s8-ac6-tip` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 3696bea>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^S8\.'`: every distributed case on one rank through a duck-typed communicator; `from_dolfinx.large` is flagged at both thread counts (the binding sent a copy of the cell ids), which 9181cf3 addressed |
| `results/2026-11-03-s8-ac6-confirm` | the same command at 9181cf3: the confirming re-run; no case flagged. The shared machine was loaded by other users during both runs (`summary.md` and `summary.json` only) |
| `results/2026-11-04-p9-ac6-final` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 4b7c8fb>/src --k 10 --threads 1 8 --cpus 0-7`: every case of `cases.py`, after Numba and threadpoolctl were removed (the branch's last change to `src` and the extension is 20a7020); flags `S2.bezier_construct.tiny@t8` and the `S4.bspline_construct` cases at every size and both thread counts (`summary.md` and `summary.json` only) |
| `results/2026-11-04-p9-ac6-confirm` | the confirming re-run, the same at 4b7c8fb with `--filter '^(S2\.bezier_construct\|S4\.bspline_construct)\.'`: `bezier_construct` is not flagged again; `bspline_construct` is, at `large` at both thread counts and at `medium` and `tiny` with one thread, the construction flag first recorded in `2026-10-31-s4b-ac6` (`summary.md` and `summary.json` only) |
| `results/2026-11-06-p10b-ac3` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 4595705>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^(S2\.find_(monotone_root\|roots)\|S6\.(thb_\|tpg_\|partition_grid\|hgrid\|multilevel\|bvh)\|S7\.detect_interfaces)'`: the cases touching monotone and Bezier root finding, THB construction, refinement and prolongation, grid uniformity and partition, and multipatch detection, after the P10 fix stage's second part; no case flagged, so no confirming re-run (`summary.md` and `summary.json` only) |
| `results/2026-11-09-p10c-ac3` | `compare.py --side main=<scratch>/main-11ab436/src --side branch=<this tree at 0ed62d7>/src --k 10 --threads 1 8 --cpus 0-7 --filter '^(S2\.bezier_(split\|slice\|restrict\|multiply)\|S5\.\|S6\.(bvh\|thb_\|tpg_\|hgrid))'`: the Bezier operations and fits whose bindings now release the GIL, and the lazily built grid, BVH and THB structures, after the lazy blocks gained their fork registry; no case flagged, so no confirming re-run (`summary.md` and `summary.json` only) |

Each holds `summary.md` (the statistic and range per side and case, and the flags), `summary.json`
(the same with every process median, the protocol, the commits and the machine) and `raw/` (each
process's record, with every repeat).
