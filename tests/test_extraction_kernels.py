"""Correctness and validation tests for the Kronecker application behind the extraction.

The element operators of :class:`~pantr.bspline.SpanwiseElementExtraction` are contractions of
``M = kron(M_0, ..., M_{d-1})`` with a vector or on both sides of a matrix. The compiled library
runs them for any number of directions on factors of any shape; ``pantr._testing.kron_apply``
reaches that application on factors that need not come from a space, and the extraction's own
methods reach it with the messages a caller sees.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr._testing import kron_apply, required_scratch_size
from pantr.bspline import BsplineSpace, BsplineSpace1D, SpanwiseElementExtraction
from pantr.bspline.spanwise_element_extraction import OpKind

RNG = np.random.default_rng(12345)


def _operation_shapes(
    in_shape: tuple[int, ...], out_shape: tuple[int, ...], op_kind: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Shapes of the operand and of the result of an operation, by the definition of each."""
    n_in, n_out = int(np.prod(in_shape)), int(np.prod(out_shape))
    return {
        "apply": ((n_in,), (n_out,)),
        "apply_T": ((n_out,), (n_in,)),
        "MT_K_M": ((n_out, n_out), (n_in, n_in)),
        "M_K_MT": ((n_in, n_in), (n_out, n_out)),
    }[op_kind]


def _make_ops(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    identity: tuple[bool, ...],
    dtype: np.dtype[Any],
) -> list[npt.NDArray[Any]]:
    """Build per-direction 2D operators, using identity for flagged directions."""
    ops: list[npt.NDArray[Any]] = []
    for n_in, n_out, is_id in zip(in_shape, out_shape, identity, strict=True):
        if is_id:
            assert n_in == n_out, "identity direction requires n_in == n_out"
            ops.append(np.eye(n_in, dtype=dtype))
        else:
            ops.append(RNG.standard_normal((n_out, n_in)).astype(dtype))
    return ops


def _full_kron(ops: list[npt.NDArray[Any]]) -> npt.NDArray[Any]:
    """Assemble the full Kronecker product from per-direction operators."""
    result = ops[0]
    for M in ops[1:]:
        result = np.kron(result, M)
    return result


def _reference(
    ops: list[npt.NDArray[Any]],
    operand: npt.NDArray[Any],
    op_kind: str,
) -> npt.NDArray[Any]:
    """Naive reference using the materialized Kronecker product.

    Computes in float64 to avoid float32 overflow/subnormal warnings; the
    result is cast back to the original dtype for comparison.
    """
    M = _full_kron([op.astype(np.float64) for op in ops])
    op64 = operand.astype(np.float64)
    if op_kind == "apply":
        result = M @ op64
    elif op_kind == "apply_T":
        result = M.T @ op64
    elif op_kind == "MT_K_M":
        result = M.T @ op64 @ M
    elif op_kind == "M_K_MT":
        result = M @ op64 @ M.T
    else:
        raise ValueError(f"unknown op_kind {op_kind}")
    return result.astype(operand.dtype)


def _identity_patterns(d: int) -> list[tuple[bool, ...]]:
    """Enumerate all 2^d identity-flag combinations for dimension d."""
    return list(itertools.product([False, True], repeat=d))


# -- Shape configurations ------------------------------------------------------

# Triples (d, in_shape, out_shape). For identity directions n_in == n_out is
# required, so we use configurations where at least the "square" axes have
# matching sizes; the test harness picks identity flags accordingly.
SHAPE_CONFIGS_SQUARE: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = [
    (1, (3,), (3,)),
    (1, (5,), (5,)),
    (2, (3, 4), (3, 4)),
    (2, (4, 2), (4, 2)),
    (3, (3, 4, 2), (3, 4, 2)),
    (3, (2, 3, 4), (2, 3, 4)),
]

SHAPE_CONFIGS_NONSQUARE: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = [
    (1, (3,), (5,)),
    (1, (5,), (2,)),
    (2, (2, 3), (4, 2)),
    (2, (3, 4), (2, 5)),
    (3, (2, 3, 4), (3, 2, 5)),
    (3, (4, 2, 3), (2, 4, 2)),
]


# -- Correctness tests for square operators (identity patterns) ---------------


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE)
@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_square_factors_all_identity_patterns(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    op_kind: OpKind,
    dtype: np.dtype[Any],
) -> None:
    """Each operation matches the naive Kronecker reference for every identity pattern."""
    tol = 5e-5 if dtype == np.float32 else 1e-10
    for identity in _identity_patterns(d):
        ops = _make_ops(in_shape, out_shape, identity, np.dtype(dtype))
        in_op_shape, _out_op_shape = _operation_shapes(in_shape, out_shape, op_kind)
        operand = RNG.standard_normal(in_op_shape).astype(dtype)
        expected = _reference(ops, operand, op_kind)

        out = kron_apply(ops, identity, operand, op_kind)
        np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_NONSQUARE)
@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_nonsquare_factors(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    op_kind: OpKind,
    dtype: np.dtype[Any],
) -> None:
    """Non-square per-direction operators produce correct results (no identity)."""
    tol = 5e-5 if dtype == np.float32 else 1e-10
    identity = tuple([False] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(dtype))
    in_op_shape, _ = _operation_shapes(in_shape, out_shape, op_kind)
    operand = RNG.standard_normal(in_op_shape).astype(dtype)
    expected = _reference(ops, operand, op_kind)

    out = kron_apply(ops, identity, operand, op_kind)
    np.testing.assert_allclose(out, expected, atol=tol, rtol=tol)


# -- All-identity aliasing tests ----------------------------------------------


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE[:3])
def test_all_identity_apply_aliasing(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> None:
    """``apply`` / ``apply_T`` are copy-through in the all-identity case; out=v is legal."""
    identity = tuple([True] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(np.float64))
    for op_kind in ("apply", "apply_T"):
        in_shape_op, _ = _operation_shapes(in_shape, out_shape, op_kind)
        v = RNG.standard_normal(in_shape_op).astype(np.float64)
        v_copy = v.copy()
        # Pass v itself as `out`: in the all-identity case this should be a
        # self-copy (no-op for the values), and v remains unchanged.
        out = kron_apply(ops, identity, v, op_kind, out=v)
        assert out is v
        np.testing.assert_array_equal(v, v_copy)


@pytest.mark.parametrize(("d", "in_shape", "out_shape"), SHAPE_CONFIGS_SQUARE[:3])
def test_all_identity_bilateral_aliasing(
    d: int,
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
) -> None:
    """``MT_K_M`` / ``M_K_MT`` are copy-through in the all-identity case; out=K is legal."""
    identity = tuple([True] * d)
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(np.float64))
    for op_kind in ("MT_K_M", "M_K_MT"):
        in_shape_op, _ = _operation_shapes(in_shape, out_shape, op_kind)
        K = RNG.standard_normal(in_shape_op).astype(np.float64)
        K_copy = K.copy()
        out = kron_apply(ops, identity, K, op_kind, out=K)
        assert out is K
        np.testing.assert_array_equal(K, K_copy)


# -- Any number of directions ---------------------------------------------------


@pytest.mark.parametrize("d", [4, 5])
@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
def test_more_than_three_directions(d: int, op_kind: OpKind) -> None:
    """The application is not specialised by dimension: ``d > 3`` matches the dense reference.

    Main raised ``NotImplementedError`` for ``d`` outside ``{1, 2, 3}``.
    """
    in_shape = tuple(2 + (k % 2) for k in range(d))
    out_shape = tuple(3 - (k % 2) for k in range(d))
    identity = tuple(k == 1 for k in range(d))
    in_shape = tuple(
        o if flag else i for i, o, flag in zip(in_shape, out_shape, identity, strict=True)
    )
    ops = _make_ops(in_shape, out_shape, identity, np.dtype(np.float64))
    in_op_shape, _ = _operation_shapes(in_shape, out_shape, op_kind)
    operand = RNG.standard_normal(in_op_shape)
    out = kron_apply(ops, identity, operand, op_kind)
    np.testing.assert_allclose(out, _reference(ops, operand, op_kind), atol=1e-10, rtol=1e-10)


def test_no_direction_raises() -> None:
    """A Kronecker product of no factors is refused (the compiled library says so)."""
    with pytest.raises(ValueError, match="At least one direction is required"):
        kron_apply([], [], np.ones(1), "apply")


# -- Validation errors --------------------------------------------------------


def test_unknown_op_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown op_kind"):
        required_scratch_size((2,), (2,), "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown op_kind"):
        kron_apply([np.eye(2)], [False], np.ones(2), "nope")  # type: ignore[arg-type]


def test_required_scratch_size_mismatched_shapes_raises() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        required_scratch_size((2, 3), (2,), "apply")


def _extraction() -> SpanwiseElementExtraction:
    """A 2D extraction on 3x3 operators: 6 x 6 elements, operand sizes 9 and 9."""
    sp1 = BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 6], 2)
    return SpanwiseElementExtraction(BsplineSpace([sp1, sp1]), "bezier")


def test_operand_wrong_shape_raises() -> None:
    ext = _extraction()
    bad = RNG.standard_normal(20).astype(np.float64)
    with pytest.raises(ValueError, match="expected shape"):
        ext.apply(bad, 7)


def test_operand_wrong_dtype_raises() -> None:
    ext = _extraction()
    bad = RNG.standard_normal(9).astype(np.float32)
    with pytest.raises(ValueError, match="expected dtype"):
        ext.apply(bad, 7)


def test_out_wrong_shape_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    bad_out = np.empty(8, dtype=np.float64)
    with pytest.raises(ValueError, match="expected shape"):
        ext.apply(v, 7, out=bad_out)


def test_out_not_writable_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    bad_out = np.empty(9, dtype=np.float64)
    bad_out.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        ext.apply(v, 7, out=bad_out)


def test_scratch_too_small_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    bad_scratch = np.empty(1, dtype=np.float64)
    with pytest.raises(ValueError, match="smaller than required"):
        ext.apply(v, 7, scratch=bad_scratch)


def test_scratch_wrong_ndim_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    with pytest.raises(ValueError, match="1D"):
        ext.apply(v, 7, scratch=np.empty((3, 3), dtype=np.float64))


def test_scratch_wrong_dtype_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    with pytest.raises(ValueError, match="expected dtype"):
        ext.apply(v, 7, scratch=np.empty(16, dtype=np.float32))


def test_scratch_not_writable_raises() -> None:
    ext = _extraction()
    v = RNG.standard_normal(9).astype(np.float64)
    scratch = np.empty(64, dtype=np.float64)
    scratch.flags.writeable = False
    with pytest.raises(ValueError, match="not writeable"):
        ext.apply(v, 7, scratch=scratch)


def test_identity_length_mismatch_raises() -> None:
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12).astype(np.float64)
    with pytest.raises(ValueError, match="length"):
        kron_apply(ops, (False,), v, "apply")


# -- Scratch size sanity -------------------------------------------------------


def test_required_scratch_size_apply_d1_is_zero() -> None:
    assert required_scratch_size((5,), (3,), "apply") == 0
    assert required_scratch_size((5,), (3,), "apply_T") == 0


def test_required_scratch_size_d2_matches_the_intermediate() -> None:
    n_in, n_out = (3, 4), (5, 6)
    # The only intermediate of ``apply`` is (n_out[0], n_in[1]); twice that holds the two halves.
    assert required_scratch_size(n_in, n_out, "apply") == 2 * (n_out[0] * n_in[1])
    # apply_T: starts at n_out, ends at n_in -- intermediate size n_in[0] * n_out[1].
    assert required_scratch_size(n_in, n_out, "apply_T") == 2 * (n_in[0] * n_out[1])


def test_required_scratch_size_bilateral_d1_matches_the_intermediate() -> None:
    # Stage 0 intermediate (only intermediate) is n_in * n_out.
    assert required_scratch_size((3,), (5,), "MT_K_M") == 2 * (5 * 3)
    assert required_scratch_size((3,), (5,), "M_K_MT") == 2 * (3 * 5)


@pytest.mark.parametrize("op_kind", ["apply", "apply_T", "MT_K_M", "M_K_MT"])
def test_exactly_the_required_scratch_is_enough(op_kind: OpKind) -> None:
    """A scratch of exactly ``required_scratch_size`` entries runs every identity pattern."""
    in_shape, out_shape = (2, 3, 2), (3, 2, 4)
    for identity in itertools.product([False, True], repeat=3):
        shape_in = tuple(
            o if f else i for i, o, f in zip(in_shape, out_shape, identity, strict=True)
        )
        ops = _make_ops(shape_in, out_shape, identity, np.dtype(np.float64))
        need = required_scratch_size(shape_in, out_shape, op_kind)
        in_op_shape, _ = _operation_shapes(shape_in, out_shape, op_kind)
        operand = RNG.standard_normal(in_op_shape)
        out = kron_apply(ops, identity, operand, op_kind, scratch=np.empty(need))
        np.testing.assert_allclose(out, _reference(ops, operand, op_kind), atol=1e-10, rtol=1e-10)


def test_scratch_overlapping_the_operand_or_out_raises() -> None:
    """Scratch that shares memory with the operand or the result is refused.

    Main computed on regardless, and returned values corrupted by its own intermediates.
    """
    ops = _make_ops((3, 4), (3, 4), (False, False), np.dtype(np.float64))
    v = RNG.standard_normal(12)
    buf = np.concatenate([v, np.zeros(64)])
    with pytest.raises(ValueError, match="scratch must not overlap"):
        kron_apply(ops, (False, False), buf[:12], "apply", scratch=buf)
    big = np.zeros(64)
    with pytest.raises(ValueError, match="scratch must not overlap"):
        kron_apply(ops, (False, False), v, "apply", out=big[:12], scratch=big)


# An identity branch copies over a single extent, which equals applying the operator only
# when it is square, so an identity-flagged factor that is not square is refused.


def test_identity_flagged_nonsquare_operator_raises() -> None:
    non_square = np.eye(3, 2, dtype=np.float64)
    v = RNG.standard_normal(2).astype(np.float64)
    with pytest.raises(ValueError, match="must be square"):
        kron_apply([non_square], [True], v, "apply")


def test_nonsquare_operator_not_flagged_identity_is_accepted() -> None:
    # The refusal keys on the flag: the same operator unflagged is an ordinary rectangular
    # operator and must still apply.
    non_square = np.eye(3, 2, dtype=np.float64)
    v = RNG.standard_normal(2).astype(np.float64)
    out = kron_apply([non_square], [False], v, "apply")
    assert out.shape == (3,)


def test_identity_flagged_square_operator_is_accepted() -> None:
    square = np.eye(3, dtype=np.float64)
    v = RNG.standard_normal(3).astype(np.float64)
    out = kron_apply([square], [True], v, "apply")
    assert out.shape == (3,)
