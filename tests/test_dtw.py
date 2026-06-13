"""Tests for sbd.dtw module."""

import pytest
import torch

from sbd.dtw.distance import (
    DistanceFunction,
    cosine,
    dot_product,
    euclidean,
    get_distance_fn,
    manhattan,
)
import warnings

from sbd.dtw.dtw import (
    _compute_accumulated_cost_matrix_cpu,
    _compute_accumulated_cost_matrix_gpu,
    _compute_cost_matrix,
    _compute_optimal_warping_path,
    dtw,
)
from sbd.dtw.utils import negate_fn, normalize_batch_of_tensors
from sbd.utils.gpu_check import is_cuda_available

# ─── utils.py ─────────────────────────────────────────────────────────────────


class TestNormalizeBatchOfTensors:
    def test_output_has_unit_norm(self):
        t = torch.tensor([[3.0, 4.0], [1.0, 0.0]])
        result = normalize_batch_of_tensors(t)
        norms = result.norm(dim=1)
        assert torch.allclose(norms, torch.ones(2))

    def test_known_values(self):
        t = torch.tensor([[3.0, 4.0]])
        result = normalize_batch_of_tensors(t)
        assert torch.allclose(result, torch.tensor([[0.6, 0.8]]))

    def test_zero_vector_returns_zeros(self):
        t = torch.tensor([[0.0, 0.0]])
        result = normalize_batch_of_tensors(t)
        assert torch.allclose(result, torch.zeros(1, 2))

    def test_already_unit_vector_unchanged(self):
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = normalize_batch_of_tensors(t)
        assert torch.allclose(result, t)

    def test_rejects_1d_input(self):
        with pytest.raises(AssertionError):
            normalize_batch_of_tensors(torch.tensor([1.0, 2.0]))

    def test_rejects_3d_input(self):
        with pytest.raises(AssertionError):
            normalize_batch_of_tensors(torch.randn(2, 3, 4))


class TestNegateFunction:
    def test_negates_scalar_result(self):
        result = negate_fn(lambda x, y: x + y, 3, 4)
        assert result == -7

    def test_negates_tensor_result(self):
        t = torch.tensor(5.0)
        result = negate_fn(lambda x: x, t)
        assert torch.allclose(result, torch.tensor(-5.0))


# ─── distance.py ──────────────────────────────────────────────────────────────


class TestDotProduct:
    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        assert dot_product(a, b).shape == (3, 5)

    def test_orthogonal_unit_vectors(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = dot_product(a, a)
        assert torch.allclose(result, torch.eye(2))

    def test_known_value(self):
        a = torch.tensor([[1.0, 2.0]])
        b = torch.tensor([[3.0, 4.0]])
        # 1*3 + 2*4 = 11
        assert torch.allclose(dot_product(a, b), torch.tensor([[11.0]]))


class TestCosine:
    def test_identical_unit_vectors(self):
        a = torch.tensor([[1.0, 0.0]])
        assert torch.allclose(cosine(a, a), torch.tensor([[1.0]]))

    def test_orthogonal_vectors(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        assert torch.allclose(cosine(a, b), torch.tensor([[0.0]]))

    def test_scale_invariant(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[5.0, 0.0]])
        assert torch.allclose(cosine(a, b), torch.tensor([[1.0]]))

    def test_opposite_vectors(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[-1.0, 0.0]])
        assert torch.allclose(cosine(a, b), torch.tensor([[-1.0]]))

    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        assert cosine(a, b).shape == (3, 5)


class TestManhattan:
    def test_known_distance(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        # |1-0| + |0-1| = 2.0
        assert torch.allclose(manhattan(a, b), torch.tensor([[2.0]]))

    def test_self_distance_is_zero(self):
        a = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(manhattan(a, a), torch.tensor([[0.0]]))

    def test_non_negative(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        assert (manhattan(a, b) >= 0).all()


class TestEuclidean:
    def test_known_distance(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        # sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        assert torch.allclose(euclidean(a, b), torch.tensor([[2.0**0.5]]))

    def test_self_distance_is_zero(self):
        a = torch.tensor([[3.0, 5.0]])
        assert torch.allclose(euclidean(a, a), torch.tensor([[0.0]]))

    def test_non_negative(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        assert (euclidean(a, b) >= 0).all()

    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        assert euclidean(a, b).shape == (3, 5)


class TestGetDistanceFn:
    @pytest.mark.parametrize("fn_enum", list(DistanceFunction))
    def test_returns_callable(self, fn_enum):
        fn = get_distance_fn(fn_enum)
        assert callable(fn)

    def test_cosine_fn_negates_similarity(self):
        a = torch.tensor([[1.0, 0.0]])
        fn = get_distance_fn(DistanceFunction.COSINE)
        # cosine(a, a) = 1.0, negated → -1.0
        assert torch.allclose(fn(a, a), torch.tensor([[-1.0]]))

    def test_dot_product_fn_negates(self):
        a = torch.tensor([[1.0, 0.0]])
        fn = get_distance_fn(DistanceFunction.DOT_PRODUCT)
        # dot([1,0],[1,0]) = 1.0, negated → -1.0
        assert torch.allclose(fn(a, a), torch.tensor([[-1.0]]))

    def test_euclidean_fn_non_negative(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        fn = get_distance_fn(DistanceFunction.EUCLIDEAN)
        assert (fn(a, b) >= 0).all()

    def test_manhattan_fn_non_negative(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        fn = get_distance_fn(DistanceFunction.MANHATTAN)
        assert (fn(a, b) >= 0).all()


# ─── dtw.py (private helpers) ─────────────────────────────────────────────────


class TestComputeCostMatrix:
    def test_output_shape(self):
        a = torch.randn(3, 4)
        b = torch.randn(5, 4)
        cm = _compute_cost_matrix(a, b, DistanceFunction.EUCLIDEAN)
        assert cm.shape == (3, 5)

    def test_euclidean_self_diagonal_is_zero(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        cm = _compute_cost_matrix(a, a, DistanceFunction.EUCLIDEAN)
        assert torch.allclose(cm.diagonal(), torch.zeros(2))

    def test_cosine_self_diagonal_is_minus_one(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        cm = _compute_cost_matrix(a, a, DistanceFunction.COSINE)
        # Cosine similarity negated: cos(a[i], a[i]) = 1 → distance = -1
        assert torch.allclose(cm.diagonal(), torch.full((2,), -1.0))

    def test_symmetric_for_euclidean(self):
        a = torch.randn(4, 3)
        cm = _compute_cost_matrix(a, a, DistanceFunction.EUCLIDEAN)
        assert torch.allclose(cm, cm.T)

    def test_known_realistic_example_fake_2d(self):
        # Example from: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
        a = torch.Tensor([[1, 1], [3, 1], [9, 1], [2, 1], [1, 1]])
        b = torch.Tensor([[2, 1], [0, 1], [0, 1], [8, 1], [7, 1], [2, 1]])
        expected = torch.tensor(
            [
                [1, 1, 1, 7, 6, 1],
                [1, 3, 3, 5, 4, 1],
                [7, 9, 9, 1, 2, 7],
                [0, 2, 2, 6, 5, 0],
                [1, 1, 1, 7, 6, 1],
            ],
            dtype=torch.float32,
        )
        cm = _compute_cost_matrix(a, b, DistanceFunction.EUCLIDEAN)
        assert torch.allclose(cm, expected)


class TestComputeAccumulatedCostMatrix:
    def test_known_2x2(self):
        cm = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        # acm[0,0]=1, acm[0,1]=3, acm[1,0]=4, acm[1,1]=4+min(1,3,4)=5
        expected = torch.tensor([[1.0, 3.0], [4.0, 5.0]])
        assert torch.allclose(acm, expected)

    def test_known_realistic_example(self):
        # Example from: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
        cm = torch.tensor(
            [
                [1, 1, 1, 7, 6, 1],
                [1, 3, 3, 5, 4, 1],
                [7, 9, 9, 1, 2, 7],
                [0, 2, 2, 6, 5, 0],
                [1, 1, 1, 7, 6, 1],
            ],
            dtype=torch.float32,
        )
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        expected = torch.tensor(
            [
                [1, 2, 3, 10, 16, 17],
                [2, 4, 5, 8, 12, 13],
                [9, 11, 13, 6, 8, 15],
                [9, 11, 13, 12, 11, 8],
                [10, 10, 11, 18, 17, 9],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(acm, expected)

    def test_origin_matches_cost_matrix(self):
        cm = torch.rand(4, 5).abs()
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        assert acm[0, 0] == cm[0, 0]

    def test_first_row_is_cumsum(self):
        cm = torch.tensor([[1.0, 2.0, 3.0]])
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        assert torch.allclose(acm[0], torch.tensor([1.0, 3.0, 6.0]))

    def test_first_col_is_cumsum(self):
        cm = torch.tensor([[1.0], [2.0], [3.0]])
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        assert torch.allclose(acm[:, 0], torch.tensor([1.0, 3.0, 6.0]))

    def test_output_shape_preserved(self):
        cm = torch.rand(5, 7)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        assert acm.shape == cm.shape


class TestComputeOptimalWarpingPath:
    def test_starts_at_origin(self):
        cm = torch.rand(3, 4)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        path = _compute_optimal_warping_path(acm)
        assert path[0, 0].item() == 0 and path[0, 1].item() == 0

    def test_ends_at_destination(self):
        n, m = 3, 4
        cm = torch.rand(n, m)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        path = _compute_optimal_warping_path(acm)
        assert path[-1, 0].item() == n - 1 and path[-1, 1].item() == m - 1

    def test_diagonal_path_for_identical_sequences(self):
        a = torch.eye(3)
        cm = _compute_cost_matrix(a, a, DistanceFunction.EUCLIDEAN)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        path = _compute_optimal_warping_path(acm)
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert torch.allclose(path, expected)

    def test_path_is_monotonically_non_decreasing(self):
        cm = torch.rand(4, 5)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        path = _compute_optimal_warping_path(acm)
        diffs = path[1:] - path[:-1]
        assert (diffs >= 0).all(), "Path must only move forward (non-decreasing in both dims)"

    def test_each_step_advances_at_least_one_dim(self):
        cm = torch.rand(4, 5)
        acm = _compute_accumulated_cost_matrix_cpu(cm)
        path = _compute_optimal_warping_path(acm)
        step_sizes = (path[1:] - path[:-1]).sum(dim=1)
        assert (step_sizes >= 1).all(), "Each step must advance in at least one dimension"

    def test_known_2x2_path(self):
        # With cm=[[1,2],[3,4]], acm=[[1,3],[4,5]], optimal path must be (0,0)→(1,1)
        acm = torch.tensor([[1.0, 3.0], [4.0, 5.0]])
        path = _compute_optimal_warping_path(acm)
        expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        assert torch.allclose(path, expected)

    def test_known_realistic_example(self):
        # Example from: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S2_DTWbasic.html
        acm = torch.tensor(
            [
                [1, 2, 3, 10, 16, 17],
                [2, 4, 5, 8, 12, 13],
                [9, 11, 13, 6, 8, 15],
                [9, 11, 13, 12, 11, 8],
                [10, 10, 11, 18, 17, 9],
            ],
            dtype=torch.float32,
        )
        path = _compute_optimal_warping_path(acm)
        expected = torch.tensor([[0, 0], [0, 1], [1, 2], [2, 3], [2, 4], [3, 5], [4, 5]], dtype=torch.float32)
        assert torch.allclose(path, expected)


# ─── _compute_accumulated_cost_matrix_gpu ─────────────────────────────────────
# The GPU implementation uses the skewed anti-diagonal layout. It also works on
# CPU tensors, so all correctness tests run regardless of GPU availability.


class TestComputeAccumulatedCostMatrixGpu:
    """Correctness: GPU implementation must match CPU implementation exactly."""

    @pytest.mark.parametrize("shape", [(1, 1), (1, 5), (5, 1), (2, 2), (4, 7), (10, 10), (50, 60)])
    def test_matches_cpu_reference(self, shape):
        torch.manual_seed(0)
        cm = torch.rand(*shape)
        ref = _compute_accumulated_cost_matrix_cpu(cm)
        got = _compute_accumulated_cost_matrix_gpu(cm)
        assert torch.allclose(ref, got, atol=1e-5), f"Mismatch for shape {shape}"

    def test_known_2x2(self):
        cm = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        acm = _compute_accumulated_cost_matrix_gpu(cm)
        expected = torch.tensor([[1.0, 3.0], [4.0, 5.0]])
        assert torch.allclose(acm, expected)

    def test_first_row_is_cumsum(self):
        cm = torch.tensor([[1.0, 2.0, 3.0]])
        acm = _compute_accumulated_cost_matrix_gpu(cm)
        assert torch.allclose(acm[0], torch.tensor([1.0, 3.0, 6.0]))

    def test_first_col_is_cumsum(self):
        cm = torch.tensor([[1.0], [2.0], [3.0]])
        acm = _compute_accumulated_cost_matrix_gpu(cm)
        assert torch.allclose(acm[:, 0], torch.tensor([1.0, 3.0, 6.0]))

    def test_output_shape_preserved(self):
        cm = torch.rand(5, 7)
        assert _compute_accumulated_cost_matrix_gpu(cm).shape == cm.shape

    def test_origin_matches_cost_matrix(self):
        cm = torch.rand(4, 5)
        acm = _compute_accumulated_cost_matrix_gpu(cm)
        assert acm[0, 0] == cm[0, 0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_matches_cpu_reference_on_cuda(self):
        torch.manual_seed(0)
        cm = torch.rand(20, 20)
        ref = _compute_accumulated_cost_matrix_cpu(cm)
        got = _compute_accumulated_cost_matrix_gpu(cm.cuda()).cpu()
        assert torch.allclose(ref, got, atol=1e-4)


# ─── dtw() public API ─────────────────────────────────────────────────────────


class TestDtw:
    def test_returns_correct_distance_for_identical_sequences(self):
        a = torch.eye(3)
        out = dtw(a, a, DistanceFunction.EUCLIDEAN)
        assert out.distance == pytest.approx(0.0, abs=1e-5)

    def test_artifacts_none_by_default(self):
        a = torch.eye(3)
        out = dtw(a, a, DistanceFunction.EUCLIDEAN)
        assert out.cost_matrix is None
        assert out.accumulated_cost_matrix is None

    def test_artifacts_returned_when_requested(self):
        a = torch.eye(3)
        out = dtw(a, a, DistanceFunction.EUCLIDEAN, keep_artifacts=True)
        assert out.cost_matrix is not None
        assert out.accumulated_cost_matrix is not None

    def test_path_endpoints(self):
        a = torch.randn(4, 3)
        b = torch.randn(5, 3)
        out = dtw(a, b, DistanceFunction.EUCLIDEAN)
        assert out.optimal_warping_path[0, 0].item() == 0
        assert out.optimal_warping_path[0, 1].item() == 0
        assert out.optimal_warping_path[-1, 0].item() == 3
        assert out.optimal_warping_path[-1, 1].item() == 4

    def test_rejects_1d_input(self):
        with pytest.raises(AssertionError):
            dtw(torch.randn(3), torch.randn(3), DistanceFunction.EUCLIDEAN)

    def test_use_gpu_false_runs_cpu(self):
        a = torch.eye(3)
        out_cpu = dtw(a, a, DistanceFunction.EUCLIDEAN, use_gpu=False)
        assert out_cpu.distance == pytest.approx(0.0, abs=1e-5)

    def test_use_gpu_warns_and_falls_back_when_cc_too_low(self):
        if is_cuda_available(min_cc=7):
            pytest.skip("GPU has CC >= 7.0; fallback path not triggered on this machine")
        a = torch.eye(4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = dtw(a, a, DistanceFunction.EUCLIDEAN, use_gpu=True)
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)
        assert out.distance == pytest.approx(0.0, abs=1e-5)

    def test_use_gpu_result_matches_cpu(self):
        """GPU and CPU paths must produce the same distance and path."""
        if is_cuda_available(min_cc=7):
            pytest.skip("Full GPU path tested separately on CC >= 7.0 hardware")
        # On CC < 7.0, use_gpu falls back to CPU — results must still match
        torch.manual_seed(42)
        a = torch.randn(5, 8)
        b = torch.randn(6, 8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out_gpu = dtw(a, b, DistanceFunction.EUCLIDEAN, use_gpu=True)
        out_cpu = dtw(a, b, DistanceFunction.EUCLIDEAN, use_gpu=False)
        assert out_gpu.distance == pytest.approx(out_cpu.distance, rel=1e-5)
        assert torch.allclose(out_gpu.optimal_warping_path, out_cpu.optimal_warping_path)
