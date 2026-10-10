import numpy as np
import pytest

from weli.layers import Softmax
from weli.losses import (
    MSE,
    MAE,
    HuberLoss,
    MSLE,
    CrossEntropy,
    BinaryCrossEntropy,
    SparseCategoricalCrossEntropy,
    HingeLoss,
    SquaredHingeLoss,
    KLDivergence,
    PoissonLoss,
    CosineSimilarityLoss,
    LogCoshLoss,
    DiceLoss,
    CombinedLoss,
    TripletLoss,
    ContrastiveLoss,
)
from weli.losses.focal_loss import FocalLoss

STEP = 1e-6


def _numerical_gradient(value_fn, y_pred):
    grad = np.zeros_like(y_pred, dtype=float)
    for idx in np.ndindex(y_pred.shape):
        plus = y_pred.astype(float).copy()
        minus = y_pred.astype(float).copy()
        plus[idx] += STEP
        minus[idx] -= STEP
        grad[idx] = (value_fn(plus) - value_fn(minus)) / (2 * STEP)
    return grad


def _check_gradient(loss, y_pred, y_true):
    loss.forward(y_pred, y_true)
    analytic = loss.backward()
    numerical = _numerical_gradient(lambda p: float(loss.forward(p, y_true)), y_pred)
    np.testing.assert_allclose(analytic, numerical, atol=1e-5, rtol=1e-4)


rng = np.random.default_rng(0)


def test_regression_gradients_match_finite_differences():
    y_pred = rng.uniform(0.5, 2.0, size=(4, 3))
    y_true = rng.uniform(0.5, 2.0, size=(4, 3))
    _check_gradient(MSE(), y_pred, y_true)
    _check_gradient(MAE(), y_pred, y_true + 0.3)
    _check_gradient(HuberLoss(delta=0.5), y_pred, y_true)
    _check_gradient(MSLE(), y_pred, y_true)


def test_msle_gradient_is_zero_for_negative_predictions():
    loss = MSLE()
    y_pred = np.array([[-0.5, 1.0]])
    y_true = np.array([[0.5, 2.0]])
    loss.forward(y_pred, y_true)
    gradient = loss.backward()
    assert gradient[0, 0] == 0.0
    assert gradient[0, 1] != 0.0


def test_mean_reduction_divides_gradient_by_element_count():
    y_pred = rng.normal(size=(4, 3))
    y_true = rng.normal(size=(4, 3))
    loss = MSE()
    loss.forward(y_pred, y_true)
    gradient = loss.backward()
    np.testing.assert_allclose(gradient, 2 * (y_pred - y_true) / 12)


def test_combined_loss_gradient_matches_finite_differences():
    y_pred = rng.uniform(0.5, 2.0, size=(4, 3))
    y_true = rng.uniform(0.5, 2.0, size=(4, 3))
    combined = CombinedLoss([MSE(), HuberLoss()], weights=[0.5, 0.5])
    _check_gradient(combined, y_pred, y_true)


def test_classification_gradients_match_finite_differences():
    logits = rng.normal(size=(5, 3))
    probs = rng.uniform(0.1, 0.9, size=(5, 3))
    indices = np.array([0, 2, 1, 2, 0])
    one_hot = np.eye(3)[indices]

    _check_gradient(CrossEntropy(from_logits=True), logits, one_hot)
    _check_gradient(CrossEntropy(from_logits=True), logits, indices)
    _check_gradient(CrossEntropy(), probs / probs.sum(axis=1, keepdims=True), one_hot)
    _check_gradient(SparseCategoricalCrossEntropy(from_logits=True), logits, indices)
    _check_gradient(SparseCategoricalCrossEntropy(), probs, indices)
    _check_gradient(BinaryCrossEntropy(), probs[:, :1], (indices[:, None] > 0).astype(float))
    _check_gradient(BinaryCrossEntropy(from_logits=True), logits[:, :1],
                    (indices[:, None] > 0).astype(float))


def test_cross_entropy_ignore_index_excludes_rows():
    logits = rng.normal(size=(4, 3))
    indices = np.array([0, 2, 1, 2])
    loss = CrossEntropy(from_logits=True, ignore_index=2)
    value = loss.forward(logits, indices)

    keep = indices != 2
    per_row = CrossEntropy(from_logits=True, reduction='none').forward(
        logits[keep], indices[keep]
    )
    assert value == pytest.approx(np.mean(per_row))

    gradient = loss.backward()
    assert np.all(gradient[~keep] == 0)


def test_sparse_cross_entropy_ignore_index_excludes_rows():
    probs = rng.uniform(0.1, 0.9, size=(4, 3))
    indices = np.array([0, -1, 1, 2])
    loss = SparseCategoricalCrossEntropy(ignore_index=-1)
    loss.forward(probs, indices)
    gradient = loss.backward()
    assert np.all(gradient[1] == 0)
    _check_gradient(SparseCategoricalCrossEntropy(ignore_index=-1), probs,
                    np.array([0, 0, 1, 2]))


def test_hinge_accepts_zero_one_and_pm_one_labels():
    y_pred = rng.normal(size=(6, 1))
    zero_one = np.array([[0], [1], [1], [0], [1], [0]], dtype=float)
    pm_one = 2 * zero_one - 1
    value_zero_one = HingeLoss().forward(y_pred, zero_one)
    value_pm_one = HingeLoss().forward(y_pred, pm_one)
    assert value_zero_one == pytest.approx(value_pm_one)
    _check_gradient(HingeLoss(), y_pred, zero_one)
    _check_gradient(SquaredHingeLoss(), y_pred, pm_one)


def test_other_losses_gradients_match_finite_differences():
    probs = rng.uniform(0.1, 0.9, size=(4, 5))
    targets = rng.uniform(0.1, 0.9, size=(4, 5))
    _check_gradient(KLDivergence(), probs, targets)
    _check_gradient(PoissonLoss(), probs, targets)
    _check_gradient(CosineSimilarityLoss(), probs, targets)
    _check_gradient(LogCoshLoss(), rng.normal(size=(4, 5)), targets)
    _check_gradient(DiceLoss(), probs, (targets > 0.5).astype(float))


def test_log_cosh_value_is_symmetric():
    loss = LogCoshLoss(reduction='none')
    forward = loss.forward(np.array([[2.0]]), np.array([[0.0]]))
    backward = loss.forward(np.array([[0.0]]), np.array([[2.0]]))
    np.testing.assert_allclose(forward, backward)
    np.testing.assert_allclose(forward, np.log(np.cosh(2.0)))


def test_focal_loss_gradients_match_finite_differences():
    logits = rng.normal(size=(5, 3))
    probs = rng.uniform(0.1, 0.9, size=(5, 3))
    indices = np.array([0, 2, 1, 2, 0])
    _check_gradient(FocalLoss(from_logits=True), logits, indices)
    _check_gradient(FocalLoss(), probs, indices)


def test_triplet_gradients_match_finite_differences():
    embeddings = rng.normal(size=(9, 4))
    _check_gradient(TripletLoss(margin=2.0), embeddings, np.zeros(9))
    _check_gradient(TripletLoss(margin=2.0, distance='cosine'), embeddings, np.zeros(9))


def test_triplet_rejects_row_count_not_multiple_of_three():
    with pytest.raises(ValueError):
        TripletLoss().forward(rng.normal(size=(7, 4)), np.zeros(7))


def test_contrastive_gradients_match_finite_differences():
    embeddings = rng.normal(size=(6, 3))
    labels = np.array([1, 0, 0])
    _check_gradient(ContrastiveLoss(margin=2.0), embeddings, labels)


def test_softmax_layer_backward_matches_finite_differences():
    layer = Softmax()
    x = rng.normal(size=(4, 5))
    dout = rng.normal(size=(4, 5))

    layer.forward(x)
    analytic = layer.backward(dout)

    def objective(z):
        return float(np.sum(dout * Softmax().forward(z)))

    numerical = _numerical_gradient(objective, x)
    np.testing.assert_allclose(analytic, numerical, atol=1e-6, rtol=1e-5)
