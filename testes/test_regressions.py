import numpy as np

from weli.layers import Conv2D, MaxPool2D, Flatten, Dense
from weli.losses import CategoricalCrossEntropy, SparseCategoricalCrossEntropy
from weli.models import Sequential, load_model
from weli.optimizers import Adam


def _cnn():
    model = Sequential([
        Conv2D(4, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2),
        Flatten(),
        Dense(3),
    ])
    model.compile(input_shape=(8, 8, 1),
                  loss_fn=CategoricalCrossEntropy(from_logits=True),
                  optimizer=Adam(lr=0.001))
    return model


def test_flatten_backward_trains_conv_model():
    np.random.seed(0)
    model = _cnn()
    x = np.random.rand(4, 8, 8, 1)
    y = np.eye(3)[[0, 1, 2, 0]]
    loss, _ = model.train_step(x, y, model._loss_fn, model._optimizer)
    assert np.isfinite(loss)


def test_load_model_keeps_trained_weights(tmp_path):
    np.random.seed(0)
    model = _cnn()
    x = np.random.rand(5, 8, 8, 1)
    expected = model.predict(x)

    path = tmp_path / "model.weli"
    model.save(str(path))

    reloaded = load_model(str(path))
    np.testing.assert_allclose(reloaded.predict(x), expected)


def test_accuracy_with_integer_labels():
    np.random.seed(0)
    model = _cnn()
    x = np.random.rand(6, 8, 8, 1)
    labels = model.predict(x).argmax(axis=1)

    _, accuracy = model.val_step(x, labels, SparseCategoricalCrossEntropy(from_logits=True))
    assert accuracy == 1.0
