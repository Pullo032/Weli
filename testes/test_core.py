import numpy as np
import pytest

from weli.core import (
    Device,
    MemoryManager,
    Profiler,
    check_gradients,
    check_numerics,
    debug_model,
    device_scope,
    get_device,
    get_memory_usage,
    profile_operation,
    tensor_memory_bytes,
    tensor_summary,
    to_cpu,
    to_device,
    visualize_graph,
)
from weli.layers import Dense
from weli.models import Functional, Input, Sequential
from weli.models.functional import Dense as FunctionalDense


def test_device_selection_transfer_and_scoped_default():
    original = get_device()
    data = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert str(Device("cpu")) == "cpu"
    on_cpu = to_device(data, "cpu")
    assert isinstance(on_cpu, np.ndarray)
    np.testing.assert_array_equal(to_cpu(on_cpu), data)
    with device_scope("cpu"):
        assert str(get_device()) == "cpu"
    assert get_device() == original

    with pytest.raises(ValueError, match="Device must"):
        Device("tpu")


def test_memory_manager_reports_tensor_size_and_cpu_usage():
    tensor = np.zeros((3, 4), dtype=np.float64)
    manager = MemoryManager("cpu")

    assert tensor_memory_bytes(tensor) == 96
    assert manager.tensor_size(tensor) == 96
    usage = manager.usage()
    assert usage["device"] == "cpu"
    assert usage["process_rss_bytes"] > 0
    assert get_memory_usage("cpu")["device"] == "cpu"
    assert manager.cleanup()["device"] == "cpu"

    with pytest.raises(TypeError, match="nbytes"):
        tensor_memory_bytes(object())


def test_profiler_measures_contexts_and_decorated_calls():
    profiler = Profiler(device="cpu")

    @profiler.profile(name="decorated")
    def multiply(value):
        return value * 2

    with profile_operation("context", profiler=profiler):
        assert multiply(4) == 8

    rows = {row["name"]: row for row in profiler.summary()}
    assert rows["decorated"]["calls"] == 1
    assert rows["decorated"]["total_time"] >= 0
    assert rows["context"]["calls"] == 1
    assert "decorated" in profiler.format_summary()
    profiler.reset()
    assert profiler.summary() == []


def test_numerical_diagnostics_and_gradient_check():
    values = np.array([1.0, 2.0])
    assert check_numerics(values)
    assert not check_numerics([values, np.array([np.inf])], raise_on_error=False)
    with pytest.raises(FloatingPointError, match="scores"):
        check_numerics(np.array([np.nan]), name="scores")

    layer = Dense(2, kernel_initializer="xavier")
    layer.initialize((3,))
    inputs = np.array([[0.2, -0.4, 0.7], [0.5, 0.3, -0.1]])
    results = check_gradients(layer, inputs)

    assert set(results) == {"Dense.W", "Dense.b"}
    assert all(result["passed"] for result in results.values())
    summary = tensor_summary(values, name="scores")
    assert summary["finite"]
    assert summary["shape"] == (2,)
    assert summary["mean"] == 1.5


def test_graph_visualization_and_layer_by_layer_debugging(tmp_path):
    model = Sequential([Dense(3, name="hidden"), Dense(1, name="output")])
    model.initialize((2,))
    graph_path = tmp_path / "network.dot"
    graph = visualize_graph(model, str(graph_path))

    assert "hidden" in graph and "output" in graph
    assert "layer_0 -> layer_1" in graph_path.read_text(encoding="utf-8")

    callback_records = []
    inputs = np.ones((2, 2))
    output, records = debug_model(
        model,
        inputs,
        callback=lambda index, layer, value, stats: callback_records.append(stats),
    )
    assert output.shape == (2, 1)
    assert len(records) == len(callback_records) == 2
    assert all(record["output"]["finite"] for record in records)


def test_functional_graph_visualization_preserves_branches():
    inputs = Input((2,), name="features")
    left = FunctionalDense(1, name="left")(inputs)
    right = FunctionalDense(1, name="right")(inputs)
    model = Functional(inputs, [left, right])
    model.initialize((2,))

    graph = visualize_graph(model)
    assert "features" in graph
    assert "left" in graph and "right" in graph
