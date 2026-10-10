"""Diagnostics numériques, contrôle des gradients et inspection des modèles."""

import numpy as np

from .device import array_module, to_cpu


def check_numerics(value, name="tensor", raise_on_error=True):
    """Vérifie qu'un tableau ou une structure de tableaux ne contient que des valeurs finies."""
    if isinstance(value, dict):
        results = [
            check_numerics(item, "{}.{}".format(name, key), raise_on_error)
            for key, item in value.items()
        ]
        return all(results)
    if isinstance(value, (list, tuple)):
        results = [
            check_numerics(item, "{}[{}]".format(name, index), raise_on_error)
            for index, item in enumerate(value)
        ]
        return all(results)

    module = array_module(value)
    values = module.asarray(value)
    finite = module.isfinite(values)
    if bool(module.all(finite)):
        return True
    if raise_on_error:
        indices = to_cpu(module.argwhere(~finite)).tolist()
        raise FloatingPointError(
            "{} contains NaN or infinite values at indices {}".format(name, indices)
        )
    return False


def tensor_summary(value, name="tensor"):
    """Retourne les informations utiles à l'inspection d'un tenseur numérique."""
    module = array_module(value)
    array = module.asarray(value)
    host_array = to_cpu(array)
    finite_mask = np.isfinite(host_array)
    finite_values = host_array[finite_mask]
    return {
        "name": name,
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite": bool(np.all(finite_mask)),
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
        "mean": float(np.mean(finite_values)) if finite_values.size else None,
    }


def check_gradients(model_or_layer, inputs, grad_output=None, epsilon=1e-5,
                    atol=1e-5, rtol=1e-3):
    """Compare les gradients analytiques aux différences finies."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    is_model = hasattr(model_or_layer, "layers")
    layers = model_or_layer.layers if is_model else [model_or_layer]
    if not layers:
        raise ValueError("At least one layer with parameters is required")

    batch_size = inputs[0].shape[0] if isinstance(inputs, list) else inputs.shape[0]

    def forward():
        if is_model:
            return model_or_layer.forward(inputs, training=False)
        model_or_layer.training = False
        return model_or_layer.forward(inputs)

    output = forward()
    if isinstance(output, (list, tuple)):
        raise ValueError("Gradient checking currently requires a single model output")
    if grad_output is None:
        grad_output = np.ones_like(output)
    if grad_output.shape != output.shape:
        raise ValueError(
            "grad_output shape {} does not match output shape {}".format(
                grad_output.shape, output.shape
            )
        )

    model_or_layer.backward(grad_output)
    analytic = {}
    parameters = []
    for layer_index, layer in enumerate(layers):
        for parameter_name, parameter in layer.parameters.items():
            key = "{}.{}".format(
                getattr(layer, "name", "layer_{}".format(layer_index)),
                parameter_name,
            )
            if parameter_name not in layer.gradients:
                raise ValueError("No analytical gradient found for {}".format(key))
            analytic[key] = np.array(layer.gradients[parameter_name], copy=True)
            parameters.append((key, parameter, layer.gradients[parameter_name]))
    if not parameters:
        raise ValueError("No parameters are available for gradient checking")

    results = {}
    try:
        for key, parameter, _ in parameters:
            numerical = np.zeros_like(parameter, dtype=float)
            for index in np.ndindex(parameter.shape):
                original = parameter[index]
                try:
                    parameter[index] = original + epsilon
                    plus = forward()
                    plus_value = float(np.sum(plus * grad_output) / batch_size)
                    parameter[index] = original - epsilon
                    minus = forward()
                    minus_value = float(np.sum(minus * grad_output) / batch_size)
                finally:
                    parameter[index] = original
                numerical[index] = (plus_value - minus_value) / (2.0 * epsilon)

            analytical = analytic[key]
            absolute_error = np.abs(analytical - numerical)
            denominator = np.maximum(
                np.maximum(np.abs(analytical), np.abs(numerical)), epsilon
            )
            relative_error = absolute_error / denominator
            results[key] = {
                "passed": bool(np.allclose(analytical, numerical, atol=atol, rtol=rtol)),
                "max_absolute_error": float(np.max(absolute_error))
                if absolute_error.size else 0.0,
                "max_relative_error": float(np.max(relative_error))
                if relative_error.size else 0.0,
            }
    finally:
        forward()
        model_or_layer.backward(grad_output)

    return results


def visualize_graph(model, filename=None):
    """Construit une représentation Graphviz DOT du graphe du modèle."""
    lines = ["digraph Weli {", "  rankdir=LR;", "  node [shape=box];"]
    forward_order = getattr(model, "_forward_order", None)
    if forward_order:
        node_ids = {node: "node_{}".format(index)
                    for index, node in enumerate(forward_order)}
        for node in forward_order:
            label = "{}\n{}".format(node.name, node.layer.__class__.__name__)
            lines.append('  {} [label="{}"];'.format(node_ids[node], _dot_escape(label)))
            for parent in node.inputs:
                if hasattr(parent, "layer") and parent in node_ids:
                    lines.append("  {} -> {};".format(node_ids[parent], node_ids[node]))
    else:
        layers = getattr(model, "layers", None)
        if not layers:
            raise ValueError("model must expose layers or a functional computation graph")
        for index, layer in enumerate(layers):
            label = "{}\n{}".format(layer.name, layer.__class__.__name__)
            lines.append('  layer_{} [label="{}"];'.format(index, _dot_escape(label)))
            if index:
                lines.append("  layer_{} -> layer_{};".format(index - 1, index))
    lines.append("}")
    graph = "\n".join(lines)
    if filename is not None:
        with open(filename, "w", encoding="utf-8") as graph_file:
            graph_file.write(graph)
            graph_file.write("\n")
    return graph


def debug_model(model, inputs, callback=None):
    """Exécute un modèle séquentiel couche par couche et expose ses activations."""
    layers = getattr(model, "layers", None)
    if not layers:
        raise ValueError("model must contain at least one layer")
    if getattr(model, "_forward_order", None):
        raise TypeError("debug_model currently supports sequential models only")

    previous_training = [getattr(layer, "training", True) for layer in layers]
    records = []
    output = inputs
    try:
        for index, layer in enumerate(layers):
            layer.training = False
            output = layer.forward(output)
            summary = tensor_summary(output, getattr(layer, "name", "layer_{}".format(index)))
            record = {"index": index, "layer": layer.name, "output": summary}
            records.append(record)
            if callback is not None:
                callback(index, layer, output, summary)
    finally:
        for layer, training in zip(layers, previous_training):
            layer.training = training
    return output, records


def _dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


__all__ = [
    "check_gradients",
    "check_numerics",
    "debug_model",
    "tensor_summary",
    "visualize_graph",
]
