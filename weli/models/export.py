"""
Exportation de modèles vers d'autres formats.
"""
import re
import numpy as np

def export_to_onnx(model: 'Model', filepath: str):
    """
    Exporte un modèle vers le format ONNX.
    Not implemented: raises NotImplementedError without creating a file.
    
    Args:
        model: Modèle à exporter
        filepath: Chemin du fichier
    """
    raise NotImplementedError(
        "ONNX export is not implemented yet; no file was written."
    )

def export_to_tflite(model: 'Model', filepath: str):
    """
    Exporte un modèle vers TensorFlow Lite.
    Not implemented: raises NotImplementedError without creating a file.
    
    Args:
        model: Modèle à exporter
        filepath: Chemin du fichier
    """
    raise NotImplementedError(
        "TensorFlow Lite export is not implemented yet; no file was written."
    )

def export_to_coreml(model: 'Model', filepath: str):
    """
    Exporte un modèle vers CoreML.
    Not implemented: raises NotImplementedError without creating a file.
    
    Args:
        model: Modèle à exporter
        filepath: Chemin du fichier
    """
    raise NotImplementedError(
        "CoreML export is not implemented yet; no file was written."
    )

def generate_c_code(model: 'Model', filepath: str):
    """
    Génère du code C pour le modèle (pour l'embarqué).
    
    Args:
        model: Modèle
        filepath: Chemin du fichier
    """
    from ..layers import Dense, ReLU, Sigmoid, Tanh, Softmax
    from .sequential import Sequential

    if not isinstance(model, Sequential):
        raise NotImplementedError("C inference generation currently supports Sequential models only.")

    layers = getattr(model, "layers", [])
    if not layers or any(not isinstance(layer, (Dense, ReLU, Sigmoid, Tanh, Softmax))
                         for layer in layers):
        raise NotImplementedError(
            "C inference generation currently supports Dense, ReLU, Sigmoid, "
            "Tanh, and Softmax layers only."
        )
    if not any(isinstance(layer, Dense) for layer in layers):
        raise ValueError("The model must contain at least one initialized Dense layer.")
    if not isinstance(layers[0], Dense):
        raise NotImplementedError(
            "C inference generation requires the first layer to be Dense."
        )

    dense_layers = [layer for layer in layers if isinstance(layer, Dense)]
    if any(layer.parameters.get("W") is None for layer in dense_layers):
        raise ValueError("Initialize the model before generating C inference code.")
    input_features = dense_layers[0].parameters["W"].shape[0]
    for previous, current in zip(dense_layers, dense_layers[1:]):
        if previous.units != current.parameters["W"].shape[0]:
            raise ValueError("Dense layer dimensions are not sequentially compatible.")

    def identifier(index, name):
        return re.sub(r"\W|^(?=\d)", "_", f"weli_{index}_{name}")

    with open(filepath, "w", encoding="utf-8") as output:
        output.write("#include <math.h>\n#include <stddef.h>\n\n")
        for index, layer in enumerate(dense_layers):
            for key, value in layer.parameters.items():
                if value is None:
                    continue
                name = identifier(index, key)
                flat = ", ".join(f"{float(item):.9g}f" for item in value.ravel())
                output.write(f"static const float {name}[{value.size}] = {{{flat}}};\n")
            if layer.parameters.get("b") is None:
                name = identifier(index, "b")
                output.write(
                    f"static const float {name}[{layer.units}] = "
                    f"{{{', '.join('0.0f' for _ in range(layer.units))}}};\n"
                )
        output.write("\n")
        output.write("void weli_predict(const float *input, float *output) {\n")
        max_width = max(layer.units for layer in dense_layers)
        output.write(f"    float buffer_a[{max_width}];\n")
        output.write(f"    float buffer_b[{max_width}];\n")
        current = "input"
        current_width = input_features
        dense_index = 0
        for layer in layers:
            if isinstance(layer, Dense):
                weight = identifier(dense_index, "W")
                bias = identifier(dense_index, "b")
                target = "output" if layer is dense_layers[-1] else (
                    "buffer_a" if dense_index % 2 == 0 else "buffer_b"
                )
                output.write(f"    for (size_t j = 0; j < {layer.units}; ++j) {{\n")
                output.write(f"        float value = {bias}[j];\n")
                output.write(f"        for (size_t i = 0; i < {current_width}; ++i) "
                             f"value += {current}[i] * {weight}[i * {layer.units} + j];\n")
                activation = layer.activation_name
                if activation == "relu":
                    output.write("        value = value > 0.0f ? value : 0.0f;\n")
                elif activation == "sigmoid":
                    output.write("        value = 1.0f / (1.0f + expf(-value));\n")
                elif activation == "tanh":
                    output.write("        value = tanhf(value);\n")
                output.write(f"        {target}[j] = value;\n    }}\n")
                if activation == "softmax":
                    output.write(f"    float max_value = {target}[0];\n")
                    output.write(f"    for (size_t j = 1; j < {layer.units}; ++j) "
                                 f"if ({target}[j] > max_value) max_value = {target}[j];\n")
                    output.write(f"    float total = 0.0f;\n"
                                 f"    for (size_t j = 0; j < {layer.units}; ++j) "
                                 f"{{ {target}[j] = expf({target}[j] - max_value); total += {target}[j]; }}\n"
                                 f"    for (size_t j = 0; j < {layer.units}; ++j) "
                                 f"{target}[j] /= total;\n")
                current = target
                current_width = layer.units
                dense_index += 1
            else:
                output.write(f"    for (size_t j = 0; j < {current_width}; ++j) {{\n")
                if isinstance(layer, ReLU):
                    output.write(f"        {current}[j] = {current}[j] > 0.0f ? {current}[j] : 0.0f;\n")
                elif isinstance(layer, Sigmoid):
                    output.write(f"        {current}[j] = 1.0f / (1.0f + expf(-{current}[j]));\n")
                elif isinstance(layer, Tanh):
                    output.write(f"        {current}[j] = tanhf({current}[j]);\n")
                output.write("    }\n")
                if isinstance(layer, Softmax):
                    output.write(f"    float max_value = {current}[0];\n")
                    output.write(f"    for (size_t j = 1; j < {current_width}; ++j) "
                                 f"if ({current}[j] > max_value) max_value = {current}[j];\n")
                    output.write(f"    float total = 0.0f;\n"
                                 f"    for (size_t j = 0; j < {current_width}; ++j) "
                                 f"{{ {current}[j] = expf({current}[j] - max_value); total += {current}[j]; }}\n"
                                 f"    for (size_t j = 0; j < {current_width}; ++j) "
                                 f"{current}[j] /= total;\n")
        output.write("}\n")

def export_parameters_csv(model: 'Model', directory: str):
    """
    Exporte tous les paramètres en fichiers CSV.
    
    Args:
        model: Modèle
        directory: Répertoire de destination
    """
    import os
    import csv
    
    os.makedirs(directory, exist_ok=True)
    params = model.get_parameters()
    
    for key, value in params.items():
        filename = os.path.join(directory, f"{key}.csv")
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if value.ndim == 1:
                writer.writerow(['index', 'value'])
                for i, val in enumerate(value):
                    writer.writerow([i, val])
            elif value.ndim == 2:
                for row in value:
                    writer.writerow(row)
            else:
                writer.writerow(['indices', 'value'])
                for indices in np.ndindex(value.shape):
                    writer.writerow([','.join(map(str, indices)), value[indices]])
        
        print(f"Exported {key} to {filename}")