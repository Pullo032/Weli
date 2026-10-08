"""Composition helpers for chaining or running models in parallel."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .model import Model


class ModelContainer(Model):
    """Compose models as a linear pipeline, feeding each output to the next."""

    def __init__(self, models: List[Model], name: Optional[str] = None):
        super().__init__(name)
        if not models:
            raise ValueError("ModelContainer requires at least one model.")
        if not all(isinstance(model, Model) for model in models):
            raise TypeError("Every item in models must be a Model instance.")

        self.models = list(models)
        for model in self.models:
            self.layers.extend(model.layers)
            self.trainable_layers.extend(model.trainable_layers)

    def initialize(self, input_shape: Tuple) -> Tuple:
        current_shape = tuple(input_shape)
        self._input_shape = current_shape
        for model in self.models:
            current_shape = tuple(model.initialize(current_shape))
        self._output_shape = current_shape
        self.initialized = True
        return self._output_shape

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.initialized:
            self.initialize(x.shape[1:])

        output = x
        for model in self.models:
            output = model.forward(output, training=training)
        return output

    def backward(self, dout: np.ndarray) -> np.ndarray:
        gradient = dout
        for model in reversed(self.models):
            gradient = model.backward(gradient)
        return gradient

    def get_config(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "models": [model.get_config() for model in self.models],
            "input_shape": self._input_shape,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ModelContainer":
        from .registry import registry

        models = [registry.deserialize_model(model_config)
                  for model_config in config.get("models", [])]
        container = cls(models, name=config.get("name"))
        input_shape = config.get("input_shape")
        if input_shape is not None:
            container.initialize(tuple(input_shape))
        return container


class Parallel(Model):
    """Run models on the same input and merge their outputs."""

    _MERGE_OPERATIONS = {"concat", "add", "average"}

    def __init__(
        self,
        branches: List[Model],
        merge_op: str = "concat",
        name: Optional[str] = None,
    ):
        super().__init__(name)
        if not branches:
            raise ValueError("Parallel requires at least one branch.")
        if not all(isinstance(branch, Model) for branch in branches):
            raise TypeError("Every branch must be a Model instance.")
        if merge_op not in self._MERGE_OPERATIONS:
            choices = ", ".join(sorted(self._MERGE_OPERATIONS))
            raise ValueError(f"Unknown merge_op {merge_op!r}; expected one of: {choices}.")

        self.branches = list(branches)
        self.merge_op = merge_op
        for branch in self.branches:
            self.layers.extend(branch.layers)
            self.trainable_layers.extend(branch.trainable_layers)

    def initialize(self, input_shape: Tuple) -> Tuple:
        self._input_shape = tuple(input_shape)
        output_shapes = [tuple(branch.initialize(self._input_shape))
                         for branch in self.branches]

        if self.merge_op == "concat":
            first_shape = output_shapes[0]
            if not first_shape:
                raise ValueError("Cannot concatenate scalar branch outputs.")
            for shape in output_shapes[1:]:
                if len(shape) != len(first_shape) or shape[:-1] != first_shape[:-1]:
                    raise ValueError(
                        "All branch output shapes must match except on the "
                        f"last axis for concat; got {output_shapes}."
                    )
            self._output_shape = first_shape[:-1] + (
                sum(shape[-1] for shape in output_shapes),
            )
        else:
            if any(shape != output_shapes[0] for shape in output_shapes[1:]):
                raise ValueError(
                    f"All branch output shapes must match for {self.merge_op}; "
                    f"got {output_shapes}."
                )
            self._output_shape = output_shapes[0]

        self.initialized = True
        return self._output_shape

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not self.initialized:
            self.initialize(x.shape[1:])

        outputs = [branch.forward(x, training=training) for branch in self.branches]
        if self.merge_op == "concat":
            return np.concatenate(outputs, axis=-1)
        if self.merge_op == "add":
            return np.add.reduce(outputs)
        return np.add.reduce(outputs) / len(outputs)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self.merge_op == "concat":
            sizes = [branch._output_shape[-1] for branch in self.branches]
            branch_gradients = np.split(dout, np.cumsum(sizes)[:-1], axis=-1)
        elif self.merge_op == "average":
            branch_gradients = [dout / len(self.branches)] * len(self.branches)
        else:
            branch_gradients = [dout] * len(self.branches)

        input_gradients = [
            branch.backward(gradient)
            for branch, gradient in zip(self.branches, branch_gradients)
        ]
        combined = input_gradients[0].copy()
        for gradient in input_gradients[1:]:
            combined += gradient
        return combined

    def get_config(self) -> Dict[str, Any]:
        return {
            "class_name": self.__class__.__name__,
            "name": self.name,
            "merge_op": self.merge_op,
            "branches": [branch.get_config() for branch in self.branches],
            "input_shape": self._input_shape,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Parallel":
        from .registry import registry

        branches = [registry.deserialize_model(branch_config)
                    for branch_config in config.get("branches", [])]
        parallel = cls(
            branches,
            merge_op=config.get("merge_op", "concat"),
            name=config.get("name"),
        )
        input_shape = config.get("input_shape")
        if input_shape is not None:
            parallel.initialize(tuple(input_shape))
        return parallel
