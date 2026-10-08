"""
API fonctionnelle pour Weli - Permet de créer des modèles avec des graphes de calcul complexes.
Inspiré de Keras Functional API.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from .model import Model

class LayerNode:
    """
    Nœud dans le graphe de calcul pour l'API fonctionnelle.
    
    Représente une couche avec ses connexions entrantes et sortantes.
    """
    
    def __init__(self, layer, name: Optional[str] = None):
        self.layer = layer
        self.name = name or layer.name
        self.inputs = []  # Nœuds d'entrée
        self.outputs = []  # Nœuds de sortie
        self.visited = False
        self.output_tensor = None
    
    def __call__(self, *inputs):
        """
        Connecte ce nœud à ses inputs.
        
        Args:
            *inputs: Nœuds d'entrée ou tenseurs
            
        Returns:
            Ce nœud (pour chaînage)
        """
        self.inputs.extend(inputs)
        for inp in inputs:
            if isinstance(inp, LayerNode):
                inp.outputs.append(self)
        return self
    
    def forward(self, *input_tensors, training: bool = True):
        """
        Propagation avant pour ce nœud.
        
        Args:
            *input_tensors: Tenseurs d'entrée
            training: Mode entraînement
            
        Returns:
            Tenseur de sortie
        """
        # Combiner les inputs si plusieurs
        if len(input_tensors) == 1:
            x = input_tensors[0]
        else:
            # Pour les couches avec multiples inputs, on les combine
            # (ex: concaténation, addition, etc.)
            x = input_tensors
        
        self.layer.training = training
        self.output_tensor = self.layer.forward(x)
        return self.output_tensor
    
    def backward(self, dout: np.ndarray):
        """
        Rétropropagation pour ce nœud.
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Gradients pour chaque input
        """
        return self.layer.backward(dout)
    
    def get_config(self):
        """Retourne la configuration du nœud."""
        return {
            'name': self.name,
            'layer_config': self.layer.get_config(),
            'input_names': [inp.name if isinstance(inp, LayerNode) else str(inp) 
                          for inp in self.inputs]
        }

class Input(LayerNode):
    """
    Nœud d'input spécial pour l'API fonctionnelle.
    """
    
    def __init__(self, shape: Tuple, name: Optional[str] = None):
        from ..layers.base import Layer
        
        # Créer une couche factice pour l'input
        class InputLayer(Layer):
            def __init__(self, shape, name):
                super().__init__(name)
                self.shape = shape
                self.trainable = False
            
            def forward(self, x):
                return x
            
            def backward(self, dout):
                return dout
            
            def initialize(self, input_shape):
                return self.shape
            
            def get_config(self):
                config = super().get_config()
                config['shape'] = self.shape
                return config
        
        layer = InputLayer(shape, name or f"input_{id(self)}")
        super().__init__(layer, layer.name)
    
    def __call__(self, *inputs):
        # L'input n'a pas d'inputs
        return self

def layer_wrapper(layer_class):
    """
    Décorateur pour envelopper les couches dans des LayerNode.
    
    Args:
        layer_class: Classe de la couche
        
    Returns:
        Fonction wrapper
    """
    def wrapper(*args, **kwargs):
        layer = layer_class(*args, **kwargs)
        return LayerNode(layer)
    return wrapper

# Wrappers pour les couches communes
Dense = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).Dense(*args, **kwargs))
Conv2D = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).Conv2D(*args, **kwargs))
MaxPool2D = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).MaxPool2D(*args, **kwargs))
Flatten = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).Flatten(*args, **kwargs))
Dropout = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).Dropout(*args, **kwargs))
BatchNorm2D = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).BatchNorm2D(*args, **kwargs))
ReLU = layer_wrapper(lambda *args, **kwargs: __import__('weli.layers', fromlist=['']).ReLU(*args, **kwargs))

# Opérations fonctionnelles
def add(inputs: List[LayerNode], name: Optional[str] = None) -> LayerNode:
    """
    Addition de plusieurs couches.
    
    Args:
        inputs: Liste des nœuds à additionner
        name: Nom de l'opération
        
    Returns:
        Nœud de l'addition
    """
    from ..layers.base import Layer
    
    class AddLayer(Layer):
        def __init__(self, name):
            super().__init__(name)
            self.trainable = False
            self.functional_operation = "add"
            self.input_count = 0
        
        def forward(self, inputs):
            # inputs est une liste de tenseurs
            result = inputs[0].copy()
            for i in range(1, len(inputs)):
                result += inputs[i]
            return result
        
        def backward(self, dout):
            # Gradient est distribué à tous les inputs
            return [dout for _ in range(self.input_count)]
        
        def initialize(self, input_shapes):
            # Tous les inputs doivent avoir la même shape
            for shape in input_shapes[1:]:
                if shape != input_shapes[0]:
                    raise ValueError(f"All inputs to Add must have same shape, got {input_shapes}")
            return input_shapes[0]
    
    layer = AddLayer(name or "add")
    layer.input_count = len(inputs)
    node = LayerNode(layer)
    
    # Connecter les inputs
    for inp in inputs:
        node.inputs.append(inp)
        inp.outputs.append(node)
    
    return node

def concatenate(inputs: List[LayerNode], axis: int = -1, name: Optional[str] = None) -> LayerNode:
    """
    Concatenation de plusieurs couches.
    
    Args:
        inputs: Liste des nœuds à concaténer
        axis: Axe de concaténation
        name: Nom de l'opération
        
    Returns:
        Nœud de la concaténation
    """
    from ..layers.base import Layer
    
    class ConcatenateLayer(Layer):
        def __init__(self, axis, name):
            super().__init__(name)
            self.axis = axis
            self.trainable = False
            self.functional_operation = "concatenate"
            self.input_shapes = []
        
        def forward(self, inputs):
            return np.concatenate(inputs, axis=self.axis)
        
        def backward(self, dout):
            # Split le gradient selon l'axe
            sizes = [shape[self.axis] for shape in self.input_shapes]
            splits = np.split(dout, np.cumsum(sizes)[:-1], axis=self.axis)
            return splits
        
        def initialize(self, input_shapes):
            # Vérifier que toutes les shapes sont compatibles sauf sur l'axe de concat
            base_shape = list(input_shapes[0])
            axis = self.axis if self.axis >= 0 else len(base_shape) + self.axis
            
            for i, shape in enumerate(input_shapes[1:], 1):
                shape = list(shape)
                if len(shape) != len(base_shape):
                    raise ValueError(f"All inputs must have same rank, got {len(base_shape)} and {len(shape)}")
                
                for j in range(len(shape)):
                    if j != axis and shape[j] != base_shape[j]:
                        raise ValueError(f"Input {i} incompatible on axis {j}: {shape[j]} != {base_shape[j]}")
            
            # Calculer la shape de sortie
            output_shape = base_shape.copy()
            output_shape[axis] = sum(shape[axis] for shape in input_shapes)
            return tuple(output_shape)

        def get_config(self):
            config = super().get_config()
            config["axis"] = self.axis
            return config
    
    layer = ConcatenateLayer(axis, name or "concatenate")
    node = LayerNode(layer)
    
    for inp in inputs:
        node.inputs.append(inp)
        inp.outputs.append(node)
    
    return node

def multiply(inputs: List[LayerNode], name: Optional[str] = None) -> LayerNode:
    """
    Multiplication élément par élément.
    
    Args:
        inputs: Liste des nœuds à multiplier
        name: Nom de l'opération
        
    Returns:
        Nœud de la multiplication
    """
    from ..layers.base import Layer
    
    class MultiplyLayer(Layer):
        def __init__(self, name):
            super().__init__(name)
            self.trainable = False
            self.functional_operation = "multiply"
            self.input_count = 0
            self.cached_inputs = []
        
        def forward(self, inputs):
            result = inputs[0].copy()
            for i in range(1, len(inputs)):
                result *= inputs[i]
            return result
        
        def backward(self, dout):
            gradients = []
            for i in range(self.input_count):
                # Gradient pour l'input i: dout * produit des autres inputs
                grad = dout.copy()
                for j, other_inp in enumerate(self.cached_inputs):
                    if j != i:
                        grad *= other_inp
                gradients.append(grad)
            return gradients
        
        def initialize(self, input_shapes):
            # Tous les inputs doivent avoir la même shape
            for shape in input_shapes[1:]:
                if shape != input_shapes[0]:
                    raise ValueError(f"All inputs to Multiply must have same shape, got {input_shapes}")
            return input_shapes[0]
    
    layer = MultiplyLayer(name or "multiply")
    layer.input_count = len(inputs)
    node = LayerNode(layer)
    
    for inp in inputs:
        node.inputs.append(inp)
        inp.outputs.append(node)
    
    return node

class Functional(Model):
    """
    Modèle créé avec l'API fonctionnelle.
    
    Supporte les graphes de calcul complexes avec branches,
    connexions multiples, et opérations diverses.
    """
    
    def __init__(self, inputs: Union[LayerNode, List[LayerNode]], 
                 outputs: Union[LayerNode, List[LayerNode]], 
                 name: Optional[str] = None):
        """
        Initialise un modèle fonctionnel.
        
        Args:
            inputs: Nœud(s) d'input
            outputs: Nœud(s) de sortie
            name: Nom du modèle
        """
        super().__init__(name)
        
        self.input_nodes = inputs if isinstance(inputs, list) else [inputs]
        self.output_nodes = outputs if isinstance(outputs, list) else [outputs]
        
        # Construire le graphe de calcul
        self._build_computation_graph()
        
        # Initialiser les flags
        self._topology_sorted = False
        self._forward_order = []
        self._backward_order = []
    
    def _build_computation_graph(self):
        """Construit le graphe de calcul à partir des inputs et outputs."""
        # Collecter tous les nœuds accessibles depuis les outputs
        self._all_nodes = set()
        
        def collect_nodes(node):
            if node in self._all_nodes:
                return
            self._all_nodes.add(node)
            
            # Parcourir récursivement les inputs
            for inp in node.inputs:
                if isinstance(inp, LayerNode):
                    collect_nodes(inp)
        
        for output in self.output_nodes:
            collect_nodes(output)
        
        # Vérifier que tous les inputs sont dans le graphe
        for inp in self.input_nodes:
            if inp not in self._all_nodes:
                self._all_nodes.add(inp)
    
    def _topological_sort(self):
        """Tri topologique du graphe pour déterminer l'ordre de calcul."""
        if self._topology_sorted:
            return
        
        # Algorithme de Kahn pour le tri topologique
        in_degree = {node: 0 for node in self._all_nodes}
        
        # Calculer les degrés entrants
        for node in self._all_nodes:
            for out_node in node.outputs:
                if out_node in self._all_nodes:
                    in_degree[out_node] += 1
        
        # File des nœuds sans dépendances
        node_order = []
        seen = set()

        def visit(node):
            if node in seen:
                return
            seen.add(node)
            node_order.append(node)
            for out_node in node.outputs:
                if out_node in self._all_nodes:
                    visit(out_node)

        for input_node in self.input_nodes:
            visit(input_node)
        for node in self._all_nodes:
            visit(node)
        order_index = {node: index for index, node in enumerate(node_order)}
        queue = sorted(
            (node for node in self._all_nodes if in_degree[node] == 0),
            key=order_index.get
        )
        self._forward_order = []
        
        while queue:
            node = queue.pop(0)
            self._forward_order.append(node)
            
            # Réduire le degré des voisins
            for out_node in node.outputs:
                if out_node in self._all_nodes:
                    in_degree[out_node] -= 1
                    if in_degree[out_node] == 0:
                        queue.append(out_node)
                        queue.sort(key=order_index.get)
        
        # Vérifier s'il y a un cycle
        if len(self._forward_order) != len(self._all_nodes):
            raise ValueError("The computation graph contains cycles!")
        
        # Ordre inverse pour la backward pass
        self._backward_order = list(reversed(self._forward_order))
        self._topology_sorted = True
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Union[Tuple, List[Tuple]]:
        """
        Initialise toutes les couches du modèle.
        
        Args:
            input_shapes: Shape(s) de(s) input(s)
            
        Returns:
            Shape(s) de(s) output(s)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) != len(self.input_nodes):
            raise ValueError(f"Expected {len(self.input_nodes)} input shapes, got {len(input_shapes)}")
        
        # Assigner les shapes aux inputs
        for inp_node, shape in zip(self.input_nodes, input_shapes):
            inp_node.layer.shape = tuple(shape)
            inp_node.layer.output_shape = (None,) + tuple(shape)
        
        # Tri topologique
        self._topological_sort()
        
        # Initialiser dans l'ordre topologique
        node_shapes = {}
        
        for node in self._forward_order:
            if node in self.input_nodes:
                node_shapes[node] = (None,) + tuple(node.layer.shape)
                continue
            
            # Récupérer les shapes des inputs
            input_shapes_for_node = []
            for inp in node.inputs:
                if isinstance(inp, LayerNode):
                    input_shapes_for_node.append(node_shapes[inp])
                else:
                    # C'est un tenseur directement
                    input_shapes_for_node.append(inp.shape if hasattr(inp, 'shape') else ())
            
            # Initialiser la couche
            if len(input_shapes_for_node) == 1:
                output_shape = node.layer.initialize(input_shapes_for_node[0])
            else:
                output_shape = node.layer.initialize(input_shapes_for_node)
                if getattr(node.layer, "functional_operation", None) == "concatenate":
                    node.layer.input_shapes = input_shapes_for_node
            
            node_shapes[node] = output_shape
        
        # Stocker les shapes d'input et output
        self._input_shape = input_shapes[0] if len(input_shapes) == 1 else input_shapes
        output_shapes = [node_shapes[out][1:] for out in self.output_nodes]
        self._output_shape = output_shapes[0] if len(output_shapes) == 1 else output_shapes
        
        # Collecter toutes les couches trainables
        self.layers = [node.layer for node in self._forward_order if node.layer.trainable]
        self.trainable_layers = self.layers.copy()
        
        self.initialized = True
        return self._output_shape
    
    def forward(self, x: Union[np.ndarray, List[np.ndarray]], 
                training: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Propagation avant à travers le graphe.
        
        Args:
            x: Input(s) du modèle
            training: Mode entraînement
            
        Returns:
            Sortie(s) du modèle
        """
        if not self.initialized:
            # Auto-initialisation
            if isinstance(x, list):
                input_shapes = [xi.shape[1:] for xi in x]
            else:
                input_shapes = x.shape[1:]
            self.initialize(input_shapes)
        
        # S'assurer que le graphe est trié
        if not self._topology_sorted:
            self._topological_sort()
        
        # Préparer les inputs
        if not isinstance(x, list):
            x = [x]
        
        if len(x) != len(self.input_nodes):
            raise ValueError(f"Expected {len(self.input_nodes)} inputs, got {len(x)}")
        
        # Stocker les inputs
        for inp_node, xi in zip(self.input_nodes, x):
            inp_node.output_tensor = xi
        
        # Propagation avant dans l'ordre topologique
        for node in self._forward_order:
            if node in self.input_nodes:
                continue
            
            # Collecter les tenseurs d'input
            input_tensors = []
            for inp in node.inputs:
                if isinstance(inp, LayerNode):
                    input_tensors.append(inp.output_tensor)
                else:
                    input_tensors.append(inp)
            
            # Forward pass
            if getattr(node.layer, "functional_operation", None) == "multiply":
                node.layer.cached_inputs = input_tensors
            node.forward(*input_tensors, training=training)
        
        # Collecter les outputs
        if len(self.output_nodes) == 1:
            return self.output_nodes[0].output_tensor
        else:
            return [out_node.output_tensor for out_node in self.output_nodes]
    
    def backward(self, dout: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Rétropropagation à travers le graphe.
        
        Args:
            dout: Gradient(s) de la loss
            
        Returns:
            Gradient(s) par rapport aux inputs
        """
        # Préparer les gradients de sortie
        if not isinstance(dout, list):
            dout = [dout]
        
        if len(dout) != len(self.output_nodes):
            raise ValueError(f"Expected {len(self.output_nodes)} output gradients, got {len(dout)}")
        
        # Initialiser les gradients pour chaque nœud
        gradients = {node: None for node in self._all_nodes}
        
        for out_node, dout_i in zip(self.output_nodes, dout):
            gradients[out_node] = dout_i
        
        # Rétropropagation dans l'ordre inverse
        for node in self._backward_order:
            if gradients[node] is None:
                continue
            
            # Backward pass
            node_grads = node.backward(gradients[node])
            
            # Distribuer les gradients aux inputs
            if not isinstance(node_grads, (list, tuple)):
                node_grads = [node_grads]
            
            for inp, grad in zip(node.inputs, node_grads):
                if isinstance(inp, LayerNode):
                    if gradients[inp] is None:
                        gradients[inp] = grad
                    else:
                        # Accumuler les gradients si plusieurs chemins
                        gradients[inp] += grad
        
        # Collecter les gradients des inputs
        input_gradients = []
        for inp_node in self.input_nodes:
            if gradients[inp_node] is not None:
                input_gradients.append(gradients[inp_node])
            else:
                input_gradients.append(np.zeros_like(inp_node.output_tensor))
        
        if len(input_gradients) == 1:
            return input_gradients[0]
        else:
            return input_gradients
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Récupère tous les paramètres du modèle.
        
        Returns:
            Dictionnaire des paramètres
        """
        params = {}
        node_index = 0
        
        for node in self._forward_order:
            if node.layer.trainable:
                layer_params = node.layer.get_parameters()
                for key, value in layer_params.items():
                    params[f'node_{node_index}_{key}'] = value
                node_index += 1
        
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Définit les paramètres du modèle.
        
        Args:
            params: Dictionnaire des paramètres
        """
        node_index = 0
        
        for node in self._forward_order:
            if node.layer.trainable:
                layer_params = {}
                for key, value in params.items():
                    if key.startswith(f'node_{node_index}_'):
                        param_name = key.replace(f'node_{node_index}_', '')
                        layer_params[param_name] = value
                
                if layer_params:
                    node.layer.set_parameters(layer_params)
                node_index += 1
    
    def summary(self):
        """
        Affiche un résumé de l'architecture du modèle.
        """
        print(f"Model: {self.name} (Functional)")
        print("=" * 80)
        print(f"{'Layer (type)':<30} {'Output Shape':<25} {'Connected to':<25}")
        print("=" * 80)
        
        for node in self._forward_order:
            layer_type = node.layer.__class__.__name__
            layer_name = f"{node.name} ({layer_type})"
            
            # Shape de sortie
            output_shape = getattr(node.layer, 'output_shape', '?')
            
            # Nœuds connectés
            input_names = []
            for inp in node.inputs:
                if isinstance(inp, LayerNode):
                    input_names.append(inp.name)
                else:
                    input_names.append(str(inp))
            
            connected_to = ", ".join(input_names) if input_names else "(input)"
            
            print(f"{layer_name:<30} {str(output_shape):<25} {connected_to:<25}")
        
        print("=" * 80)
        
        # Compter les paramètres
        total_params = 0
        trainable_params = 0
        
        for node in self._forward_order:
            if hasattr(node.layer, 'parameters'):
                layer_params = sum(p.size for p in node.layer.parameters.values())
                total_params += layer_params
                if node.layer.trainable:
                    trainable_params += layer_params
        
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        
        if isinstance(self._input_shape, list):
            print(f"Multiple inputs: {len(self._input_shape)}")
            for i, shape in enumerate(self._input_shape):
                print(f"Input {i} shape: {shape}")
        else:
            print(f"Input shape: {self._input_shape}")
        
        if isinstance(self._output_shape, list):
            print(f"Multiple outputs: {len(self._output_shape)}")
            for i, shape in enumerate(self._output_shape):
                print(f"Output {i} shape: {shape}")
        else:
            print(f"Output shape: {self._output_shape}")
        
        print("=" * 80)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du modèle.
        
        Returns:
            Configuration
        """
        if not self._topology_sorted:
            self._topological_sort()
        node_indices = {node: index for index, node in enumerate(self._forward_order)}
        node_configs = []
        for node in self._forward_order:
            operation = "input" if node in self.input_nodes else getattr(
                node.layer, "functional_operation", None
            )
            node_configs.append({
                "name": node.name,
                "operation": operation,
                "layer_config": node.layer.get_config(),
                "input_indices": [
                    node_indices[inp] for inp in node.inputs
                    if isinstance(inp, LayerNode)
                ],
            })
        
        return {
            'name': self.name,
            'input_indices': [node_indices[inp] for inp in self.input_nodes],
            'output_indices': [node_indices[out] for out in self.output_nodes],
            'nodes': node_configs,
            'input_shape': self._input_shape,
            'output_shape': self._output_shape,
            'class_name': self.__class__.__name__
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Rebuild a functional graph from its serialized node list."""
        from .registry import registry

        nodes = []
        for node_config in config.get("nodes", []):
            operation = node_config.get("operation")
            layer_config = node_config["layer_config"]
            input_nodes = [nodes[index] for index in node_config.get("input_indices", [])]

            if operation == "input":
                node = Input(tuple(layer_config["shape"]), name=node_config.get("name"))
            elif operation == "add":
                node = add(input_nodes, name=node_config.get("name"))
            elif operation == "concatenate":
                node = concatenate(
                    input_nodes,
                    axis=layer_config.get("axis", -1),
                    name=node_config.get("name")
                )
            elif operation == "multiply":
                node = multiply(input_nodes, name=node_config.get("name"))
            else:
                layer = registry.deserialize_layer(layer_config)
                node = LayerNode(layer, node_config.get("name"))
                node(*input_nodes)
            nodes.append(node)

        model = cls(
            inputs=[nodes[index] for index in config["input_indices"]],
            outputs=[nodes[index] for index in config["output_indices"]],
            name=config.get("name")
        )
        if config.get("input_shape") is not None:
            model.initialize(config["input_shape"])
        return model

# Fonctions utilitaires pour créer des modèles complexes
def create_residual_block(input_node: LayerNode, 
                         filters: int,
                         kernel_size: int = 3,
                         strides: int = 1,
                         name: str = "residual_block") -> LayerNode:
    """
    Crée un bloc résiduel type ResNet.
    
    Args:
        input_node: Nœud d'input
        filters: Nombre de filtres
        kernel_size: Taille du kernel
        strides: Stride
        name: Nom du bloc
        
    Returns:
        Nœud de sortie du bloc
    """
    from .functional import Conv2D as Conv2D_func, BatchNorm2D as BatchNorm2D_func
    from .functional import ReLU as ReLU_func

    def channels(node):
        if isinstance(node, Input):
            return node.layer.shape[-1]
        layer = node.layer
        if hasattr(layer, "filters"):
            return layer.filters
        if hasattr(layer, "units"):
            return layer.units
        if node.inputs and isinstance(node.inputs[0], LayerNode):
            return channels(node.inputs[0])
        return None

    # Branche principale
    x = Conv2D_func(filters, kernel_size, strides=strides, padding='same', 
                   name=f"{name}_conv1")(input_node)
    x = BatchNorm2D_func(name=f"{name}_bn1")(x)
    x = ReLU_func(name=f"{name}_relu1")(x)
    x = Conv2D_func(filters, kernel_size, strides=1, padding='same',
                    name=f"{name}_conv2")(x)
    x = BatchNorm2D_func(name=f"{name}_bn2")(x)

    shortcut = input_node
    if strides != 1 or channels(input_node) != filters:
        shortcut = Conv2D_func(
            filters, 1, strides=strides, padding='same',
            name=f"{name}_conv_short"
        )(shortcut)
        shortcut = BatchNorm2D_func(name=f"{name}_bn_short")(shortcut)

    output = add([x, shortcut], name=f"{name}_add")
    return ReLU_func(name=f"{name}_relu_out")(output)

def create_inception_module(input_node: LayerNode,
                           filters_1x1: int,
                           filters_3x3_reduce: int,
                           filters_3x3: int,
                           filters_5x5_reduce: int,
                           filters_5x5: int,
                           filters_pool: int,
                           name: str = "inception") -> LayerNode:
    """
    Crée un module Inception type GoogLeNet.
    
    Args:
        input_node: Nœud d'input
        filters_1x1: Filtres pour la branche 1x1
        filters_3x3_reduce: Filtres de réduction pour 3x3
        filters_3x3: Filtres pour la branche 3x3
        filters_5x5_reduce: Filtres de réduction pour 5x5
        filters_5x5: Filtres pour la branche 5x5
        filters_pool: Filtres pour la branche pool
        name: Nom du module
        
    Returns:
        Nœud de sortie du module
    """
    # Branche 1x1
    branch1 = Conv2D(filters_1x1, 1, padding='same', name=f"{name}_1x1")(input_node)
    
    # Branche 3x3
    branch2 = Conv2D(filters_3x3_reduce, 1, padding='same', name=f"{name}_3x3_reduce")(input_node)
    branch2 = Conv2D(filters_3x3, 3, padding='same', name=f"{name}_3x3")(branch2)
    
    # Branche 5x5
    branch3 = Conv2D(filters_5x5_reduce, 1, padding='same', name=f"{name}_5x5_reduce")(input_node)
    branch3 = Conv2D(filters_5x5, 5, padding='same', name=f"{name}_5x5")(branch3)
    
    # Branche pool
    branch4 = MaxPool2D(3, strides=1, padding='same', name=f"{name}_pool")(input_node)
    branch4 = Conv2D(filters_pool, 1, padding='same', name=f"{name}_pool_proj")(branch4)
    
    # Concaténation
    output = concatenate([branch1, branch2, branch3, branch4], axis=-1, name=f"{name}_concat")
    
    return output