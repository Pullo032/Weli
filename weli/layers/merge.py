"""
Couches de fusion pour Weli.

À implémenter:
- Add: Addition élément par élément
- Multiply: Multiplication élément par élément
- Average: Moyenne élément par élément
- Maximum: Maximum élément par élément
- Concatenate: Concaténation
- Dot: Produit scalaire
- Subtract: Soustraction élément par élément
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from .base import Layer


class Add(Layer):
    """
    Couche d'addition élément par élément.
    
    Combine plusieurs tenseurs en les additionnant élément par élément.
    Tous les inputs doivent avoir la même shape (ou être broadcastables).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs
            
        Returns:
            Shape de sortie (identique à celle des inputs)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) < 2:
            raise ValueError("Add layer requires at least 2 inputs")
        
        # Vérifier que toutes les shapes sont compatibles
        base_shape = input_shapes[0]
        for i, shape in enumerate(input_shapes[1:], 1):
            try:
                # Tester le broadcasting
                np.broadcast_shapes(base_shape, shape)
            except ValueError:
                raise ValueError(
                    f"Input {i} shape {shape} is not broadcastable with input 0 shape {base_shape}"
                )
        
        self.input_shapes = input_shapes
        # La shape de sortie est la shape résultante du broadcasting
        self.output_shape = np.broadcast_shapes(*input_shapes)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: addition élément par élément.
        
        Args:
            inputs: Liste des tenseurs à additionner
            
        Returns:
            Tenseur résultant de l'addition
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) < 2:
            raise ValueError("Add layer requires at least 2 inputs")
        
        self.inputs = inputs
        
        # Addition élément par élément avec broadcasting
        result = inputs[0]
        for i in range(1, len(inputs)):
            result = result + inputs[i]
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: distribue le gradient à tous les inputs.
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        gradients = []
        for inp in self.inputs:
            # Le gradient est distribué à tous les inputs
            # Si les shapes diffèrent, on doit réduire le gradient
            grad = dout.copy()
            
            # Réduire les dimensions ajoutées par broadcasting
            for axis in range(len(grad.shape) - len(inp.shape)):
                grad = np.sum(grad, axis=0)
            
            # Réduire les dimensions où la taille est 1
            for axis in range(len(inp.shape)):
                if inp.shape[axis] == 1 and grad.shape[axis] > 1:
                    grad = np.sum(grad, axis=axis, keepdims=True)
            
            gradients.append(grad)
        
        return gradients


class Multiply(Layer):
    """
    Couche de multiplication élément par élément.
    
    Combine plusieurs tenseurs en les multipliant élément par élément.
    Tous les inputs doivent avoir la même shape (ou être broadcastables).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs
            
        Returns:
            Shape de sortie (identique à celle des inputs)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) < 2:
            raise ValueError("Multiply layer requires at least 2 inputs")
        
        # Vérifier que toutes les shapes sont compatibles
        base_shape = input_shapes[0]
        for i, shape in enumerate(input_shapes[1:], 1):
            try:
                np.broadcast_shapes(base_shape, shape)
            except ValueError:
                raise ValueError(
                    f"Input {i} shape {shape} is not broadcastable with input 0 shape {base_shape}"
                )
        
        self.input_shapes = input_shapes
        self.output_shape = np.broadcast_shapes(*input_shapes)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: multiplication élément par élément.
        
        Args:
            inputs: Liste des tenseurs à multiplier
            
        Returns:
            Tenseur résultant de la multiplication
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) < 2:
            raise ValueError("Multiply layer requires at least 2 inputs")
        
        self.inputs = inputs
        
        # Multiplication élément par élément avec broadcasting
        result = inputs[0]
        for i in range(1, len(inputs)):
            result = result * inputs[i]
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: calcule le gradient pour chaque input.
        
        Pour input i: gradient = dout * produit de tous les autres inputs
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        gradients = []
        for i, inp in enumerate(self.inputs):
            # Gradient pour l'input i: dout * produit des autres inputs
            grad = dout.copy()
            
            for j, other_inp in enumerate(self.inputs):
                if j != i:
                    grad = grad * other_inp
            
            # Réduire les dimensions ajoutées par broadcasting
            for axis in range(len(grad.shape) - len(inp.shape)):
                grad = np.sum(grad, axis=0)
            
            # Réduire les dimensions où la taille est 1
            for axis in range(len(inp.shape)):
                if inp.shape[axis] == 1 and grad.shape[axis] > 1:
                    grad = np.sum(grad, axis=axis, keepdims=True)
            
            gradients.append(grad)
        
        return gradients


class Average(Layer):
    """
    Couche de moyenne élément par élément.
    
    Combine plusieurs tenseurs en calculant leur moyenne élément par élément.
    Tous les inputs doivent avoir la même shape (ou être broadcastables).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs
            
        Returns:
            Shape de sortie (identique à celle des inputs)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) < 2:
            raise ValueError("Average layer requires at least 2 inputs")
        
        # Vérifier que toutes les shapes sont compatibles
        base_shape = input_shapes[0]
        for i, shape in enumerate(input_shapes[1:], 1):
            try:
                np.broadcast_shapes(base_shape, shape)
            except ValueError:
                raise ValueError(
                    f"Input {i} shape {shape} is not broadcastable with input 0 shape {base_shape}"
                )
        
        self.input_shapes = input_shapes
        self.output_shape = np.broadcast_shapes(*input_shapes)
        self.num_inputs = len(input_shapes)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: moyenne élément par élément.
        
        Args:
            inputs: Liste des tenseurs à moyenner
            
        Returns:
            Tenseur résultant de la moyenne
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) < 2:
            raise ValueError("Average layer requires at least 2 inputs")
        
        self.inputs = inputs
        
        # Somme avec broadcasting
        result = inputs[0]
        for i in range(1, len(inputs)):
            result = result + inputs[i]
        
        # Division par le nombre d'inputs
        result = result / len(inputs)
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: distribue le gradient divisé par le nombre d'inputs.
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        gradients = []
        for inp in self.inputs:
            # Gradient divisé par le nombre d'inputs
            grad = dout / len(self.inputs)
            
            # Réduire les dimensions ajoutées par broadcasting
            for axis in range(len(grad.shape) - len(inp.shape)):
                grad = np.sum(grad, axis=0)
            
            # Réduire les dimensions où la taille est 1
            for axis in range(len(inp.shape)):
                if inp.shape[axis] == 1 and grad.shape[axis] > 1:
                    grad = np.sum(grad, axis=axis, keepdims=True)
            
            gradients.append(grad)
        
        return gradients


class Maximum(Layer):
    """
    Couche de maximum élément par élément.
    
    Combine plusieurs tenseurs en prenant le maximum élément par élément.
    Tous les inputs doivent avoir la même shape (ou être broadcastables).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
        self.inputs = None
        self.max_indices = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs
            
        Returns:
            Shape de sortie (identique à celle des inputs)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) < 2:
            raise ValueError("Maximum layer requires at least 2 inputs")
        
        # Vérifier que toutes les shapes sont compatibles
        base_shape = input_shapes[0]
        for i, shape in enumerate(input_shapes[1:], 1):
            try:
                np.broadcast_shapes(base_shape, shape)
            except ValueError:
                raise ValueError(
                    f"Input {i} shape {shape} is not broadcastable with input 0 shape {base_shape}"
                )
        
        self.input_shapes = input_shapes
        self.output_shape = np.broadcast_shapes(*input_shapes)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: maximum élément par élément.
        
        Args:
            inputs: Liste des tenseurs
            
        Returns:
            Tenseur résultant du maximum
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) < 2:
            raise ValueError("Maximum layer requires at least 2 inputs")
        
        self.inputs = inputs
        
        # Stack les inputs pour trouver le maximum
        # Utiliser np.maximum.reduce pour gérer le broadcasting
        result = inputs[0]
        for i in range(1, len(inputs)):
            result = np.maximum(result, inputs[i])
        
        # Stocker les indices des maximums pour la backward pass
        # Pour chaque position, trouver quel input a le maximum
        stacked = np.stack([np.broadcast_to(inp, self.output_shape) for inp in inputs], axis=0)
        self.max_indices = np.argmax(stacked, axis=0)
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: le gradient va uniquement aux inputs qui ont le maximum.
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None or self.max_indices is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        gradients = []
        for i, inp in enumerate(self.inputs):
            # Le gradient va uniquement aux positions où cet input est le maximum
            grad = np.zeros_like(np.broadcast_to(inp, self.output_shape))
            mask = (self.max_indices == i)
            grad[mask] = dout[mask]
            
            # Réduire les dimensions ajoutées par broadcasting
            for axis in range(len(grad.shape) - len(inp.shape)):
                grad = np.sum(grad, axis=0)
            
            # Réduire les dimensions où la taille est 1
            for axis in range(len(inp.shape)):
                if inp.shape[axis] == 1 and grad.shape[axis] > 1:
                    grad = np.sum(grad, axis=axis, keepdims=True)
            
            gradients.append(grad)
        
        return gradients


class Concatenate(Layer):
    """
    Couche de concaténation.
    
    Combine plusieurs tenseurs en les concaténant le long d'un axe spécifié.
    Tous les inputs doivent avoir la même shape sauf sur l'axe de concaténation.
    """
    
    def __init__(self, axis: int = -1, name: Optional[str] = None):
        """
        Initialise la couche de concaténation.
        
        Args:
            axis: Axe le long duquel concaténer (par défaut: -1, dernier axe)
            name: Nom de la couche
        """
        super().__init__(name)
        self.trainable = False
        self.axis = axis
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs
            
        Returns:
            Shape de sortie
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) < 2:
            raise ValueError("Concatenate layer requires at least 2 inputs")
        
        # Normaliser l'axe
        axis = self.axis if self.axis >= 0 else len(input_shapes[0]) + self.axis
        
        if axis < 0 or axis >= len(input_shapes[0]):
            raise ValueError(f"Axis {self.axis} is out of range for input shape {input_shapes[0]}")
        
        # Vérifier que toutes les shapes sont compatibles sauf sur l'axe de concat
        base_shape = list(input_shapes[0])
        
        for i, shape in enumerate(input_shapes[1:], 1):
            shape = list(shape)
            if len(shape) != len(base_shape):
                raise ValueError(
                    f"All inputs must have same rank, got {len(base_shape)} and {len(shape)}"
                )
            
            for j in range(len(shape)):
                if j != axis and shape[j] != base_shape[j]:
                    raise ValueError(
                        f"Input {i} incompatible on axis {j}: {shape[j]} != {base_shape[j]}"
                    )
        
        # Calculer la shape de sortie
        output_shape = base_shape.copy()
        output_shape[axis] = sum(shape[axis] for shape in input_shapes)
        
        self.input_shapes = input_shapes
        self.output_shape = tuple(output_shape)
        self.axis = axis
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: concaténation le long de l'axe spécifié.
        
        Args:
            inputs: Liste des tenseurs à concaténer
            
        Returns:
            Tenseur résultant de la concaténation
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) < 2:
            raise ValueError("Concatenate layer requires at least 2 inputs")
        
        self.inputs = inputs
        
        # Concaténation
        result = np.concatenate(inputs, axis=self.axis)
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: split le gradient selon l'axe de concaténation.
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        # Calculer les tailles sur l'axe de concaténation
        sizes = [inp.shape[self.axis] for inp in self.inputs]
        
        # Split le gradient
        splits = np.split(dout, np.cumsum(sizes)[:-1], axis=self.axis)
        
        return splits


class Dot(Layer):
    """
    Couche de produit scalaire (dot product).
    
    Calcule le produit scalaire entre deux tenseurs le long d'un axe spécifié.
    """
    
    def __init__(self, axes: Union[int, Tuple[int, int]] = -1, normalize: bool = False, 
                 name: Optional[str] = None):
        """
        Initialise la couche de produit scalaire.
        
        Args:
            axes: Axe(s) pour le produit scalaire. 
                  Si int: même axe pour les deux inputs.
                  Si tuple: (axe_input1, axe_input2)
            normalize: Si True, normalise les vecteurs avant le produit scalaire
            name: Nom de la couche
        """
        super().__init__(name)
        self.trainable = False
        self.axes = axes
        self.normalize = normalize
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs (doit être exactement 2)
            
        Returns:
            Shape de sortie
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) != 2:
            raise ValueError("Dot layer requires exactly 2 inputs")
        
        shape1, shape2 = input_shapes
        
        # Normaliser les axes
        if isinstance(self.axes, int):
            axes = (self.axes, self.axes)
        else:
            axes = self.axes
        
        axis1 = axes[0] if axes[0] >= 0 else len(shape1) + axes[0]
        axis2 = axes[1] if axes[1] >= 0 else len(shape2) + axes[1]
        
        if shape1[axis1] != shape2[axis2]:
            raise ValueError(
                f"Dimension mismatch: input1 axis {axis1} has size {shape1[axis1]}, "
                f"but input2 axis {axis2} has size {shape2[axis2]}"
            )
        
        # Calculer la shape de sortie
        # Retirer les axes utilisés pour le produit scalaire
        output_shape = list(shape1)
        output_shape.pop(axis1)
        
        # Ajouter les dimensions restantes de shape2
        shape2_list = list(shape2)
        shape2_list.pop(axis2)
        
        # Broadcasting des dimensions restantes
        for dim in shape2_list:
            if len(output_shape) == 0 or output_shape[-1] == 1 or dim == 1:
                if dim != 1:
                    output_shape.append(dim)
            elif output_shape[-1] != dim and dim != 1:
                raise ValueError(f"Broadcasting not possible: {output_shape[-1]} and {dim}")
        
        self.input_shapes = input_shapes
        self.axes = (axis1, axis2)
        self.output_shape = tuple(output_shape) if output_shape else (1,)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: produit scalaire.
        
        Args:
            inputs: Liste de 2 tenseurs
            
        Returns:
            Tenseur résultant du produit scalaire
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) != 2:
            raise ValueError("Dot layer requires exactly 2 inputs")
        
        self.inputs = inputs
        x1, x2 = inputs
        axis1, axis2 = self.axes
        
        # Normalisation si demandée
        if self.normalize:
            # Normaliser le long de l'axe spécifié
            x1_norm = np.linalg.norm(x1, axis=axis1, keepdims=True)
            x2_norm = np.linalg.norm(x2, axis=axis2, keepdims=True)
            x1 = x1 / (x1_norm + 1e-8)
            x2 = x2 / (x2_norm + 1e-8)
        
        # Produit scalaire le long des axes spécifiés
        # Utiliser einsum pour plus de flexibilité
        # Créer les indices pour einsum
        ndim1, ndim2 = x1.ndim, x2.ndim
        
        # Indices pour x1
        subscripts1 = list(range(ndim1))
        subscripts1[axis1] = 'k'  # Axe à contracter
        
        # Indices pour x2
        subscripts2 = list(range(ndim1, ndim1 + ndim2))
        subscripts2[axis2] = 'k'  # Axe à contracter
        
        # Indices de sortie
        subscripts_out = [i for i in subscripts1 if i != 'k'] + \
                        [i for i in subscripts2 if i != 'k']
        
        # Convertir en string pour einsum
        sub1_str = ''.join(str(s) if isinstance(s, int) else s for s in subscripts1)
        sub2_str = ''.join(str(s) if isinstance(s, int) else s for s in subscripts2)
        sub_out_str = ''.join(str(s) if isinstance(s, int) else s for s in subscripts_out)
        
        einsum_str = f"{sub1_str},{sub2_str}->{sub_out_str}"
        
        result = np.einsum(einsum_str, x1, x2)
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: calcule le gradient pour chaque input.
        
        Pour un produit scalaire le long d'axes spécifiés:
        - Gradient pour x1: dout contracté avec x2
        - Gradient pour x2: dout contracté avec x1
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        x1, x2 = self.inputs
        axis1, axis2 = self.axes
        
        # Recalculer les valeurs normalisées si nécessaire
        if self.normalize:
            x1_norm = np.linalg.norm(x1, axis=axis1, keepdims=True)
            x2_norm = np.linalg.norm(x2, axis=axis2, keepdims=True)
            x1_normalized = x1 / (x1_norm + 1e-8)
            x2_normalized = x2 / (x2_norm + 1e-8)
        else:
            x1_normalized = x1
            x2_normalized = x2
        
        # Pour le backward, on inverse l'opération einsum
        # Forward: einsum('...k,...k->...', x1, x2) le long de axis1 et axis2
        # Backward grad1: einsum('...,...k->...k', dout, x2) 
        # Backward grad2: einsum('...,...k->...k', dout, x1)
        
        # Construire les strings einsum pour le backward
        ndim1, ndim2 = x1.ndim, x2.ndim
        
        # Pour grad1: on doit réintroduire l'axe axis1
        # dout a les dimensions non-contractées de x1 et x2
        # On veut multiplier dout par x2_normalized et réintroduire l'axe axis1
        
        # Approche: utiliser einsum_path pour construire l'opération inverse
        # Plus simple: utiliser une approche directe avec expand_dims et multiplication
        
        # Étendre dout pour qu'il ait les bonnes dimensions
        # dout doit être multiplié par x2_normalized, puis on réintroduit l'axe axis1
        
        # Pour grad1: on multiplie dout (broadcasté) par x2_normalized
        # Puis on somme sur les dimensions supplémentaires de x2 qui ne sont pas dans x1
        
        # Approche simplifiée: utiliser np.tensordot ou une multiplication avec broadcasting
        # Étendre dout pour correspondre aux dimensions de x1 (sauf axis1)
        grad1 = dout.copy()
        
        # Insérer une dimension à la position axis1
        grad1 = np.expand_dims(grad1, axis=axis1)
        
        # Multiplier par x2_normalized avec broadcasting
        # On doit aligner les dimensions correctement
        # Pour cela, on utilise une approche plus directe
        
        # Utiliser une multiplication avec broadcasting manuel
        # grad1 doit avoir la shape de x1
        # On multiplie dout (avec dimensions étendues) par x2_normalized
        
        # Solution: utiliser einsum de manière inverse
        # Construire la string einsum pour le backward
        
        # Créer les indices pour einsum
        # Pour grad1: on veut contracter dout avec x2_normalized pour obtenir x1
        # Les dimensions de dout correspondent aux dimensions non-contractées
        
        # Approche pratique: utiliser une multiplication simple avec gestion du broadcasting
        # Étendre dout pour qu'il puisse être multiplié par x2
        grad1 = np.expand_dims(dout, axis=axis1)
        
        # Multiplier par x2_normalized
        # Ajuster le broadcasting pour que ça fonctionne
        for _ in range(grad1.ndim - x1.ndim):
            # Réduire les dimensions supplémentaires
            grad1 = np.sum(grad1, axis=-1)
        
        # Multiplier par x2_normalized avec broadcasting
        grad1 = grad1 * x2_normalized
        
        # Ajuster pour correspondre exactement à la shape de x1
        while grad1.shape != x1.shape:
            # Si grad1 a plus de dimensions, réduire
            if grad1.ndim > x1.ndim:
                # Trouver une dimension de taille 1 à réduire
                for i in range(grad1.ndim):
                    if i < len(x1.shape) and grad1.shape[i] == 1 and x1.shape[i] != 1:
                        continue
                    if i >= len(x1.shape) or (i < len(x1.shape) and grad1.shape[i] == 1):
                        grad1 = np.squeeze(grad1, axis=i)
                        break
                else:
                    # Réduire la dernière dimension
                    grad1 = np.sum(grad1, axis=-1)
            else:
                break
        
        # Même chose pour grad2
        grad2 = np.expand_dims(dout, axis=axis2)
        
        for _ in range(grad2.ndim - x2.ndim):
            grad2 = np.sum(grad2, axis=-1)
        
        grad2 = grad2 * x1_normalized
        
        while grad2.shape != x2.shape:
            if grad2.ndim > x2.ndim:
                for i in range(grad2.ndim):
                    if i < len(x2.shape) and grad2.shape[i] == 1 and x2.shape[i] != 1:
                        continue
                    if i >= len(x2.shape) or (i < len(x2.shape) and grad2.shape[i] == 1):
                        grad2 = np.squeeze(grad2, axis=i)
                        break
                else:
                    grad2 = np.sum(grad2, axis=-1)
            else:
                break
        
        return [grad1, grad2]


class Subtract(Layer):
    """
    Couche de soustraction élément par élément.
    
    Soustrait le deuxième tenseur du premier (input[0] - input[1]).
    Les deux inputs doivent avoir la même shape (ou être broadcastables).
    """
    
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.trainable = False
        self.inputs = None
    
    def initialize(self, input_shapes: Union[Tuple, List[Tuple]]) -> Tuple:
        """
        Initialise la couche.
        
        Args:
            input_shapes: Liste des shapes des inputs (doit être exactement 2)
            
        Returns:
            Shape de sortie (identique à celle des inputs)
        """
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        
        if len(input_shapes) != 2:
            raise ValueError("Subtract layer requires exactly 2 inputs")
        
        shape1, shape2 = input_shapes
        
        # Vérifier que les shapes sont compatibles
        try:
            np.broadcast_shapes(shape1, shape2)
        except ValueError:
            raise ValueError(
                f"Input shapes {shape1} and {shape2} are not broadcastable"
            )
        
        self.input_shapes = input_shapes
        self.output_shape = np.broadcast_shapes(shape1, shape2)
        return self.output_shape
    
    def forward(self, inputs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Propagation avant: soustraction élément par élément.
        
        Args:
            inputs: Liste de 2 tenseurs (input[0] - input[1])
            
        Returns:
            Tenseur résultant de la soustraction
        """
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        if len(inputs) != 2:
            raise ValueError("Subtract layer requires exactly 2 inputs")
        
        self.inputs = inputs
        
        # Soustraction avec broadcasting
        result = inputs[0] - inputs[1]
        
        self.output = result
        return self.output
    
    def backward(self, dout: np.ndarray) -> List[np.ndarray]:
        """
        Rétropropagation: 
        - Gradient pour input[0]: dout
        - Gradient pour input[1]: -dout
        
        Args:
            dout: Gradient de la loss
            
        Returns:
            Liste des gradients pour chaque input
        """
        if self.inputs is None:
            raise RuntimeError("Forward pass must be called before backward")
        
        inp1, inp2 = self.inputs
        gradients = []
        
        # Gradient pour input[0]: dout
        grad1 = dout.copy()
        for axis in range(len(grad1.shape) - len(inp1.shape)):
            grad1 = np.sum(grad1, axis=0)
        for axis in range(len(inp1.shape)):
            if inp1.shape[axis] == 1 and grad1.shape[axis] > 1:
                grad1 = np.sum(grad1, axis=axis, keepdims=True)
        gradients.append(grad1)
        
        # Gradient pour input[1]: -dout
        grad2 = -dout.copy()
        for axis in range(len(grad2.shape) - len(inp2.shape)):
            grad2 = np.sum(grad2, axis=0)
        for axis in range(len(inp2.shape)):
            if inp2.shape[axis] == 1 and grad2.shape[axis] > 1:
                grad2 = np.sum(grad2, axis=axis, keepdims=True)
        gradients.append(grad2)
        
        return gradients
