"""
Conteneurs de modèles pour des architectures complexes.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .model import Model

class ModelContainer(Model):
    """
    Conteneur pour combiner plusieurs modèles.
    """
    
    def __init__(self, models: List[Model], name: Optional[str] = None):
        """
        Initialise un conteneur de modèles.
        
        Args:
            models: Liste de modèles
            name: Nom du conteneur
        """
        super().__init__(name)
        self.models = models
        
        # Collecter toutes les couches
        for model in models:
            self.layers.extend(model.layers)
            self.trainable_layers.extend(model.trainable_layers)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagation avant à travers tous les modèles.
        
        Args:
            x: Input
            training: Mode entraînement
            
        Returns:
            Sortie
        """
        outputs = []
        for model in self.models:
            output = model.forward(x, training)
            outputs.append(output)
        
        # Pour l'instant, retourne seulement le premier
        # On pourrait ajouter une logique de fusion
        return outputs[0]
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Rétropropagation.
        """
        # Backward à travers tous les modèles
        gradients = []
        for model in self.models:
            grad = model.backward(dout)
            gradients.append(grad)
        
        return gradients[0]

class Parallel(Model):
    """
    Modèle parallèle - exécute plusieurs modèles en parallèle.
    """
    
    def __init__(self, branches: List[Model], merge_op: str = 'concat', 
                 name: Optional[str] = None):
        """
        Initialise un modèle parallèle.
        
        Args:
            branches: Liste de modèles (branches)
            merge_op: Opération de fusion ('concat', 'add', 'average')
            name: Nom du modèle
        """
        super().__init__(name)
        self.branches = branches
        self.merge_op = merge_op
        
        # Collecter toutes les couches
        for branch in branches:
            self.layers.extend(branch.layers)
            self.trainable_layers.extend(branch.trainable_layers)
    
    def initialize(self, input_shape: Tuple) -> Tuple:
        """
        Initialise toutes les branches.
        """
        output_shapes = []
        for branch in self.branches:
            shape = branch.initialize(input_shape)
            output_shapes.append(shape)
        
        # Calculer la shape de sortie fusionnée
        if self.merge_op == 'concat':
            # Concaténation sur le dernier axe
            total_features = sum(shape[-1] for shape in output_shapes)
            self._output_shape = output_shapes[0][:-1] + (total_features,)
        elif self.merge_op in ['add', 'average']:
            # Toutes les branches doivent avoir la même shape
            for shape in output_shapes[1:]:
                if shape != output_shapes[0]:
                    raise ValueError(f"All branches must have same shape for {self.merge_op}")
            self._output_shape = output_shapes[0]
        
        self.initialized = True
        return self._output_shape
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagation avant parallèle.
        """
        branch_outputs = []
        for branch in self.branches:
            output = branch.forward(x, training)
            branch_outputs.append(output)
        
        # Fusionner les sorties
        if self.merge_op == 'concat':
            return np.concatenate(branch_outputs, axis=-1)
        elif self.merge_op == 'add':
            result = branch_outputs[0]
            for out in branch_outputs[1:]:
                result += out
            return result
        elif self.merge_op == 'average':
            result = branch_outputs[0]
            for out in branch_outputs[1:]:
                result += out
            return result / len(branch_outputs)
        else:
            raise ValueError(f"Unknown merge operation: {self.merge_op}")
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Rétropropagation.
        """
        if self.merge_op == 'concat':
            # Split le gradient
            sizes = [branch._output_shape[-1] for branch in self.branches]
            splits = np.split(dout, np.cumsum(sizes)[:-1], axis=-1)
            
            gradients = []
            for branch, split in zip(self.branches, splits):
                grad = branch.backward(split)
                gradients.append(grad)
            
            # Combiner les gradients (moyenne)
            combined_grad = gradients[0]
            for grad in gradients[1:]:
                combined_grad += grad
            return combined_grad / len(gradients)
        
        elif self.merge_op in ['add', 'average']:
            # Gradient distribué également
            gradients = []
            for branch in self.branches:
                grad = branch.backward(dout)
                gradients.append(grad)
            
            if self.merge_op == 'add':
                # Pour 'add', le gradient est le même pour tous
                return gradients[0]
            else:  # 'average'
                # Pour 'average', diviser par le nombre de branches
                return gradients[0] / len(self.branches)