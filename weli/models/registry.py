"""
Registry pour la sérialisation/désérialisation des modèles et couches.
"""
from typing import Dict, Type, Any, Callable
import importlib

class ModelRegistry:
    """
    Registry pour gérer les classes de modèles et couches.
    """
    
    def __init__(self):
        self._models = {}
        self._layers = {}
        self._custom_objects = {}
    
    def register_model(self, name: str, model_class: Type):
        """
        Enregistre une classe de modèle.
        
        Args:
            name: Nom de la classe
            model_class: Classe du modèle
        """
        self._models[name] = model_class
    
    def register_layer(self, name: str, layer_class: Type):
        """
        Enregistre une classe de couche.
        
        Args:
            name: Nom de la classe
            layer_class: Classe de la couche
        """
        self._layers[name] = layer_class
    
    def register_custom_object(self, name: str, obj: Any):
        """
        Enregistre un objet personnalisé.
        
        Args:
            name: Nom de l'objet
            obj: Objet à enregistrer
        """
        self._custom_objects[name] = obj
    
    def get_model(self, name: str) -> Type:
        """
        Récupère une classe de modèle.
        
        Args:
            name: Nom de la classe
            
        Returns:
            Classe du modèle
        """
        if name in self._models:
            return self._models[name]
        
        # Essayer d'importer dynamiquement
        try:
            module_name, class_name = name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            self.register_model(name, model_class)
            return model_class
        except (ImportError, AttributeError):
            raise ValueError(f"Model class '{name}' not found in registry")
    
    def get_layer(self, name: str) -> Type:
        """
        Récupère une classe de couche.
        
        Args:
            name: Nom de la classe
            
        Returns:
            Classe de la couche
        """
        if name in self._layers:
            return self._layers[name]
        
        # Essayer d'importer dynamiquement
        try:
            module_name, class_name = name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            layer_class = getattr(module, class_name)
            self.register_layer(name, layer_class)
            return layer_class
        except (ImportError, AttributeError):
            raise ValueError(f"Layer class '{name}' not found in registry")
    
    def get_custom_object(self, name: str) -> Any:
        """
        Récupère un objet personnalisé.
        
        Args:
            name: Nom de l'objet
            
        Returns:
            Objet
        """
        if name in self._custom_objects:
            return self._custom_objects[name]
        raise ValueError(f"Custom object '{name}' not found in registry")
    
    def deserialize_model(self, config: Dict[str, Any]) -> Any:
        """
        Désérialise un modèle à partir de sa configuration.
        
        Args:
            config: Configuration du modèle
            
        Returns:
            Modèle désérialisé
        """
        model_class_name = config.get('class_name')
        if not model_class_name:
            raise ValueError("Model configuration must contain 'class_name'")
        
        model_class = self.get_model(model_class_name)
        
        if model_class_name == 'Sequential':
            from .sequential import Sequential
            model = Sequential(name=config.get('name'))
            
            # Créer les couches
            for layer_config in config.get('layer_configs', []):
                layer = self.deserialize_layer(layer_config)
                model.add(layer)
            
            # Charger les paramètres si disponibles
            if 'parameters' in config:
                model.set_parameters(config['parameters'])
            
            return model
        
        elif model_class_name == 'Functional':
            # Plus complexe - nécessite de reconstruire le graphe
            # Pour l'instant, retourner un modèle vide
            from .functional import Functional
            return Functional(inputs=[], outputs=[], name=config.get('name'))
        
        else:
            # Pour les autres modèles, utiliser from_config si disponible
            if hasattr(model_class, 'from_config'):
                return model_class.from_config(config)
            else:
                return model_class(**config)
    
    def deserialize_layer(self, config: Dict[str, Any]) -> Any:
        """
        Désérialise une couche à partir de sa configuration.
        
        Args:
            config: Configuration de la couche
            
        Returns:
            Couche désérialisée
        """
        layer_class_name = config.get('class_name')
        if not layer_class_name:
            raise ValueError("Layer configuration must contain 'class_name'")
        
        layer_class = self.get_layer(layer_class_name)
        
        # Retirer les champs spéciaux
        layer_config = config.copy()
        layer_config.pop('class_name', None)
        layer_config.pop('name', None)
        
        # Créer la couche
        layer = layer_class(**layer_config)
        
        # Définir le nom si spécifié
        if 'name' in config:
            layer.name = config['name']
        
        return layer

# Instance globale du registry
registry = ModelRegistry()

# Enregistrer les classes par défaut
def register_default_classes():
    """Enregistre les classes par défaut dans le registry."""
    from .model import Model
    from .sequential import Sequential
    from .functional import Functional
    
    # Enregistrer les modèles
    registry.register_model('Model', Model)
    registry.register_model('Sequential', Sequential)
    registry.register_model('Functional', Functional)
    
    # Enregistrer les couches (importées dynamiquement)
    try:
        from ..layers import (
            Dense, Conv2D, MaxPool2D, Flatten, Dropout,
            BatchNorm1D, BatchNorm2D, ReLU, Sigmoid, Tanh,
            Softmax, LeakyReLU, ELU, SimpleRNN, LSTM, GRU
        )
        
        registry.register_layer('Dense', Dense)
        registry.register_layer('Conv2D', Conv2D)
        registry.register_layer('MaxPool2D', MaxPool2D)
        registry.register_layer('Flatten', Flatten)
        registry.register_layer('Dropout', Dropout)
        registry.register_layer('BatchNorm1D', BatchNorm1D)
        registry.register_layer('BatchNorm2D', BatchNorm2D)
        registry.register_layer('ReLU', ReLU)
        registry.register_layer('Sigmoid', Sigmoid)
        registry.register_layer('Tanh', Tanh)
        registry.register_layer('Softmax', Softmax)
        registry.register_layer('LeakyReLU', LeakyReLU)
        registry.register_layer('ELU', ELU)
        registry.register_layer('SimpleRNN', SimpleRNN)
        registry.register_layer('LSTM', LSTM)
        registry.register_layer('GRU', GRU)
        
    except ImportError:
        pass

# Appeler l'enregistrement
register_default_classes()