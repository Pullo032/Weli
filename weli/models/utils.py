"""
Utilitaires pour les modèles.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json

def model_to_json(model: 'Model', filepath: Optional[str] = None) -> Dict[str, Any]:
    """
    Convertit un modèle en dictionnaire JSON-serializable.
    
    Args:
        model: Modèle à convertir
        filepath: Si fourni, sauvegarde dans un fichier
        
    Returns:
        Dictionnaire de configuration
    """
    config = {
        'model_name': model.name,
        'model_class': model.__class__.__name__,
        'layers': []
    }
    
    for layer in model.layers:
        layer_config = layer.get_config()
        config['layers'].append(layer_config)
    
    if filepath:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Model configuration saved to {filepath}")
    
    return config

def model_from_json(config: Dict[str, Any]) -> 'Model':
    """
    Crée un modèle à partir d'une configuration JSON.
    
    Args:
        config: Configuration du modèle
        
    Returns:
        Modèle créé
    """
    # Cette fonction nécessiterait un registry des couches
    # Pour l'instant, c'est un placeholder
    from .sequential import Sequential
    
    model = Sequential(name=config['model_name'])
    
    # Note: La création des couches à partir des configs
    # nécessiterait une factory de couches
    
    return model

def count_parameters(model: 'Model') -> Dict[str, int]:
    """
    Compte les paramètres du modèle.
    
    Args:
        model: Modèle à analyser
        
    Returns:
        Dictionnaire avec le nombre total et trainable
    """
    total_params = 0
    trainable_params = 0
    
    for layer in model.layers:
        if hasattr(layer, 'parameters'):
            layer_params = sum(p.size for p in layer.parameters.values())
            total_params += layer_params
            
            if layer.trainable:
                trainable_params += layer_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }

def print_model_summary(model: 'Model'):
    """
    Affiche un résumé détaillé du modèle.
    
    Args:
        model: Modèle à résumer
    """
    print(f"{'='*60}")
    print(f"Model: {model.name}")
    print(f"Type: {model.__class__.__name__}")
    print(f"{'='*60}")
    
    param_counts = count_parameters(model)
    print(f"Total params: {param_counts['total_params']:,}")
    print(f"Trainable params: {param_counts['trainable_params']:,}")
    print(f"Non-trainable params: {param_counts['non_trainable_params']:,}")
    print(f"{'='*60}")
    
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    print(f"{'-'*60}")
    
    total_params = 0
    current_shape = getattr(model, '_input_shape', 'Unknown')
    
    for i, layer in enumerate(model.layers):
        layer_name = f"{layer.name} ({layer.__class__.__name__})"
        
        # Calculer les paramètres
        layer_params = 0
        if hasattr(layer, 'parameters'):
            for param in layer.parameters.values():
                layer_params += param.size
        
        total_params += layer_params
        
        # Essayer d'obtenir la shape de sortie
        output_shape = getattr(layer, 'output_shape', current_shape)
        
        print(f"{layer_name:<25} {str(output_shape):<20} {layer_params:<15,}")
        
        current_shape = output_shape
    
    print(f"{'='*60}")
    print(f"Input shape: {getattr(model, '_input_shape', 'Unknown')}")
    print(f"Output shape: {getattr(model, '_output_shape', 'Unknown')}")
    print(f"{'='*60}")

def get_layer_by_name(model: 'Model', name: str) -> Optional[Any]:
    """
    Récupère une couche par son nom.
    
    Args:
        model: Modèle
        name: Nom de la couche
        
    Returns:
        Couche ou None si non trouvée
    """
    for layer in model.layers:
        if layer.name == name:
            return layer
    return None

def get_layer_output(model: 'Model', 
                     x: np.ndarray, 
                     layer_index: int) -> np.ndarray:
    """
    Récupère la sortie d'une couche spécifique.
    
    Args:
        model: Modèle
        x: Input
        layer_index: Index de la couche
        
    Returns:
        Sortie de la couche
    """
    if layer_index < 0 or layer_index >= len(model.layers):
        raise ValueError(f"Layer index {layer_index} out of range")
    
    # Propagation avant jusqu'à la couche spécifiée
    output = x
    for i in range(layer_index + 1):
        output = model.layers[i].forward(output)
    
    return output