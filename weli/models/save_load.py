"""
Fonctions avancées de sauvegarde et chargement de modèles.
"""
import pickle
import json
import numpy as np
from typing import Dict, Any, Union
import os
from .registry import registry

def save_model(model: 'Model', filepath: str, 
               save_weights_only: bool = False,
               include_optimizer: bool = False,
               optimizer=None):
    """
    Sauvegarde un modèle dans un fichier.
    
    Args:
        model: Modèle à sauvegarder
        filepath: Chemin du fichier
        save_weights_only: Si True, sauvegarde seulement les poids
        include_optimizer: Si True, inclut l'état de l'optimiseur
        optimizer: Optimiseur à sauvegarder
    """
    if save_weights_only:
        # Sauvegarde seulement les poids
        weights_file = filepath + '.npz' if not filepath.endswith('.npz') else filepath
        params = model.get_parameters()
        np.savez(weights_file, **params)
        print(f"Model weights saved to {weights_file}")
    
    else:
        # Sauvegarde complète du modèle
        model_data = {
            'model_config': model.get_config(),
            'parameters': model.get_parameters(),
            'metadata': {
                'weli_version': '0.1.0',
                'model_class': model.__class__.__name__,
                'input_shape': getattr(model, '_input_shape', None),
                'output_shape': getattr(model, '_output_shape', None)
            }
        }
        
        # Inclure l'optimiseur si demandé
        if include_optimizer and optimizer is not None:
            model_data['optimizer_state'] = {
                'class_name': optimizer.__class__.__name__,
                'config': optimizer.get_config(),
                'weights': optimizer.get_weights()
            }
        
        # Sauvegarde dans un fichier
        if filepath.endswith('.json'):
            # Sauvegarde JSON (configuration seulement)
            with open(filepath, 'w') as f:
                json.dump(model_data['model_config'], f, indent=2, default=str)
        else:
            # Sauvegarde binaire complète
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

def load_model(filepath: str, 
               custom_objects: Dict[str, Any] = None) -> 'Model':
    """
    Charge un modèle depuis un fichier.
    
    Args:
        filepath: Chemin du fichier
        custom_objects: Objets personnalisés pour la désérialisation
        
    Returns:
        Modèle chargé
    """
    # Enregistrer les objets personnalisés
    if custom_objects:
        for name, obj in custom_objects.items():
            registry.register_custom_object(name, obj)
    
    if filepath.endswith('.json'):
        # Chargement depuis JSON (configuration seulement)
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        model = registry.deserialize_model(config)
    
    elif filepath.endswith('.npz'):
        # Chargement des poids seulement
        weights = np.load(filepath)
        
        # Créer un modèle vide (nécessite de connaître l'architecture)
        # Pour l'instant, retourner juste les poids
        # Dans la pratique, il faudrait avoir le modèle déjà créé
        raise ValueError("Loading from .npz requires a pre-existing model. "
                        "Use model.load_weights() instead.")
    
    else:
        # Chargement binaire complet
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Désérialiser le modèle
        model = registry.deserialize_model(model_data['model_config'])
        
        # Charger les paramètres
        if 'parameters' in model_data:
            model.set_parameters(model_data['parameters'])
    
    return model

def load_weights(model: 'Model', filepath: str):
    """
    Charge les poids dans un modèle existant.
    
    Args:
        model: Modèle à charger
        filepath: Chemin du fichier de poids
    """
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        params = {key: data[key] for key in data.files}
        model.set_parameters(params)
    else:
        raise ValueError(f"Unsupported weights file format: {filepath}")
    
    print(f"Weights loaded from {filepath}")

def save_model_architecture(model: 'Model', filepath: str):
    """
    Sauvegarde seulement l'architecture du modèle.
    
    Args:
        model: Modèle
        filepath: Chemin du fichier
    """
    config = model.get_config()
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Model architecture saved to {filepath}")

def model_to_dict(model: 'Model') -> Dict[str, Any]:
    """
    Convertit un modèle en dictionnaire.
    
    Args:
        model: Modèle
        
    Returns:
        Dictionnaire représentant le modèle
    """
    return {
        'name': model.name,
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
        'parameters_count': sum(p.size for p in model.get_parameters().values()),
        'layers': [
            {
                'name': layer.name,
                'class_name': layer.__class__.__name__,
                'config': layer.get_config(),
                'parameters': sum(p.size for p in layer.get_parameters().values()) 
                if hasattr(layer, 'parameters') else 0
            }
            for layer in model.layers
        ]
    }