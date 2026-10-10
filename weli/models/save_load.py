"""
Sauvegarde et chargement avancé des modèles Weli.
"""
import pickle
import json
import numpy as np
import os
import h5py
from typing import Dict, Any, Union, Optional, List, Tuple
import warnings

# Éviter les imports circulaires - on utilise des strings pour les type hints
# Les imports réels se font à l'intérieur des fonctions quand nécessaire

class ModelSaveError(Exception):
    """Exception pour les erreurs de sauvegarde de modèle."""
    pass

class ModelLoadError(Exception):
    """Exception pour les erreurs de chargement de modèle."""
    pass

def save_model(model,  # Type: tout objet avec get_parameters() et get_config()
               filepath: str, 
               save_weights_only: bool = False,
               include_optimizer: bool = False,
               optimizer: Optional[Any] = None,
               overwrite: bool = True) -> None:
    """
    Sauvegarde un modèle Weli dans un fichier.
    
    Args:
        model: Modèle à sauvegarder (doit avoir get_parameters() et get_config())
        filepath: Chemin du fichier de sauvegarde
        save_weights_only: Si True, sauvegarde seulement les poids
        include_optimizer: Si True, inclut l'état de l'optimiseur
        optimizer: Optimiseur à sauvegarder (requis si include_optimizer=True)
        overwrite: Si True, écrase le fichier existant
        
    Raises:
        ModelSaveError: Si la sauvegarde échoue
        ValueError: Si les arguments sont invalides
    """
    try:
        # Vérifier les arguments
        if include_optimizer and optimizer is None:
            raise ValueError("optimizer must be provided when include_optimizer=True")
        
        # Vérifier que le modèle a les méthodes nécessaires
        if not hasattr(model, 'get_parameters'):
            raise ModelSaveError("Le modèle doit avoir une méthode get_parameters()")
        if not hasattr(model, 'get_config'):
            raise ModelSaveError("Le modèle doit avoir une méthode get_config()")
        
        # Vérifier si le fichier existe
        if os.path.exists(filepath) and not overwrite:
            raise ModelSaveError(f"File {filepath} already exists and overwrite=False")
        
        # Créer le répertoire si nécessaire
        directory = os.path.dirname(os.path.abspath(filepath))
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        if save_weights_only:
            # Sauvegarde seulement des poids
            _save_weights_only(model, filepath)
        else:
            # Sauvegarde complète du modèle
            _save_full_model(model, filepath, include_optimizer, optimizer)
        
    except Exception as e:
        raise ModelSaveError(f"Échec de la sauvegarde du modèle: {str(e)}")

def _save_weights_only(model, filepath: str) -> None:
    """
    Sauvegarde seulement les poids du modèle.
    
    Args:
        model: Modèle avec get_parameters()
        filepath: Chemin du fichier
    """
    # Ajuster l'extension si nécessaire
    if not filepath.endswith(('.npz', '.h5', '.weights')):
        filepath += '.npz'  # Par défaut NPZ
    
    params = model.get_parameters()
    
    if filepath.endswith('.npz'):
        # Format NPZ (NumPy) compressé
        np.savez_compressed(filepath, **params)
    
    elif filepath.endswith('.h5'):
        # Format HDF5
        with h5py.File(filepath, 'w') as f:
            for key, value in params.items():
                # Nettoyer les noms pour HDF5
                safe_key = key.replace('/', '_').replace('.', '_').replace(':', '_')
                f.create_dataset(safe_key, data=value, compression='gzip')
    
    else:
        raise ValueError(f"Format non supporté pour les poids: {filepath}")

def _save_full_model(model, 
                    filepath: str, 
                    include_optimizer: bool,
                    optimizer: Optional[Any]) -> None:
    """
    Sauvegarde complète du modèle.
    
    Args:
        model: Modèle avec get_parameters() et get_config()
        filepath: Chemin du fichier
        include_optimizer: Inclure l'optimiseur
        optimizer: Optimiseur
    """
    # Ajuster l'extension si nécessaire
    if not filepath.endswith(('.weli', '.pkl', '.json', '.model')):
        filepath += '.weli'  # Extension par défaut de Weli
    
    # Préparer les données du modèle
    model_data = {
        'metadata': {
            'weli_version': '0.1.0',
            'model_class': model.__class__.__name__,
            'model_name': getattr(model, 'name', 'unnamed'),
            'timestamp': np.datetime64('now').astype(str),
            'format_version': 1
        },
        'model_config': model.get_config(),
        'parameters': _convert_params_for_storage(model.get_parameters()),
        'training_info': {
            'input_shape': getattr(model, '_input_shape', None),
            'output_shape': getattr(model, '_output_shape', None),
            'initialized': getattr(model, 'initialized', False)
        }
    }
    
    # Inclure l'historique si disponible
    if hasattr(model, 'history') and model.history:
        model_data['history'] = model.history
    
    # Inclure l'optimiseur si demandé
    if include_optimizer and optimizer is not None:
        try:
            optimizer_state = {
                'class_name': optimizer.__class__.__name__,
                'config': optimizer.get_config() if hasattr(optimizer, 'get_config') else {},
                'weights': optimizer.get_weights() if hasattr(optimizer, 'get_weights') else None
            }
            model_data['optimizer_state'] = optimizer_state
        except Exception as e:
            warnings.warn(f"Impossible de sauvegarder l'optimiseur: {e}")
    
    # Sauvegarde selon le format
    if filepath.endswith('.json'):
        # Sauvegarde JSON (sérialisable seulement)
        with open(filepath, 'w', encoding='utf-8') as f:
            json_data = _make_json_serializable(model_data)
            json.dump(json_data, f, indent=2)
    
    else:
        # Sauvegarde binaire (pickle ou format Weli)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

def _convert_params_for_storage(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Convertit les paramètres pour le stockage.
    
    Args:
        params: Paramètres du modèle
        
    Returns:
        Paramètres convertis
    """
    # Pour l'instant, retourne simplement les paramètres
    # On pourrait ajouter de la compression ici
    return params

def _make_json_serializable(data: Any) -> Any:
    """
    Convertit les données pour les rendre JSON sérialisables.
    
    Args:
        data: Données à convertir
        
    Returns:
        Données JSON sérialisables
    """
    if isinstance(data, dict):
        return {k: _make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_make_json_serializable(v) for v in data]
    elif isinstance(data, tuple):
        return [_make_json_serializable(v) for v in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, (np.float32, np.float64, np.float16)):
        return float(data)
    elif isinstance(data, (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(data)
    elif isinstance(data, (np.bool_)):
        return bool(data)
    elif data is None:
        return None
    elif isinstance(data, (str, int, float, bool)):
        return data
    else:
        # Pour les autres types, convertir en string
        try:
            return str(data)
        except:
            return None

def load_model(filepath: str, 
               custom_objects: Optional[Dict[str, Any]] = None,
               compile: bool = True):
    """
    Charge un modèle Weli depuis un fichier.
    
    Args:
        filepath: Chemin du fichier
        custom_objects: Dictionnaire d'objets personnalisés pour la désérialisation
        compile: Si True, tente de recompiler le modèle après chargement
        
    Returns:
        Modèle chargé
        
    Raises:
        ModelLoadError: Si le chargement échoue
        FileNotFoundError: Si le fichier n'existe pas
    """
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        # Détecter le format du fichier
        if filepath.endswith('.npz'):
            raise ModelLoadError("Les fichiers .npz contiennent seulement les poids. "
                               "Utilisez load_weights() avec un modèle existant.")
        
        elif filepath.endswith('.h5'):
            raise ModelLoadError("Les fichiers .h5 contiennent seulement les poids. "
                               "Utilisez load_weights() avec un modèle existant.")
        
        elif filepath.endswith('.json'):
            # Charger depuis JSON
            model = _load_from_json(filepath, custom_objects)
        
        else:
            # Charger depuis format binaire
            model = _load_from_binary(filepath, custom_objects)
        
        # Tenter de recompiler si demandé (sans réinitialiser les poids chargés)
        if compile and hasattr(model, 'compile') and model.initialized:
            try:
                model.compile()
            except Exception as e:
                warnings.warn(f"Impossible de recompiler le modèle: {e}")
        
        return model
        
    except Exception as e:
        raise ModelLoadError(f"Échec du chargement du modèle: {str(e)}")

def _load_from_json(filepath: str, custom_objects: Optional[Dict[str, Any]] = None):
    """
    Charge un modèle depuis un fichier JSON.
    
    Args:
        filepath: Chemin du fichier JSON
        custom_objects: Objets personnalisés
        
    Returns:
        Modèle chargé
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Importer le registry ici pour éviter les imports circulaires
    from .registry import registry
    
    # Enregistrer les objets personnalisés
    if custom_objects:
        for name, obj in custom_objects.items():
            registry.register_custom_object(name, obj)
    
    # Vérifier la version
    metadata = model_data.get('metadata', {})
    if metadata.get('format_version', 0) > 1:
        warnings.warn(f"Version du format ({metadata.get('format_version')}) plus récente que supportée (1)")
    
    # Désérialiser le modèle
    model = registry.deserialize_model(model_data['model_config'])

    # Charger les paramètres si disponibles
    if 'parameters' in model_data:
        # Convertir les listes de JSON en arrays NumPy
        params = {}
        for key, value in model_data['parameters'].items():
            if isinstance(value, list):
                params[key] = np.array(value)
            else:
                params[key] = value
        
        try:
            model.set_parameters(params)
        except Exception as e:
            warnings.warn(f"Impossible de charger certains paramètres: {e}")
    
    # Restaurer les informations d'entraînement
    if 'training_info' in model_data:
        info = model_data['training_info']
        if info['input_shape']:
            model._input_shape = tuple(info['input_shape'])
        if info['output_shape']:
            model._output_shape = tuple(info['output_shape'])
        model.initialized = info.get('initialized', False)
    
    # Restaurer l'historique
    if 'history' in model_data:
        model.history = model_data['history']
    
    return model

def _load_from_binary(filepath: str, custom_objects: Optional[Dict[str, Any]] = None):
    """
    Charge un modèle depuis un fichier binaire.
    
    Args:
        filepath: Chemin du fichier
        custom_objects: Objets personnalisés
        
    Returns:
        Modèle chargé
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # Importer le registry ici pour éviter les imports circulaires
    from .registry import registry
    
    # Enregistrer les objets personnalisés
    if custom_objects:
        for name, obj in custom_objects.items():
            registry.register_custom_object(name, obj)
    
    # Vérifier la version
    metadata = model_data.get('metadata', {})
    if metadata.get('format_version', 0) > 1:
        warnings.warn(f"Version du format ({metadata.get('format_version')}) plus récente que supportée (1)")
    
    # Désérialiser le modèle
    model = registry.deserialize_model(model_data['model_config'])

    # Charger les paramètres
    if 'parameters' in model_data:
        model.set_parameters(model_data['parameters'])
    
    # Restaurer les informations d'entraînement
    if 'training_info' in model_data:
        info = model_data['training_info']
        model._input_shape = info['input_shape']
        model._output_shape = info['output_shape']
        model.initialized = info.get('initialized', False)
    
    # Restaurer l'historique
    if 'history' in model_data:
        model.history = model_data['history']
    
    # Restaurer l'optimiseur si présent
    if 'optimizer_state' in model_data:
        model._optimizer_state = model_data['optimizer_state']
    
    return model

def load_weights(model, 
                 filepath: str, 
                 skip_mismatch: bool = False,
                 by_name: bool = False) -> None:
    """
    Charge les poids dans un modèle existant.
    
    Args:
        model: Modèle existant (doit avoir get_parameters() et set_parameters())
        filepath: Chemin du fichier de poids
        skip_mismatch: Si True, ignore les poids qui ne correspondent pas
        by_name: Si True, charge les poids par nom de couche
        
    Raises:
        ValueError: Si le format du fichier n'est pas supporté
        ModelLoadError: Si le chargement échoue
    """
    try:
        # Vérifier que le modèle a les méthodes nécessaires
        if not hasattr(model, 'get_parameters'):
            raise ModelLoadError("Le modèle doit avoir une méthode get_parameters()")
        if not hasattr(model, 'set_parameters'):
            raise ModelLoadError("Le modèle doit avoir une méthode set_parameters()")
        
        # Vérifier que le fichier existe
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        # Charger selon le format
        if filepath.endswith('.npz'):
            params = _load_weights_npz(filepath)
        elif filepath.endswith('.h5'):
            params = _load_weights_h5(filepath)
        else:
            raise ValueError(f"Format de poids non supporté: {filepath}. Utilisez .npz ou .h5")
        
        # Charger les poids
        if by_name:
            _load_weights_by_name(model, params, skip_mismatch)
        else:
            _load_weights_by_order(model, params, skip_mismatch)
        
        
    except Exception as e:
        raise ModelLoadError(f"Échec du chargement des poids: {str(e)}")

def _load_weights_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Charge les poids depuis un fichier NPZ."""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}

def _load_weights_h5(filepath: str) -> Dict[str, np.ndarray]:
    """Charge les poids depuis un fichier HDF5."""
    params = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            params[key] = f[key][()]
    return params

def _load_weights_by_order(model, 
                          params: Dict[str, np.ndarray],
                          skip_mismatch: bool) -> None:
    """
    Charge les poids par ordre des couches.
    
    Args:
        model: Modèle avec trainable_layers
        params: Dictionnaire de paramètres
        skip_mismatch: Ignorer les mismatches
    """
    if not hasattr(model, 'trainable_layers'):
        raise ModelLoadError("Le modèle n'a pas d'attribut trainable_layers")
    
    trainable_layers = model.trainable_layers
    
    param_keys = list(params.keys())
    param_index = 0
    
    for layer in trainable_layers:
        if param_index >= len(param_keys):
            break
        
        try:
            layer_params = layer.get_parameters()
        except:
            continue  # Passer cette couche si elle n'a pas de paramètres
        
        loaded_params = {}
        
        for param_name in layer_params.keys():
            if param_index < len(param_keys):
                key = param_keys[param_index]
                value = params[key]
                
                # Vérifier la compatibilité des shapes
                expected_shape = layer_params[param_name].shape
                actual_shape = value.shape
                
                if expected_shape == actual_shape:
                    loaded_params[param_name] = value
                    param_index += 1
                elif not skip_mismatch:
                    raise ValueError(
                        f"Shape mismatch for parameter {key}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )
                else:
                    warnings.warn(f"Ignoring parameter {key} due to shape mismatch")
        
        if loaded_params:
            try:
                layer.set_parameters(loaded_params)
            except Exception as e:
                warnings.warn(f"Impossible de charger les poids pour la couche {layer.name}: {e}")

def _load_weights_by_name(model, 
                         params: Dict[str, np.ndarray],
                         skip_mismatch: bool) -> None:
    """
    Charge les poids par nom de couche/paramètre.
    
    Args:
        model: Modèle avec trainable_layers
        params: Dictionnaire de paramètres
        skip_mismatch: Ignorer les mismatches
    """
    if not hasattr(model, 'trainable_layers'):
        raise ModelLoadError("Le modèle n'a pas d'attribut trainable_layers")
    
    trainable_layers = model.trainable_layers
    
    for layer in trainable_layers:
        try:
            layer_params = layer.get_parameters()
        except:
            continue  # Passer cette couche si elle n'a pas de paramètres
        
        loaded_params = {}
        
        for param_name in layer_params.keys():
            # Chercher le paramètre par nom
            possible_keys = [
                f"{layer.name}_{param_name}",
                f"{layer.name}.{param_name}",
                f"{layer.name}/{param_name}",
                param_name
            ]
            
            found = False
            for key in possible_keys:
                if key in params:
                    value = params[key]
                    
                    # Vérifier la compatibilité des shapes
                    expected_shape = layer_params[param_name].shape
                    actual_shape = value.shape
                    
                    if expected_shape == actual_shape:
                        loaded_params[param_name] = value
                        found = True
                        break
                    elif not skip_mismatch:
                        raise ValueError(
                            f"Shape mismatch for parameter {key} in layer {layer.name}: "
                            f"expected {expected_shape}, got {actual_shape}"
                        )
                    else:
                        warnings.warn(
                            f"Ignoring parameter {key} in layer {layer.name} due to shape mismatch"
                        )
            
            if not found and not skip_mismatch:
                raise ValueError(f"Parameter {param_name} not found for layer {layer.name}")
        
        if loaded_params:
            try:
                layer.set_parameters(loaded_params)
            except Exception as e:
                warnings.warn(f"Impossible de charger les poids pour la couche {layer.name}: {e}")

def save_model_architecture(model, 
                           filepath: str,
                           include_weights_info: bool = True) -> None:
    """
    Sauvegarde seulement l'architecture du modèle.
    
    Args:
        model: Modèle avec get_config()
        filepath: Chemin du fichier
        include_weights_info: Inclure les informations sur les poids
    """
    try:
        if not hasattr(model, 'get_config'):
            raise ModelSaveError("Le modèle doit avoir une méthode get_config()")
        
        config = model.get_config()
        
        # Ajouter des informations supplémentaires
        arch_data = {
            'model_config': config,
            'metadata': {
                'model_name': getattr(model, 'name', 'unnamed'),
                'model_class': model.__class__.__name__,
                'weli_version': '0.1.0',
                'timestamp': np.datetime64('now').astype(str)
            }
        }
        
        if include_weights_info and hasattr(model, 'get_parameters'):
            try:
                params = model.get_parameters()
                arch_data['weights_info'] = {
                    'total_parameters': sum(p.size for p in params.values()),
                    'parameter_shapes': {k: v.shape for k, v in params.items()}
                }
            except:
                warnings.warn("Impossible d'obtenir les informations sur les poids")
        
        # Sauvegarder en JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(arch_data, f, indent=2, default=str)
        
        
    except Exception as e:
        raise ModelSaveError(f"Échec de la sauvegarde de l'architecture: {str(e)}")

def model_to_dict(model) -> Dict[str, Any]:
    """
    Convertit un modèle en dictionnaire détaillé.
    
    Args:
        model: Modèle Weli
        
    Returns:
        Dictionnaire détaillé du modèle
    """
    result = {
        'name': getattr(model, 'name', 'unnamed'),
        'class_name': model.__class__.__name__,
        'initialized': getattr(model, 'initialized', False),
        'config': model.get_config() if hasattr(model, 'get_config') else {}
    }
    
    # Informations sur les couches
    if hasattr(model, 'layers'):
        result['num_layers'] = len(model.layers)
        result['layers'] = []
        
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'name': getattr(layer, 'name', f'layer_{i}'),
                'class_name': layer.__class__.__name__,
                'trainable': getattr(layer, 'trainable', True)
            }
            
            # Informations sur les paramètres
            if hasattr(layer, 'get_parameters'):
                try:
                    params = layer.get_parameters()
                    layer_info['parameters'] = {
                        'count': len(params),
                        'shapes': {k: v.shape for k, v in params.items()},
                        'total_size': sum(v.size for v in params.values())
                    }
                except:
                    layer_info['parameters'] = {'count': 0, 'total_size': 0}
            
            # Shape de sortie
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = layer.output_shape
            
            result['layers'].append(layer_info)
    
    # Informations sur les couches trainables
    if hasattr(model, 'trainable_layers'):
        result['num_trainable_layers'] = len(model.trainable_layers)
    
    # Paramètres totaux
    if hasattr(model, 'get_parameters'):
        try:
            params = model.get_parameters()
            result['total_parameters'] = sum(p.size for p in params.values())
            result['parameter_shapes'] = {k: v.shape for k, v in params.items()}
        except:
            result['total_parameters'] = 0
            result['parameter_shapes'] = {}
    
    # Shapes d'input/output
    if hasattr(model, '_input_shape'):
        result['input_shape'] = model._input_shape
    if hasattr(model, '_output_shape'):
        result['output_shape'] = model._output_shape
    
    # Historique si disponible
    if hasattr(model, 'history') and model.history:
        history = model.history
        result['history_summary'] = {
            'epochs_trained': len(history.get('train_loss', [])),
            'final_train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'final_val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
            'final_train_acc': history.get('train_acc', [])[-1] if history.get('train_acc') else None,
            'final_val_acc': history.get('val_acc', [])[-1] if history.get('val_acc') else None
        }
    
    return result

def get_model_info(model) -> str:
    """
    Retourne une chaîne d'information formatée sur le modèle.
    
    Args:
        model: Modèle Weli
        
    Returns:
        String formatée avec les informations
    """
    info = model_to_dict(model)
    
    lines = []
    lines.append("=" * 80)
    lines.append(f"MODEL INFORMATION: {info['name']}")
    lines.append("=" * 80)
    lines.append(f"Class: {info['class_name']}")
    lines.append(f"Initialized: {info['initialized']}")
    
    if 'num_layers' in info:
        lines.append(f"Total layers: {info['num_layers']}")
    if 'num_trainable_layers' in info:
        lines.append(f"Trainable layers: {info['num_trainable_layers']}")
    if 'total_parameters' in info:
        lines.append(f"Total parameters: {info['total_parameters']:,}")
    
    if 'input_shape' in info:
        lines.append(f"Input shape: {info['input_shape']}")
    if 'output_shape' in info:
        lines.append(f"Output shape: {info['output_shape']}")
    
    # Informations sur les couches
    if 'layers' in info and info['layers']:
        lines.append("\nLAYERS:")
        lines.append("-" * 80)
        
        for layer in info['layers']:
            lines.append(f"[{layer['index']}] {layer['name']} ({layer['class_name']})")
            lines.append(f"    Trainable: {layer['trainable']}")
            
            if 'output_shape' in layer:
                lines.append(f"    Output shape: {layer['output_shape']}")
            
            if 'parameters' in layer and layer['parameters']['total_size'] > 0:
                lines.append(f"    Parameters: {layer['parameters']['total_size']:,}")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def check_model_compatibility(model1, 
                             model2,
                             check_weights: bool = False) -> Dict[str, Any]:
    """
    Vérifie la compatibilité entre deux modèles.
    
    Args:
        model1: Premier modèle
        model2: Deuxième modèle
        check_weights: Vérifier aussi la compatibilité des poids
        
    Returns:
        Dictionnaire avec les résultats de compatibilité
    """
    result = {
        'architecture_compatible': True,
        'weights_compatible': True if check_weights else None,
        'issues': []
    }
    
    # Vérifier que les deux modèles ont des couches
    if not hasattr(model1, 'layers') or not hasattr(model2, 'layers'):
        result['architecture_compatible'] = False
        result['issues'].append("Un des modèles n'a pas d'attribut 'layers'")
        return result
    
    # Vérifier le nombre de couches
    if len(model1.layers) != len(model2.layers):
        result['architecture_compatible'] = False
        result['issues'].append(
            f"Nombre de couches différent: {len(model1.layers)} vs {len(model2.layers)}"
        )
    
    # Vérifier chaque couche
    for i in range(min(len(model1.layers), len(model2.layers))):
        layer1 = model1.layers[i]
        layer2 = model2.layers[i]
        
        if layer1.__class__.__name__ != layer2.__class__.__name__:
            result['architecture_compatible'] = False
            result['issues'].append(
                f"Couche {i}: types différents {layer1.__class__.__name__} vs {layer2.__class__.__name__}"
            )
        
        if hasattr(layer1, 'output_shape') and hasattr(layer2, 'output_shape'):
            if layer1.output_shape != layer2.output_shape:
                result['architecture_compatible'] = False
                result['issues'].append(
                    f"Couche {i}: shapes de sortie différents {layer1.output_shape} vs {layer2.output_shape}"
                )
    
    # Vérifier les poids si demandé
    if check_weights and result['architecture_compatible']:
        if hasattr(model1, 'get_parameters') and hasattr(model2, 'get_parameters'):
            try:
                params1 = model1.get_parameters()
                params2 = model2.get_parameters()
                
                if len(params1) != len(params2):
                    result['weights_compatible'] = False
                    result['issues'].append(
                        f"Nombre de paramètres différent: {len(params1)} vs {len(params2)}"
                    )
                else:
                    # Comparer les shapes
                    keys1 = list(params1.keys())
                    keys2 = list(params2.keys())
                    
                    for k1, k2 in zip(keys1, keys2):
                        if params1[k1].shape != params2[k2].shape:
                            result['weights_compatible'] = False
                            result['issues'].append(
                                f"Paramètre {k1}: shape différent {params1[k1].shape} vs {params2[k2].shape}"
                            )
            except:
                result['weights_compatible'] = False
                result['issues'].append("Impossible de récupérer les paramètres des modèles")
        else:
            result['weights_compatible'] = False
            result['issues'].append("Un des modèles n'a pas de méthode get_parameters()")
    
    return result