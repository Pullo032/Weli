"""
Utilitaires pour les modèles Weli.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings

def count_parameters(model) -> Dict[str, int]:
    """
    Compte les paramètres du modèle.
    
    Args:
        model: Modèle Weli (doit avoir 'layers' ou 'get_parameters')
        
    Returns:
        Dictionnaire avec les counts
        
    Example:
        >>> from weli.models import Sequential
        >>> model = Sequential([...])
        >>> counts = count_parameters(model)
        >>> print(f"Total: {counts['total_params']:,}")
    """
    try:
        # Essayer d'abord avec get_parameters
        if hasattr(model, 'get_parameters'):
            params = model.get_parameters()
            total = sum(p.size for p in params.values())
            return {
                'total_params': total,
                'trainable_params': total,
                'non_trainable_params': 0
            }
        
        # Sinon, parcourir les couches
        elif hasattr(model, 'layers'):
            total_params = 0
            trainable_params = 0
            
            for layer in model.layers:
                if hasattr(layer, 'parameters'):
                    layer_params = sum(p.size for p in layer.parameters.values())
                    total_params += layer_params
                    if getattr(layer, 'trainable', True):
                        trainable_params += layer_params
            
            return {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'non_trainable_params': total_params - trainable_params
            }
        
        else:
            warnings.warn("Model has no 'get_parameters' or 'layers' attribute")
            return {
                'total_params': 0,
                'trainable_params': 0,
                'non_trainable_params': 0
            }
            
    except Exception as e:
        warnings.warn(f"Error counting parameters: {e}")
        return {
            'total_params': 0,
            'trainable_params': 0,
            'non_trainable_params': 0
        }

def print_model_summary(model, detailed: bool = False) -> None:
    """
    Affiche un résumé détaillé du modèle.
    
    Args:
        model: Modèle Weli
        detailed: Si True, affiche plus de détails
        
    Example:
        >>> print_model_summary(model)
        ========================================
        Model: MyModel (Sequential)
        ========================================
        Total layers: 5
        Total params: 1,234,567
        Trainable params: 1,234,567
        Input shape: (784,)
        Output shape: (10,)
        ========================================
    """
    try:
        model_name = getattr(model, 'name', 'Unnamed')
        model_class = model.__class__.__name__
        
        print("=" * 60)
        print(f"Model: {model_name} ({model_class})")
        print("=" * 60)
        
        # Compter les couches
        if hasattr(model, 'layers'):
            n_layers = len(model.layers)
            print(f"Total layers: {n_layers}")
        else:
            n_layers = 0
            print("Total layers: Unknown")
        
        # Compter les paramètres
        params_info = count_parameters(model)
        print(f"Total params: {params_info['total_params']:,}")
        print(f"Trainable params: {params_info['trainable_params']:,}")
        print(f"Non-trainable params: {params_info['non_trainable_params']:,}")
        
        # Shapes
        if hasattr(model, '_input_shape'):
            print(f"Input shape: {model._input_shape}")
        if hasattr(model, '_output_shape'):
            print(f"Output shape: {model._output_shape}")
        
        # Initialisé ou non
        if hasattr(model, 'initialized'):
            print(f"Initialized: {model.initialized}")
        
        print("=" * 60)
        
        # Détails par couche
        if detailed and hasattr(model, 'layers') and model.layers:
            print("\nLAYER DETAILS:")
            print("-" * 60)
            print(f"{'Index':<6} {'Name':<20} {'Type':<15} {'Output Shape':<20} {'Params':<10}")
            print("-" * 60)
            
            for i, layer in enumerate(model.layers):
                layer_name = getattr(layer, 'name', f'layer_{i}')
                layer_type = layer.__class__.__name__
                
                # Output shape
                output_shape = getattr(layer, 'output_shape', '?')
                
                # Param count
                param_count = 0
                if hasattr(layer, 'parameters'):
                    param_count = sum(p.size for p in layer.parameters.values())
                
                # Trainable
                trainable = getattr(layer, 'trainable', True)
                trainable_str = "T" if trainable else "NT"
                
                print(f"{i:<6} {layer_name:<20} {layer_type:<15} "
                      f"{str(output_shape):<20} {param_count:<10,} ({trainable_str})")
            
            print("=" * 60)
            
    except Exception as e:
        print(f"Error printing model summary: {e}")

def get_layer_by_name(model, name: str, exact: bool = True):
    """
    Récupère une couche par son nom.
    
    Args:
        model: Modèle Weli
        name: Nom de la couche
        exact: Si True, recherche exacte, sinon partielle
        
    Returns:
        Couche ou liste de couches
        
    Example:
        >>> layer = get_layer_by_name(model, "dense_1")
        >>> layers = get_layer_by_name(model, "dense", exact=False)
    """
    if not hasattr(model, 'layers'):
        return None if exact else []
    
    if exact:
        # Recherche exacte
        for layer in model.layers:
            if getattr(layer, 'name', '') == name:
                return layer
        return None
    else:
        # Recherche partielle
        results = []
        for layer in model.layers:
            layer_name = getattr(layer, 'name', '')
            if name in layer_name:
                results.append(layer)
        return results

def get_layer_by_index(model, index: int):
    """
    Récupère une couche par son index.
    
    Args:
        model: Modèle Weli
        index: Index de la couche (0-based)
        
    Returns:
        Couche
        
    Raises:
        IndexError: Si l'index est hors limites
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model has no layers")
    
    if index < 0 or index >= len(model.layers):
        raise IndexError(f"Layer index {index} out of range (0-{len(model.layers)-1})")
    
    return model.layers[index]

def get_layer_output(model, x: np.ndarray, 
                     layer_spec: Union[int, str] = -1,
                     training: bool = False) -> np.ndarray:
    """
    Récupère la sortie d'une couche spécifique.
    
    Args:
        model: Modèle Weli
        x: Données d'input
        layer_spec: Index ou nom de la couche (-1 pour dernière)
        training: Mode entraînement
        
    Returns:
        Sortie de la couche
        
    Example:
        >>> output = get_layer_output(model, x, layer_spec=2)
        >>> output = get_layer_output(model, x, layer_spec="dense_1")
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model has no layers")
    
    # Déterminer l'index de la couche
    if isinstance(layer_spec, int):
        layer_index = layer_spec
        if layer_index < 0:
            layer_index = len(model.layers) + layer_index
    elif isinstance(layer_spec, str):
        # Chercher par nom
        layer = get_layer_by_name(model, layer_spec, exact=True)
        if layer is None:
            raise ValueError(f"Layer '{layer_spec}' not found")
        layer_index = model.layers.index(layer)
    else:
        raise TypeError("layer_spec must be int or str")
    
    # Vérifier l'index
    if layer_index < 0 or layer_index >= len(model.layers):
        raise IndexError(f"Layer index {layer_index} out of range")
    
    # Propagation avant jusqu'à la couche
    output = x
    for i in range(layer_index + 1):
        layer = model.layers[i]
        
        # Définir le mode training
        if hasattr(layer, 'training'):
            layer.training = training
        
        # Forward pass
        output = layer.forward(output)
    
    return output

def get_all_layer_outputs(model, x: np.ndarray, 
                          training: bool = False) -> List[Tuple[str, np.ndarray]]:
    """
    Récupère les sorties de toutes les couches.
    
    Args:
        model: Modèle Weli
        x: Données d'input
        training: Mode entraînement
        
    Returns:
        Liste de (nom_couche, sortie)
    """
    if not hasattr(model, 'layers'):
        return []
    
    outputs = []
    current = x
    
    for i, layer in enumerate(model.layers):
        # Définir le mode training
        if hasattr(layer, 'training'):
            layer.training = training
        
        # Forward pass
        current = layer.forward(current)
        
        # Stocker
        layer_name = getattr(layer, 'name', f'layer_{i}')
        outputs.append((layer_name, current.copy()))
    
    return outputs

def model_to_json(model, filepath: Optional[str] = None, 
                  include_weights_info: bool = True) -> Dict[str, Any]:
    """
    Convertit un modèle en dictionnaire JSON-serializable.
    
    Args:
        model: Modèle Weli
        filepath: Si fourni, sauvegarde dans un fichier
        include_weights_info: Inclure les infos sur les poids
        
    Returns:
        Configuration du modèle
    """
    try:
        # Configuration de base
        config = {
            'model_name': getattr(model, 'name', 'Unnamed'),
            'model_class': model.__class__.__name__,
            'weli_version': '0.1.0',
            'timestamp': np.datetime64('now').astype(str)
        }
        
        # Config du modèle si disponible
        if hasattr(model, 'get_config'):
            config['model_config'] = model.get_config()
        
        # Informations sur les couches
        if hasattr(model, 'layers'):
            layers_info = []
            for i, layer in enumerate(model.layers):
                layer_info = {
                    'index': i,
                    'name': getattr(layer, 'name', f'layer_{i}'),
                    'class': layer.__class__.__name__,
                    'trainable': getattr(layer, 'trainable', True)
                }
                
                # Config de la couche
                if hasattr(layer, 'get_config'):
                    layer_info['config'] = layer.get_config()
                
                # Output shape
                if hasattr(layer, 'output_shape'):
                    layer_info['output_shape'] = layer.output_shape
                
                layers_info.append(layer_info)
            
            config['layers'] = layers_info
        
        # Informations sur les paramètres
        if include_weights_info:
            params_info = count_parameters(model)
            config['parameters_info'] = params_info
            
            # Shapes des paramètres si disponible
            if hasattr(model, 'get_parameters'):
                try:
                    params = model.get_parameters()
                    config['parameter_shapes'] = {
                        k: v.shape for k, v in params.items()
                    }
                except:
                    pass
        
        # Shapes d'input/output
        if hasattr(model, '_input_shape'):
            config['input_shape'] = model._input_shape
        if hasattr(model, '_output_shape'):
            config['output_shape'] = model._output_shape
        
        # Initialisation
        if hasattr(model, 'initialized'):
            config['initialized'] = model.initialized
        
        # Sauvegarder dans un fichier si demandé
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, default=str)
            print(f"✓ Model configuration saved to {filepath}")
        
        return config
        
    except Exception as e:
        warnings.warn(f"Error converting model to JSON: {e}")
        return {}

def model_from_json(filepath: str, custom_objects: Optional[Dict[str, Any]] = None):
    """
    Crée un modèle depuis un fichier JSON.
    Note: Nécessite que le registry soit initialisé.
    
    Args:
        filepath: Chemin du fichier JSON
        custom_objects: Objets personnalisés
        
    Returns:
        Modèle créé
    """
    try:
        # Charger la configuration
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Utiliser le registry pour créer le modèle
        # Note: Pour l'instant, retourne None car nécessite un système complet
        # À implémenter avec le registry
        
        warnings.warn("model_from_json requires full registry implementation")
        return None
        
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON file: {filepath}")
    except Exception as e:
        raise RuntimeError(f"Error loading model from JSON: {e}")

def compare_models(model1, model2, 
                   compare_weights: bool = False,
                   tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Compare deux modèles.
    
    Args:
        model1: Premier modèle
        model2: Deuxième modèle
        compare_weights: Comparer aussi les poids
        tolerance: Tolérance pour la comparaison des poids
        
    Returns:
        Résultats de comparaison
    """
    result = {
        'architectures_match': True,
        'weights_match': None,
        'differences': [],
        'similarity_score': 0.0
    }
    
    try:
        # Comparer les classes
        if model1.__class__.__name__ != model2.__class__.__name__:
            result['architectures_match'] = False
            result['differences'].append(
                f"Different classes: {model1.__class__.__name__} vs {model2.__class__.__name__}"
            )
        
        # Comparer les noms
        name1 = getattr(model1, 'name', 'Unnamed')
        name2 = getattr(model2, 'name', 'Unnamed')
        if name1 != name2:
            result['differences'].append(f"Different names: {name1} vs {name2}")
        
        # Comparer les couches
        if hasattr(model1, 'layers') and hasattr(model2, 'layers'):
            layers1 = model1.layers
            layers2 = model2.layers
            
            if len(layers1) != len(layers2):
                result['architectures_match'] = False
                result['differences'].append(
                    f"Different number of layers: {len(layers1)} vs {len(layers2)}"
                )
            else:
                # Comparer chaque couche
                for i, (layer1, layer2) in enumerate(zip(layers1, layers2)):
                    if layer1.__class__.__name__ != layer2.__class__.__name__:
                        result['architectures_match'] = False
                        result['differences'].append(
                            f"Layer {i}: different types - "
                            f"{layer1.__class__.__name__} vs {layer2.__class__.__name__}"
                        )
                    
                    # Comparer les noms
                    name1 = getattr(layer1, 'name', f'layer_{i}')
                    name2 = getattr(layer2, 'name', f'layer_{i}')
                    if name1 != name2:
                        result['differences'].append(
                            f"Layer {i}: different names - {name1} vs {name2}"
                        )
        
        # Comparer les poids si demandé
        if compare_weights and result['architectures_match']:
            if hasattr(model1, 'get_parameters') and hasattr(model2, 'get_parameters'):
                try:
                    params1 = model1.get_parameters()
                    params2 = model2.get_parameters()
                    
                    if len(params1) != len(params2):
                        result['weights_match'] = False
                        result['differences'].append(
                            f"Different number of parameters: {len(params1)} vs {len(params2)}"
                        )
                    else:
                        all_match = True
                        total_diff = 0.0
                        total_params = 0
                        
                        keys1 = list(params1.keys())
                        keys2 = list(params2.keys())
                        
                        for k1, k2 in zip(keys1, keys2):
                            if params1[k1].shape != params2[k2].shape:
                                result['weights_match'] = False
                                result['differences'].append(
                                    f"Parameter shape mismatch: {k1} {params1[k1].shape} vs "
                                    f"{k2} {params2[k2].shape}"
                                )
                                all_match = False
                            else:
                                diff = np.abs(params1[k1] - params2[k2]).sum()
                                total_diff += diff
                                total_params += params1[k1].size
                        
                        if all_match:
                            result['weights_match'] = total_diff < tolerance
                            if total_params > 0:
                                result['similarity_score'] = 1.0 - (total_diff / total_params)
                            else:
                                result['similarity_score'] = 1.0
                            
                            if total_diff > tolerance:
                                result['differences'].append(
                                    f"Weight difference exceeds tolerance: {total_diff:.6f} > {tolerance}"
                                )
                except Exception as e:
                    result['weights_match'] = False
                    result['differences'].append(f"Error comparing weights: {e}")
        
        return result
        
    except Exception as e:
        warnings.warn(f"Error comparing models: {e}")
        result['differences'].append(f"Comparison error: {e}")
        return result

def extract_submodel(model, start_layer: Union[int, str], 
                     end_layer: Union[int, str] = None):
    """
    Extrait un sous-modèle entre deux couches.
    
    Args:
        model: Modèle source
        start_layer: Index ou nom de la couche de début
        end_layer: Index ou nom de la couche de fin (inclusive)
        
    Returns:
        Sous-modèle
        
    Note: Pour l'instant, fonction de base. À améliorer.
    """
    if not hasattr(model, 'layers'):
        raise ValueError("Model has no layers")
    
    # Convertir les spécifications en indices
    def get_layer_index(spec):
        if isinstance(spec, int):
            return spec
        elif isinstance(spec, str):
            layer = get_layer_by_name(model, spec, exact=True)
            if layer is None:
                raise ValueError(f"Layer '{spec}' not found")
            return model.layers.index(layer)
        else:
            raise TypeError("Layer spec must be int or str")
    
    start_idx = get_layer_index(start_layer)
    if start_idx < 0:
        start_idx = len(model.layers) + start_idx
    
    if end_layer is None:
        end_idx = len(model.layers) - 1
    else:
        end_idx = get_layer_index(end_layer)
        if end_idx < 0:
            end_idx = len(model.layers) + end_idx
    
    # Vérifier les indices
    if start_idx < 0 or start_idx >= len(model.layers):
        raise IndexError(f"Start layer index {start_idx} out of range")
    if end_idx < 0 or end_idx >= len(model.layers):
        raise IndexError(f"End layer index {end_idx} out of range")
    if start_idx > end_idx:
        raise ValueError(f"Start layer ({start_idx}) must be before end layer ({end_idx})")
    
    # Extraire les couches
    from .sequential import Sequential
    sub_layers = model.layers[start_idx:end_idx + 1]
    
    # Créer le sous-modèle
    submodel = Sequential(sub_layers, 
                         name=f"{model.name}_sub_{start_idx}_{end_idx}")
    
    # Copier les paramètres si possible
    for i, (orig_layer, sub_layer) in enumerate(zip(sub_layers, submodel.layers)):
        if hasattr(orig_layer, 'parameters') and hasattr(sub_layer, 'set_parameters'):
            try:
                sub_layer.set_parameters(orig_layer.get_parameters())
            except:
                pass
    
    # Essayer de déterminer l'input shape
    if start_idx == 0 and hasattr(model, '_input_shape'):
        submodel._input_shape = model._input_shape
    elif start_idx > 0:
        # Essayer d'inférer à partir de la couche précédente
        prev_layer = model.layers[start_idx - 1]
        if hasattr(prev_layer, 'output_shape'):
            submodel._input_shape = prev_layer.output_shape
    
    return submodel