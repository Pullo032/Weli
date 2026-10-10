"""
Modèles pour le framework Weli.
"""

from .model import Model
from .sequential import Sequential
from .functional import (
    Functional, Input, 
    add, concatenate, multiply,
    create_residual_block, create_inception_module
)

# Import conditionnel des wrappers
try:
    from .functional import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNorm2D
    _FUNCTIONAL_WRAPPERS_AVAILABLE = True
except ImportError:
    _FUNCTIONAL_WRAPPERS_AVAILABLE = False

# Import des conteneurs
from .container import ModelContainer, Parallel

# Import du registry
from .registry import registry, ModelRegistry

# Import save/load
from .save_load import (
    save_model, load_model, load_weights,
    save_model_architecture, model_to_dict,
    get_model_info, check_model_compatibility,
    ModelSaveError, ModelLoadError
)

# Import visualisation
from .visualization import (
    plot_model_architecture, plot_training_history,
    plot_layer_activations, visualize_weights_distribution,
    generate_model_summary_table, export_architecture_to_html
)

# Import export
from .export import (
    export_to_onnx, export_to_tflite, export_to_coreml,
    generate_c_code, export_parameters_csv
)

# Import utils
from .utils import (
    count_parameters, print_model_summary,
    get_layer_by_name, get_layer_output,
    model_to_json, model_from_json
)

# Modèles pré-construits
from .prebuilt import (
    resnet18,
    text_classifier_gru,
    text_classifier_lstm,
    generator_mlp,
    discriminator_mlp,
    gan_mlp,
    text_to_image_mlp,
    text_to_video_mlp,
)

__all__ = [
    # Classes de base
    'Model',
    'Sequential',
    'Functional',
    
    # Composants fonctionnels
    'Input',
    'add',
    'concatenate',
    'multiply',
    'create_residual_block',
    'create_inception_module',
    
    # Conteneurs
    'ModelContainer',
    'Parallel',
    
    # Registry
    'registry',
    'ModelRegistry',
    
    # Save/Load
    'save_model',
    'load_model',
    'load_weights',
    'save_model_architecture',
    'model_to_dict',
    'get_model_info',
    'check_model_compatibility',
    'ModelSaveError',
    'ModelLoadError',
    
    # Visualisation
    'plot_model_architecture',
    'plot_training_history',
    'plot_layer_activations',
    'visualize_weights_distribution',
    'generate_model_summary_table',
    'export_architecture_to_html',
    
    # Export
    'export_to_onnx',
    'export_to_tflite',
    'export_to_coreml',
    'generate_c_code',
    'export_parameters_csv',
    
    # Utils
    'count_parameters',
    'print_model_summary',
    'get_layer_by_name',
    'get_layer_output',
    'model_to_json',
    'model_from_json',

    # Modèles pré-construits
    'resnet18',
    'text_classifier_gru',
    'text_classifier_lstm',
    'generator_mlp',
    'discriminator_mlp',
    'gan_mlp',
    'text_to_image_mlp',
    'text_to_video_mlp',
]

# Ajouter les wrappers fonctionnels si disponibles
if _FUNCTIONAL_WRAPPERS_AVAILABLE:
    __all__.extend(['Dense', 'Conv2D', 'MaxPool2D', 'Flatten', 'Dropout', 'BatchNorm2D'])

# Version
__version__ = '1.0.0'

# Fonctions utilitaires supplémentaires
def create_sequential(layers=None, name="SequentialModel"):
    """
    Crée un modèle Sequential.
    
    Args:
        layers: Liste de couches (optionnel)
        name: Nom du modèle
        
    Returns:
        Instance de Sequential
    """
    return Sequential(layers, name=name)

def create_model_from_config(config):
    """
    Crée un modèle à partir d'une configuration.
    
    Args:
        config: Configuration du modèle
        
    Returns:
        Modèle créé
    """
    # Pour l'instant, simple création Sequential
    # À améliorer pour supporter différents types de modèles
    if 'layers' in config:
        return Sequential(config['layers'], name=config.get('name', 'Model'))
    raise ValueError("Configuration de modèle non supportée")

# Ajouter ces fonctions aux exports
__all__.extend(['create_sequential', 'create_model_from_config'])

# Initialisation du registry
def _init():
    """Initialise le module models."""
    registry.register_model('Model', Model)
    registry.register_model('Sequential', Sequential)
    registry.register_model('Functional', Functional)
    registry.register_model('ModelContainer', ModelContainer)
    registry.register_model('Parallel', Parallel)

# Exécuter l'initialisation
_init()