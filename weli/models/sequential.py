import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .model import Model

class Sequential(Model):
    """
    Modèle Sequential - Empilement linéaire de couches.
    
    C'est le type de modèle le plus courant où les couches
    sont empilées les unes après les autres.
    """
    
    def __init__(self, layers: Optional[List] = None, name: Optional[str] = None):
        """
        Initialise un modèle Sequential.
        
        Args:
            layers: Liste de couches (optionnel)
            name: Nom du modèle
        """
        super().__init__(name)
        
        if layers is not None:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer):
        """
        Ajoute une couche à la fin du modèle.
        
        Args:
            layer: Couche à ajouter
            
        Returns:
            self pour le chaînage
        """
        super().add(layer)
        
        # Si le modèle est déjà initialisé, mettre à jour la shape
        if self.initialized and self.layers[-1] == layer:
            # Réinitialiser à partir de la nouvelle couche
            self.initialize(self._input_shape)
        
        return self
    
    def __getitem__(self, index):
        """
        Permet d'accéder aux couches par index.
        
        Args:
            index: Index ou slice
            
        Returns:
            Couche(s) correspondante(s)
        """
        return self.layers[index]
    
    def __len__(self):
        """Retourne le nombre de couches."""
        return len(self.layers)
    
    def compile(self, 
                input_shape: Optional[Tuple] = None,
                loss_fn = None,
                optimizer = None):
        """
        Compile le modèle (prépare pour l'entraînement).
        
        Args:
            input_shape: Shape de l'input (sans batch_size)
            loss_fn: Fonction de perte
            optimizer: Optimiseur
        """
        if input_shape is not None:
            self.initialize(input_shape)
        
        # Vérifier que le modèle est initialisé
        if not self.initialized:
            raise ValueError("Model must be initialized before compilation. "
                           "Provide input_shape or call initialize() first.")
        
        # Ces attributs sont stockés pour être utilisés dans fit()
        self._loss_fn = loss_fn
        self._optimizer = optimizer
    
    def fit(self, 
            x_train: np.ndarray, 
            y_train: np.ndarray,
            epochs: int = 10,
            batch_size: int = 32,
            validation_data: Optional[Tuple] = None,
            loss_fn = None,
            optimizer = None,
            verbose: int = 1,
            shuffle: bool = True) -> Dict[str, List]:
        """
        Entraîne le modèle Sequential.
        
        Utilise la loss_fn et l'optimizer fournis en paramètre
        ou ceux définis avec compile().
        """
        # Utiliser ceux fournis en paramètre ou ceux de compile()
        if loss_fn is None:
            if hasattr(self, '_loss_fn'):
                loss_fn = self._loss_fn
            else:
                raise ValueError("loss_fn must be provided either in fit() or compile()")
        
        if optimizer is None:
            if hasattr(self, '_optimizer'):
                optimizer = self._optimizer
            else:
                raise ValueError("optimizer must be provided either in fit() or compile()")
        
        return super().fit(
            x_train, y_train, epochs, batch_size,
            validation_data, loss_fn, optimizer, verbose, shuffle
        )
    
    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 loss_fn = None,
                 batch_size: int = 32) -> Tuple[float, float]:
        """
        Évalue le modèle Sequential.
        """
        if loss_fn is None:
            if hasattr(self, '_loss_fn'):
                loss_fn = self._loss_fn
            else:
                raise ValueError("loss_fn must be provided")
        
        return super().evaluate(x_test, y_test, loss_fn, batch_size)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prédiction avec le modèle Sequential.
        
        Args:
            x: Données d'input
            
        Returns:
            Prédictions
        """
        return super().predict(x)
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle Sequential complet.
        """
        from .save_load import save_model
        save_model(self, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Charge un modèle Sequential sauvegardé.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Modèle Sequential chargé
        """
        from .save_load import load_model
        model = load_model(filepath, compile=False)
        if not isinstance(model, cls):
            raise ValueError(f"Saved model is {type(model).__name__}, not {cls.__name__}")
        return model
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Crée un modèle Sequential à partir d'une configuration.
        
        Args:
            config: Configuration du modèle
            
        Returns:
            Modèle Sequential
        """
        model = cls(name=config.get('name'))
        from .registry import registry
        layer_configs = config.get('layer_configs', config.get('layers', []))
        for layer_config in layer_configs:
            model.add(registry.deserialize_layer(layer_config))
        input_shape = config.get('input_shape')
        if input_shape is not None:
            model.initialize(tuple(input_shape))
        return model
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration du modèle.
        
        Returns:
            Configuration
        """
        return {
            'name': self.name,
            'layers': [layer.get_config() for layer in self.layers],
            'layer_configs': [layer.get_config() for layer in self.layers],
            'input_shape': self._input_shape,
            'output_shape': self._output_shape,
            'class_name': self.__class__.__name__
        }