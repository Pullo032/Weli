"""
Classe de base pour les fonctions de perte de Weli.
"""
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
import warnings

class Loss:
    """
    Classe de base abstraite pour toutes les fonctions de perte.
    
    Une fonction de perte doit implémenter:
    - forward(): calcul de la perte
    - backward(): calcul du gradient
    
    Attributes:
        name (str): Nom de la fonction de perte
        reduction (str): Méthode de réduction ('mean', 'sum', 'none')
        y_pred (np.ndarray): Prédictions stockées pour backward
        y_true (np.ndarray): Vérité terrain stockée pour backward
        loss_value (Union[float, np.ndarray]): Valeur de la perte calculée
        epsilon (float): Petite valeur pour stabilité numérique
    """
    
    def __init__(self, name: Optional[str] = None, 
                 reduction: str = 'mean',
                 epsilon: float = 1e-7):
        """
        Initialise une fonction de perte.
        
        Args:
            name: Nom de la fonction de perte (par défaut: nom de la classe)
            reduction: Méthode de réduction:
                - 'mean': moyenne sur le batch
                - 'sum': somme sur le batch
                - 'none': pas de réduction
            epsilon: Petite valeur pour stabilité numérique
        """
        self.name = name or self.__class__.__name__
        
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Reduction '{reduction}' invalide. "
                           f"Utilisez 'mean', 'sum', ou 'none'.")
        self.reduction = reduction
        
        self.epsilon = epsilon
        
        # Variables stockées pour backward
        self.y_pred = None
        self.y_true = None
        self.loss_value = None
        self.batch_size = None
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calcule la perte (alias pour forward).
        
        Args:
            y_pred: Prédictions du modèle
            y_true: Vérité terrain (labels)
            
        Returns:
            Valeur de la perte (scalaire ou tableau selon reduction)
        """
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> Union[float, np.ndarray]:
        """
        Calcule la perte. Méthode abstraite à implémenter.
        
        Args:
            y_pred: Prédictions du modèle
            y_true: Vérité terrain
            
        Returns:
            Valeur de la perte
            
        Raises:
            NotImplementedError: Si non implémentée
        """
        raise NotImplementedError(
            f"La classe {self.__class__.__name__} doit implémenter forward()"
        )
    
    def backward(self) -> np.ndarray:
        """
        Calcule le gradient de la perte par rapport aux prédictions.
        Méthode abstraite à implémenter.
        
        Returns:
            Gradient de la perte par rapport à y_pred
            
        Raises:
            NotImplementedError: Si non implémentée
            RuntimeError: Si forward() n'a pas été appelé avant
        """
        raise NotImplementedError(
            f"La classe {self.__class__.__name__} doit implémenter backward()"
        )
    
    def _apply_reduction(self, loss_array: np.ndarray) -> Union[float, np.ndarray]:
        """
        Applique la réduction spécifiée à un tableau de pertes.
        
        Args:
            loss_array: Tableau de pertes individuelles (batch_size, ...)
            
        Returns:
            Perte réduite selon self.reduction
            
        Example:
            >>> loss._apply_reduction(np.array([1.0, 2.0, 3.0]))
            2.0  # si reduction='mean'
        """
        if self.reduction == 'mean':
            return np.mean(loss_array)
        elif self.reduction == 'sum':
            return np.sum(loss_array)
        elif self.reduction == 'none':
            return loss_array
        else:
            # Ne devrait jamais arriver (vérifié dans __init__)
            raise ValueError(f"Réduction '{self.reduction}' non supportée")
    
    def _validate_inputs(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        Valide les inputs y_pred et y_true.
        
        Args:
            y_pred: Prédictions
            y_true: Vérité terrain
            
        Raises:
            ValueError: Si les inputs sont invalides
        """
        # Vérifier que ce sont des numpy arrays
        if not isinstance(y_pred, np.ndarray):
            raise TypeError(f"y_pred doit être un numpy array, got {type(y_pred)}")
        if not isinstance(y_true, np.ndarray):
            raise TypeError(f"y_true doit être un numpy array, got {type(y_true)}")
        
        # Vérifier les shapes
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shapes mismatch: y_pred {y_pred.shape} != y_true {y_true.shape}"
            )
        
        # Vérifier qu'il y a au moins un échantillon
        if y_pred.size == 0:
            raise ValueError("Les inputs ne peuvent pas être vides")
    
    def _prepare_inputs(self, y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les inputs pour le calcul.
        
        Args:
            y_pred: Prédictions
            y_true: Vérité terrain
            
        Returns:
            Tuple (y_pred_prepared, y_true_prepared)
        """
        # Stocker les inputs originaux
        self.y_pred = y_pred.copy()
        self.y_true = y_true.copy()
        
        # Déterminer la taille du batch
        if y_pred.ndim == 0:
            self.batch_size = 1
        else:
            self.batch_size = y_pred.shape[0]
        
        return y_pred, y_true
    
    def _check_backward_prerequisites(self) -> None:
        """
        Vérifie que forward() a été appelé avant backward().
        
        Raises:
            RuntimeError: Si forward() n'a pas été appelé
        """
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError(
                "Vous devez appeler forward() avant backward(). "
                "La fonction de perte a besoin des prédictions et labels."
            )
    
    def _apply_reduction_to_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        Applique la réduction au gradient si nécessaire.
        
        Args:
            gradient: Gradient non réduit
            
        Returns:
            Gradient ajusté selon la réduction
        """
        if self.reduction == 'mean' and self.batch_size is not None:
            return gradient / self.batch_size
        return gradient
    
    def compute(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Calcule la perte et retourne des informations détaillées.
        
        Args:
            y_pred: Prédictions
            y_true: Vérité terrain
            
        Returns:
            Dictionnaire avec:
                - 'loss': valeur de la perte
                - 'gradient': gradient calculé
                - 'reduction': méthode de réduction utilisée
                - 'batch_size': taille du batch
        """
        # Calcul de la perte
        loss = self.forward(y_pred, y_true)
        
        # Calcul du gradient
        gradient = self.backward()
        
        return {
            'loss': loss,
            'gradient': gradient,
            'reduction': self.reduction,
            'batch_size': self.batch_size,
            'loss_name': self.name
        }
    
    def reset(self) -> None:
        """
        Réinitialise l'état de la fonction de perte.
        """
        self.y_pred = None
        self.y_true = None
        self.loss_value = None
        self.batch_size = None
    
    def get_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration de la fonction de perte.
        
        Returns:
            Dictionnaire de configuration
        """
        return {
            'name': self.name,
            'class_name': self.__class__.__name__,
            'reduction': self.reduction,
            'epsilon': self.epsilon
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Loss':
        """
        Crée une instance de Loss depuis une configuration.
        
        Args:
            config: Configuration
            
        Returns:
            Instance de Loss
        """
        # Extraire les arguments
        name = config.get('name')
        reduction = config.get('reduction', 'mean')
        epsilon = config.get('epsilon', 1e-7)
        
        # Créer l'instance
        return cls(name=name, reduction=reduction, epsilon=epsilon)
    
    def __repr__(self) -> str:
        """
        Représentation textuelle de la fonction de perte.
        """
        return f"{self.__class__.__name__}(name='{self.name}', reduction='{self.reduction}')"
    
    def __str__(self) -> str:
        """
        String lisible de la fonction de perte.
        """
        return f"Loss: {self.name} ({self.__class__.__name__})"
    
    def summary(self) -> str:
        """
        Retourne un résumé de la fonction de perte.
        
        Returns:
            String formaté avec les informations
        """
        lines = []
        lines.append("=" * 50)
        lines.append(f"LOSS FUNCTION: {self.name}")
        lines.append("=" * 50)
        lines.append(f"Class: {self.__class__.__name__}")
        lines.append(f"Reduction: {self.reduction}")
        lines.append(f"Epsilon: {self.epsilon}")
        lines.append(f"State: {'Initialized' if self.y_pred is not None else 'Not initialized'}")
        
        if self.y_pred is not None:
            lines.append(f"Last batch size: {self.batch_size}")
            lines.append(f"y_pred shape: {self.y_pred.shape}")
            lines.append(f"y_true shape: {self.y_true.shape}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class LossFunctionRegistry:
    """
    Registry pour gérer les fonctions de perte.
    """
    
    def __init__(self):
        self._losses = {}
        self._register_default_losses()
    
    def _register_default_losses(self) -> None:
        """Enregistre les fonctions de perte par défaut."""
        # Ces imports seront ajoutés plus tard
        pass
    
    def register(self, name: str, loss_class: type) -> None:
        """
        Enregistre une fonction de perte.
        
        Args:
            name: Nom de la fonction de perte
            loss_class: Classe de la fonction de perte
        """
        if not issubclass(loss_class, Loss):
            raise TypeError(f"loss_class doit être une sous-classe de Loss, got {loss_class}")
        
        self._losses[name.lower()] = loss_class
    
    def get(self, name: str, **kwargs) -> Loss:
        """
        Crée une instance de fonction de perte.
        
        Args:
            name: Nom de la fonction de perte
            **kwargs: Arguments pour le constructeur
            
        Returns:
            Instance de la fonction de perte
            
        Raises:
            KeyError: Si la fonction de perte n'est pas enregistrée
        """
        name_lower = name.lower()
        
        if name_lower not in self._losses:
            # Essayer d'importer dynamiquement
            try:
                self._try_import_loss(name)
                name_lower = name.lower()
            except ImportError:
                available = ', '.join(self._losses.keys())
                raise KeyError(
                    f"Fonction de perte '{name}' non trouvée. "
                    f"Disponibles: {available}"
                )
        
        loss_class = self._losses[name_lower]
        return loss_class(**kwargs)
    
    def _try_import_loss(self, name: str) -> None:
        """
        Essaie d'importer une fonction de perte dynamiquement.
        """
        # Mapping des noms aux modules
        loss_mapping = {
            'mse': ('regression', 'MSE'),
            'mae': ('regression', 'MAE'),
            'huber': ('regression', 'HuberLoss'),
            'crossentropy': ('classification', 'CrossEntropy'),
            'binarycrossentropy': ('classification', 'BinaryCrossEntropy'),
        }
        
        name_lower = name.lower()
        
        if name_lower in loss_mapping:
            module_name, class_name = loss_mapping[name_lower]
            
            # Import dynamique
            module = __import__(f'weli.losses.{module_name}', 
                              fromlist=[class_name])
            loss_class = getattr(module, class_name)
            
            # Enregistrer
            self.register(name_lower, loss_class)
    
    def list(self) -> list:
        """
        Liste toutes les fonctions de perte enregistrées.
        
        Returns:
            Liste des noms de fonctions de perte
        """
        return list(self._losses.keys())
    
    def create_from_config(self, config: Dict[str, Any]) -> Loss:
        """
        Crée une fonction de perte depuis une configuration.
        
        Args:
            config: Configuration
            
        Returns:
            Instance de Loss
        """
        if 'class_name' not in config:
            raise ValueError("Configuration must contain 'class_name'")
        
        class_name = config['class_name']
        return self.get(class_name, **config.get('config', {}))


# Instance globale du registry
loss_registry = LossFunctionRegistry()


class CombinedLoss(Loss):
    """
    Combinaison de plusieurs fonctions de perte.
    
    Formule: L = Σ w_i * L_i
    """
    
    def __init__(self, losses: list, weights: Optional[list] = None,
                 name: Optional[str] = None, reduction: str = 'mean'):
        """
        Initialise une combinaison de pertes.
        
        Args:
            losses: Liste de fonctions de perte ou de noms
            weights: Liste de poids (doit avoir même longueur que losses)
            name: Nom de la combinaison
            reduction: Méthode de réduction
        """
        super().__init__(name or 'CombinedLoss', reduction)
        
        # Convertir les noms en instances si nécessaire
        self.losses = []
        for loss in losses:
            if isinstance(loss, str):
                self.losses.append(loss_registry.get(loss))
            elif isinstance(loss, Loss):
                self.losses.append(loss)
            else:
                raise TypeError(f"loss doit être str ou Loss, got {type(loss)}")
        
        # Définir les poids
        if weights is None:
            self.weights = [1.0 / len(losses)] * len(losses)
        else:
            if len(weights) != len(losses):
                raise ValueError("weights doit avoir même longueur que losses")
            self.weights = weights
        
        # Normaliser les poids
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
        
        # Stockage pour backward
        self.loss_values = []
        self.gradients = []
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> Union[float, np.ndarray]:
        """Calcule la perte combinée."""
        self._prepare_inputs(y_pred, y_true)
        self.loss_values = []
        
        total_loss = 0
        
        for loss, weight in zip(self.losses, self.weights):
            loss_value = loss.forward(y_pred, y_true)
            self.loss_values.append(loss_value)
            
            # Appliquer le poids
            if isinstance(loss_value, np.ndarray):
                weighted_loss = weight * loss_value
            else:
                weighted_loss = weight * loss_value
            
            total_loss += weighted_loss
        
        # Appliquer la réduction finale
        if isinstance(total_loss, np.ndarray):
            self.loss_value = self._apply_reduction(total_loss)
        else:
            self.loss_value = total_loss
        
        return self.loss_value
    
    def backward(self) -> np.ndarray:
        """Calcule le gradient combiné."""
        self._check_backward_prerequisites()
        self.gradients = []
        
        total_gradient = None
        
        for loss, weight in zip(self.losses, self.weights):
            gradient = loss.backward()
            self.gradients.append(gradient)
            
            weighted_gradient = weight * gradient
            
            if total_gradient is None:
                total_gradient = weighted_gradient
            else:
                total_gradient += weighted_gradient
        
        # Appliquer la réduction
        total_gradient = self._apply_reduction_to_gradient(total_gradient)
        
        return total_gradient
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration."""
        config = super().get_config()
        config.update({
            'losses': [
                loss.get_config() if isinstance(loss, Loss) else loss
                for loss in self.losses
            ],
            'weights': self.weights
        })
        return config
    
    def summary(self) -> str:
        """Résumé détaillé."""
        lines = []
        lines.append(super().summary())
        lines.append("\nCOMPOSITION:")
        lines.append("-" * 50)
        
        for i, (loss, weight) in enumerate(zip(self.losses, self.weights)):
            lines.append(f"  [{i}] {loss.name} (weight: {weight:.3f})")
            if i < len(self.loss_values):
                lines.append(f"       Loss value: {self.loss_values[i]:.6f}")
        
        return "\n".join(lines)


# Enregistrer les classes de base dans le registry
loss_registry.register('loss', Loss)
loss_registry.register('combinedloss', CombinedLoss)