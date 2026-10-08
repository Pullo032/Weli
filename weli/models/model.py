import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class Model:
    """
    Classe de base pour tous les modèles de Weli.
    
    Cette classe définit l'interface commune que tous les modèles doivent implémenter.
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialise un modèle.
        
        Args:
            name: Nom du modèle
        """
        self.name = name or self.__class__.__name__
        self.layers = []
        self.trainable_layers = []
        self.initialized = False
        self.training = True
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        self._input_shape = None
        self._output_shape = None
    
    def add(self, layer):
        """
        Ajoute une couche au modèle.
        
        Args:
            layer: Couche à ajouter
            
        Returns:
            self pour le chaînage
        """
        self.layers.append(layer)
        if layer.trainable:
            self.trainable_layers.append(layer)
        return self
    
    def initialize(self, input_shape: Tuple):
        """
        Initialise toutes les couches du modèle.
        
        Args:
            input_shape: Shape de l'input (sans batch_size)
            
        Returns:
            Shape de l'output
        """
        self._input_shape = tuple(input_shape)
        current_shape = (None,) + tuple(input_shape)
        
        # Initialiser chaque couche
        for layer in self.layers:
            current_shape = layer.initialize(current_shape)
        
        self._output_shape = current_shape[1:]
        self.initialized = True
        return self._output_shape
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Propagation avant à travers toutes les couches.
        
        Args:
            x: Input du modèle
            training: Si True, mode entraînement
            
        Returns:
            Sortie du modèle
        """
        if not self.initialized:
            # Inférer la shape si nécessaire
            if len(x.shape) > 1:
                self.initialize(x.shape[1:])
            else:
                self.initialize((x.shape[0],))
        
        # Définir le mode pour toutes les couches
        self.training = training
        for layer in self.layers:
            layer.training = training
        
        # Propagation avant
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Rétropropagation à travers toutes les couches.
        
        Args:
            dout: Gradient de la loss par rapport à la sortie
            
        Returns:
            Gradient par rapport à l'input
        """
        # Backward à travers les couches dans l'ordre inverse
        gradient = dout
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        
        return gradient
    
    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Permet d'appeler le modèle comme une fonction: model(x)"""
        return self.forward(x, training)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Prédiction (mode inference).
        
        Args:
            x: Données d'input
            
        Returns:
            Prédictions
        """
        return self.forward(x, training=False)
    
    def train_step(self, x_batch: np.ndarray, y_batch: np.ndarray, 
                   loss_fn, optimizer) -> Tuple[float, float]:
        """
        Effectue une étape d'entraînement.
        
        Args:
            x_batch: Batch d'inputs
            y_batch: Batch de targets
            loss_fn: Fonction de perte
            optimizer: Optimiseur
            
        Returns:
            Tuple (loss, accuracy)
        """
        # Forward pass
        predictions = self.forward(x_batch, training=True)
        
        # Calcul de la perte
        loss = loss_fn.forward(predictions, y_batch)
        
        # Calcul de l'accuracy (si classification)
        if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
            # Classification multi-classes
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_batch, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
        elif len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
            # Régression ou classification binaire
            pred_binary = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_binary == y_batch)
        else:
            # Régression simple
            accuracy = 0.0
        
        # Backward pass
        dout = loss_fn.backward()
        self.backward(dout)
        
        # Mise à jour des paramètres
        optimizer.step(self.trainable_layers)
        
        # Réinitialiser les gradients
        self.zero_grad()
        
        return loss, accuracy
    
    def val_step(self, x_batch: np.ndarray, y_batch: np.ndarray, 
                 loss_fn) -> Tuple[float, float]:
        """
        Effectue une étape de validation.
        
        Args:
            x_batch: Batch d'inputs
            y_batch: Batch de targets
            loss_fn: Fonction de perte
            
        Returns:
            Tuple (loss, accuracy)
        """
        # Forward pass (mode inference)
        predictions = self.predict(x_batch)
        
        # Calcul de la perte
        loss = loss_fn.forward(predictions, y_batch)
        
        # Calcul de l'accuracy
        if len(y_batch.shape) > 1 and y_batch.shape[1] > 1:
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y_batch, axis=1)
            accuracy = np.mean(pred_classes == true_classes)
        elif len(y_batch.shape) > 1 and y_batch.shape[1] == 1:
            pred_binary = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_binary == y_batch)
        else:
            accuracy = 0.0
        
        return loss, accuracy
    
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
        Entraîne le modèle.
        
        Args:
            x_train: Données d'entraînement
            y_train: Labels d'entraînement
            epochs: Nombre d'époques
            batch_size: Taille des batches
            validation_data: Tuple (x_val, y_val) pour la validation
            loss_fn: Fonction de perte
            optimizer: Optimiseur
            verbose: Niveau de verbosité (0, 1, 2)
            shuffle: Si True, mélange les données
            
        Returns:
            Historique d'entraînement
        """
        if loss_fn is None or optimizer is None:
            raise ValueError("loss_fn and optimizer must be provided")
        
        n_samples = x_train.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Réinitialiser l'historique
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        for epoch in range(epochs):
            # Mélanger les données si demandé
            if shuffle:
                indices = np.random.permutation(n_samples)
                x_train = x_train[indices]
                y_train = y_train[indices]
            
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            
            # Entraînement par batch
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                
                x_batch = x_train[start:end]
                y_batch = y_train[start:end]
                
                # Step d'entraînement
                batch_loss, batch_acc = self.train_step(
                    x_batch, y_batch, loss_fn, optimizer
                )
                
                epoch_train_loss += batch_loss
                epoch_train_acc += batch_acc
            
            # Moyennes sur l'époque
            epoch_train_loss /= n_batches
            epoch_train_acc /= n_batches
            
            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(x_val, y_val, loss_fn, batch_size)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Stocker les métriques
            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            
            # Affichage
            if verbose > 0:
                if epoch % verbose == 0 or epoch == epochs - 1:
                    msg = f"Epoch {epoch+1}/{epochs} - "
                    msg += f"loss: {epoch_train_loss:.4f} - acc: {epoch_train_acc:.4f}"
                    if validation_data is not None:
                        msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                    print(msg)
        
        return self.history
    
    def evaluate(self, 
                 x_test: np.ndarray, 
                 y_test: np.ndarray,
                 loss_fn,
                 batch_size: int = 32) -> Tuple[float, float]:
        """
        Évalue le modèle sur des données de test.
        
        Args:
            x_test: Données de test
            y_test: Labels de test
            loss_fn: Fonction de perte
            batch_size: Taille des batches
            
        Returns:
            Tuple (loss, accuracy)
        """
        n_samples = x_test.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        total_loss = 0.0
        total_acc = 0.0
        
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_samples)
            
            x_batch = x_test[start:end]
            y_batch = y_test[start:end]
            
            # Step de validation
            batch_loss, batch_acc = self.val_step(x_batch, y_batch, loss_fn)
            
            total_loss += batch_loss
            total_acc += batch_acc
        
        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        
        return avg_loss, avg_acc
    
    def zero_grad(self):
        """Réinitialise tous les gradients à zéro."""
        for layer in self.trainable_layers:
            layer.zero_grad()
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Récupère tous les paramètres du modèle.
        
        Returns:
            Dictionnaire des paramètres
        """
        params = {}
        for i, layer in enumerate(self.trainable_layers):
            layer_params = layer.get_parameters()
            for key, value in layer_params.items():
                params[f'layer_{i}_{key}'] = value
        return params
    
    def set_parameters(self, params: Dict[str, np.ndarray]):
        """
        Définit les paramètres du modèle.
        
        Args:
            params: Dictionnaire des paramètres
        """
        for i, layer in enumerate(self.trainable_layers):
            layer_params = {}
            for key, value in params.items():
                if key.startswith(f'layer_{i}_'):
                    param_name = key.replace(f'layer_{i}_', '')
                    layer_params[param_name] = value
            
            if layer_params:
                layer.set_parameters(layer_params)
    
    def save_weights(self, filepath: str):
        """
        Sauvegarde les poids du modèle.
        
        Args:
            filepath: Chemin du fichier
        """
        params = self.get_parameters()
        np.savez(filepath, **params)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath: str):
        """
        Charge les poids du modèle.
        
        Args:
            filepath: Chemin du fichier
        """
        data = np.load(filepath)
        params = {key: data[key] for key in data.files}
        self.set_parameters(params)
        print(f"Model weights loaded from {filepath}")
    
    def save(self, filepath: str):
        """
        Sauvegarde le modèle complet (architecture + poids).
        
        Args:
            filepath: Chemin du fichier
        """
        from .save_load import save_model
        save_model(self, filepath)
    
    @classmethod
    def load(cls, filepath: str):
        """
        Charge un modèle sauvegardé.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Modèle chargé
        """
        from .save_load import load_model
        model = load_model(filepath, compile=False)
        if not isinstance(model, cls):
            raise ValueError(f"Saved model is {type(model).__name__}, not {cls.__name__}")
        return model
    
    def summary(self):
        """
        Affiche un résumé de l'architecture du modèle.
        """
        print(f"Model: {self.name}")
        print("=" * 60)
        print(f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10}")
        print("=" * 60)
        
        total_params = 0
        current_shape = self._input_shape
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.name} ({layer.__class__.__name__})"
            
            # Calculer les paramètres
            layer_params = 0
            if hasattr(layer, 'parameters'):
                for param in layer.parameters.values():
                    layer_params += param.size
            
            total_params += layer_params
            
            # Mettre à jour la shape
            if hasattr(layer, 'output_shape'):
                current_shape = layer.output_shape
            
            print(f"{layer_name:<20} {str(current_shape):<20} {layer_params:<10}")
        
        print("=" * 60)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {total_params}")
        print(f"Non-trainable params: 0")
        print(f"Input shape: {self._input_shape}")
        print(f"Output shape: {self._output_shape}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, layers={len(self.layers)})"