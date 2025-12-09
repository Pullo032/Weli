import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .base import Layer
from .activation import Tanh, Sigmoid

class SimpleRNN(Layer):
    """
    Couche RNN (Recurrent Neural Network) simple.
    
    Formules:
        h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b)
        y_t = h_t
    
    Args:
        units: Nombre de cellules/neurones RNN
        activation: Fonction d'activation ('tanh' ou 'sigmoid')
        return_sequences: Si True, retourne toutes les sorties de la séquence
        return_state: Si True, retourne aussi le dernier état caché
        name: Nom de la couche
    """
    
    def __init__(self,
                 units: int,
                 activation: str = 'tanh',
                 return_sequences: bool = False,
                 return_state: bool = False,
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.units = units
        self.activation_name = activation
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Choix de l'activation
        if activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        else:
            raise ValueError(f"RNN activation must be 'tanh' or 'sigmoid', got {activation}")
        
        # Caches pour la backward pass
        self.x_cache = []  # Inputs à chaque pas de temps
        self.h_cache = []  # États cachés à chaque pas de temps
        self.a_cache = []  # Activations avant activation
    
    def initialize(self, input_shape: tuple) -> tuple:
        """
        Initialise les paramètres de la couche RNN.
        
        Args:
            input_shape: (batch_size, timesteps, input_dim)
            
        Returns:
            output_shape: (batch_size, timesteps, units) si return_sequences=True
                         sinon (batch_size, units)
        """
        if len(input_shape) != 3:
            raise ValueError(f"RNN expects 3D input (batch, timesteps, features), got {len(input_shape)}D")
        
        batch_size, timesteps, input_dim = input_shape
        
        # Initialisation des poids (Xavier/Glorot pour tanh)
        limit = np.sqrt(6.0 / (input_dim + self.units))
        
        # W_x: poids pour l'input
        self.parameters['W_x'] = np.random.uniform(-limit, limit, (input_dim, self.units))
        # W_h: poids pour l'état caché précédent
        self.parameters['W_h'] = np.random.uniform(-limit, limit, (self.units, self.units))
        # b: biais
        self.parameters['b'] = np.zeros((1, self.units))
        
        # Initialisation des gradients
        self.gradients['W_x'] = np.zeros_like(self.parameters['W_x'])
        self.gradients['W_h'] = np.zeros_like(self.parameters['W_h'])
        self.gradients['b'] = np.zeros_like(self.parameters['b'])
        
        # Définition de la shape de sortie
        if self.return_sequences:
            self.output_shape = (batch_size, timesteps, self.units)
        else:
            self.output_shape = (batch_size, self.units)
            
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagation avant pour la RNN.
        
        Args:
            x: Input de shape (batch_size, timesteps, input_dim)
            
        Returns:
            Sortie de la RNN
        """
        batch_size, timesteps, input_dim = x.shape
        
        # Réinitialiser les caches
        self.x_cache = []
        self.h_cache = []
        self.a_cache = []
        
        # Initialiser l'état caché à zéro
        h_prev = np.zeros((batch_size, self.units))
        self.h_cache.append(h_prev)  # h_0
        
        # Liste pour stocker les sorties à chaque pas de temps
        outputs = []
        
        # Boucle sur les pas de temps
        for t in range(timesteps):
            # Input au pas de temps t
            x_t = x[:, t, :]
            self.x_cache.append(x_t)
            
            # Calcul de l'activation
            a_t = x_t @ self.parameters['W_x'] + h_prev @ self.parameters['W_h'] + self.parameters['b']
            self.a_cache.append(a_t)
            
            # Application de l'activation
            h_t = self.activation.forward(a_t)
            self.h_cache.append(h_t)
            
            # Stocker la sortie si nécessaire
            if self.return_sequences:
                outputs.append(h_t)
            
            # Mettre à jour l'état caché pour le prochain pas de temps
            h_prev = h_t
        
        # Préparer la sortie finale
        if self.return_sequences:
            # Concaténer toutes les sorties sur l'axe du temps
            self.output = np.stack(outputs, axis=1)
        else:
            # Retourner seulement le dernier état caché
            self.output = h_prev
        
        # Si return_state=True, retourner aussi le dernier état
        if self.return_state:
            return self.output, h_prev
        else:
            return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Rétropropagation à travers le temps (BPTT).
        
        Args:
            dout: Gradient de la loss par rapport à la sortie
                 Shape: (batch_size, timesteps, units) si return_sequences=True
                       (batch_size, units) sinon
                 
        Returns:
            Gradient par rapport à l'input: (batch_size, timesteps, input_dim)
        """
        batch_size, timesteps, input_dim = self.x_cache[0].shape[0], len(self.x_cache), self.x_cache[0].shape[1]
        
        # Initialiser les gradients à zéro
        dW_x = np.zeros_like(self.parameters['W_x'])
        dW_h = np.zeros_like(self.parameters['W_h'])
        db = np.zeros_like(self.parameters['b'])
        
        # Initialiser le gradient par rapport à l'input
        dx = np.zeros((batch_size, timesteps, input_dim))
        
        # Initialiser le gradient du prochain état caché
        dh_next = np.zeros((batch_size, self.units))
        
        # Backward à travers le temps (du dernier au premier pas de temps)
        for t in reversed(range(timesteps)):
            # Récupérer les caches
            x_t = self.x_cache[t]
            a_t = self.a_cache[t]
            h_t = self.h_cache[t + 1]
            h_prev = self.h_cache[t]
            
            # Calculer dh_total = dout + dh_next (gradient du pas suivant)
            if self.return_sequences:
                dh_total = dout[:, t, :] + dh_next
            else:
                # Si on ne retourne pas les séquences, dout est seulement pour le dernier pas
                if t == timesteps - 1:
                    dh_total = dout + dh_next
                else:
                    dh_total = dh_next
            
            # Backward à travers l'activation
            da = self.activation.backward(dh_total)
            
            # Gradient par rapport aux poids W_x
            dW_x += x_t.T @ da / batch_size
            
            # Gradient par rapport aux poids W_h
            dW_h += h_prev.T @ da / batch_size
            
            # Gradient par rapport au biais
            db += np.sum(da, axis=0, keepdims=True) / batch_size
            
            # Gradient par rapport à l'input x_t
            dx[:, t, :] = da @ self.parameters['W_x'].T
            
            # Gradient par rapport à l'état caché précédent (pour le pas suivant)
            dh_next = da @ self.parameters['W_h'].T
        
        # Stocker les gradients
        self.gradients['W_x'] = dW_x
        self.gradients['W_h'] = dW_h
        self.gradients['b'] = db
        
        return dx
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration de la couche."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': self.activation_name,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config


class LSTM(Layer):
    """
    Couche LSTM (Long Short-Term Memory).
    
    Formules LSTM:
        f_t = σ(x_t @ W_f + h_{t-1} @ U_f + b_f)  # Forget gate
        i_t = σ(x_t @ W_i + h_{t-1} @ U_i + b_i)  # Input gate
        o_t = σ(x_t @ W_o + h_{t-1} @ U_o + b_o)  # Output gate
        
        c̃_t = tanh(x_t @ W_c + h_{t-1} @ U_c + b_c)  # Cell candidate
        c_t = f_t * c_{t-1} + i_t * c̃_t             # Cell state
        h_t = o_t * tanh(c_t)                       # Hidden state
    
    Args:
        units: Nombre de cellules LSTM
        return_sequences: Si True, retourne toutes les sorties
        return_state: Si True, retourne (output, (h, c))
        name: Nom de la couche
    """
    
    def __init__(self,
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Fonctions d'activation
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        
        # Caches pour la backward pass
        self.x_cache = []
        self.h_cache = []
        self.c_cache = []
        self.f_cache = []  # Forget gate
        self.i_cache = []  # Input gate
        self.o_cache = []  # Output gate
        self.c_tilde_cache = []  # Cell candidate
    
    def initialize(self, input_shape: tuple) -> tuple:
        """
        Initialise les paramètres LSTM.
        
        Args:
            input_shape: (batch_size, timesteps, input_dim)
        """
        if len(input_shape) != 3:
            raise ValueError(f"LSTM expects 3D input, got {len(input_shape)}D")
        
        batch_size, timesteps, input_dim = input_shape
        
        # Initialisation des poids (Xavier pour tanh, normal pour sigmoid)
        limit_tanh = np.sqrt(6.0 / (input_dim + self.units))
        limit_sigmoid = np.sqrt(2.0 / (input_dim + self.units))
        
        # Poids pour les 4 transformations (f, i, o, c)
        for gate in ['f', 'i', 'o', 'c']:
            # W: poids pour l'input
            self.parameters[f'W_{gate}'] = np.random.uniform(
                -limit_sigmoid if gate != 'c' else -limit_tanh, 
                limit_sigmoid if gate != 'c' else limit_tanh,
                (input_dim, self.units)
            )
            # U: poids pour l'état caché
            self.parameters[f'U_{gate}'] = np.random.uniform(
                -limit_sigmoid if gate != 'c' else -limit_tanh,
                limit_sigmoid if gate != 'c' else limit_tanh,
                (self.units, self.units)
            )
            # b: biais
            self.parameters[f'b_{gate}'] = np.zeros((1, self.units))
            
            # Initialisation des gradients
            self.gradients[f'W_{gate}'] = np.zeros_like(self.parameters[f'W_{gate}'])
            self.gradients[f'U_{gate}'] = np.zeros_like(self.parameters[f'U_{gate}'])
            self.gradients[f'b_{gate}'] = np.zeros_like(self.parameters[f'b_{gate}'])
        
        # Shape de sortie
        if self.return_sequences:
            self.output_shape = (batch_size, timesteps, self.units)
        else:
            self.output_shape = (batch_size, self.units)
            
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Propagation avant LSTM.
        
        Args:
            x: (batch_size, timesteps, input_dim)
        """
        batch_size, timesteps, input_dim = x.shape
        
        # Réinitialiser les caches
        self.x_cache = []
        self.h_cache = [np.zeros((batch_size, self.units))]  # h_0
        self.c_cache = [np.zeros((batch_size, self.units))]  # c_0
        self.f_cache = []
        self.i_cache = []
        self.o_cache = []
        self.c_tilde_cache = []
        
        outputs = []
        
        # Boucle sur les pas de temps
        for t in range(timesteps):
            x_t = x[:, t, :]
            self.x_cache.append(x_t)
            
            h_prev = self.h_cache[-1]
            c_prev = self.c_cache[-1]
            
            # Forget gate
            f_t = self.sigmoid.forward(
                x_t @ self.parameters['W_f'] + 
                h_prev @ self.parameters['U_f'] + 
                self.parameters['b_f']
            )
            self.f_cache.append(f_t)
            
            # Input gate
            i_t = self.sigmoid.forward(
                x_t @ self.parameters['W_i'] + 
                h_prev @ self.parameters['U_i'] + 
                self.parameters['b_i']
            )
            self.i_cache.append(i_t)
            
            # Output gate
            o_t = self.sigmoid.forward(
                x_t @ self.parameters['W_o'] + 
                h_prev @ self.parameters['U_o'] + 
                self.parameters['b_o']
            )
            self.o_cache.append(o_t)
            
            # Cell candidate
            c_tilde_t = self.tanh.forward(
                x_t @ self.parameters['W_c'] + 
                h_prev @ self.parameters['U_c'] + 
                self.parameters['b_c']
            )
            self.c_tilde_cache.append(c_tilde_t)
            
            # Cell state
            c_t = f_t * c_prev + i_t * c_tilde_t
            self.c_cache.append(c_t)
            
            # Hidden state
            h_t = o_t * self.tanh.forward(c_t)
            self.h_cache.append(h_t)
            
            if self.return_sequences:
                outputs.append(h_t)
        
        # Préparer la sortie
        if self.return_sequences:
            self.output = np.stack(outputs, axis=1)
        else:
            self.output = self.h_cache[-1]
        
        # Gestion de return_state
        if self.return_state:
            return self.output, (self.h_cache[-1], self.c_cache[-1])
        else:
            return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass pour LSTM.
        """
        batch_size, timesteps, input_dim = self.x_cache[0].shape[0], len(self.x_cache), self.x_cache[0].shape[1]
        
        # Initialiser les gradients
        gradients = {}
        for gate in ['f', 'i', 'o', 'c']:
            gradients[f'W_{gate}'] = np.zeros_like(self.parameters[f'W_{gate}'])
            gradients[f'U_{gate}'] = np.zeros_like(self.parameters[f'U_{gate}'])
            gradients[f'b_{gate}'] = np.zeros_like(self.parameters[f'b_{gate}'])
        
        dx = np.zeros((batch_size, timesteps, input_dim))
        
        # Initialiser les gradients des états
        dh_next = np.zeros((batch_size, self.units))
        dc_next = np.zeros((batch_size, self.units))
        
        # Backward à travers le temps
        for t in reversed(range(timesteps)):
            x_t = self.x_cache[t]
            h_prev = self.h_cache[t]
            c_prev = self.c_cache[t]
            c_t = self.c_cache[t + 1]
            h_t = self.h_cache[t + 1]
            
            f_t = self.f_cache[t]
            i_t = self.i_cache[t]
            o_t = self.o_cache[t]
            c_tilde_t = self.c_tilde_cache[t]
            
            # Calculer dh_total
            if self.return_sequences:
                dh_total = dout[:, t, :] + dh_next
            else:
                if t == timesteps - 1:
                    dh_total = dout + dh_next
                else:
                    dh_total = dh_next
            
            # Gradients pour l'output gate
            tanh_c_t = np.tanh(c_t)
            do = dh_total * tanh_c_t
            do_raw = self.sigmoid.backward(do)
            
            # Gradients pour le cell state
            dc = dh_total * o_t * (1 - tanh_c_t ** 2) + dc_next
            
            # Gradients pour le forget gate
            df = dc * c_prev
            df_raw = self.sigmoid.backward(df)
            
            # Gradients pour l'input gate
            di = dc * c_tilde_t
            di_raw = self.sigmoid.backward(di)
            
            # Gradients pour le cell candidate
            dc_tilde = dc * i_t
            dc_tilde_raw = self.tanh.backward(dc_tilde)
            
            # Mettre à jour dc_next pour le pas précédent
            dc_next = dc * f_t
            
            # Calculer les gradients pour les poids
            # Forget gate
            gradients['W_f'] += x_t.T @ df_raw / batch_size
            gradients['U_f'] += h_prev.T @ df_raw / batch_size
            gradients['b_f'] += np.sum(df_raw, axis=0, keepdims=True) / batch_size
            
            # Input gate
            gradients['W_i'] += x_t.T @ di_raw / batch_size
            gradients['U_i'] += h_prev.T @ di_raw / batch_size
            gradients['b_i'] += np.sum(di_raw, axis=0, keepdims=True) / batch_size
            
            # Output gate
            gradients['W_o'] += x_t.T @ do_raw / batch_size
            gradients['U_o'] += h_prev.T @ do_raw / batch_size
            gradients['b_o'] += np.sum(do_raw, axis=0, keepdims=True) / batch_size
            
            # Cell candidate
            gradients['W_c'] += x_t.T @ dc_tilde_raw / batch_size
            gradients['U_c'] += h_prev.T @ dc_tilde_raw / batch_size
            gradients['b_c'] += np.sum(dc_tilde_raw, axis=0, keepdims=True) / batch_size
            
            # Gradient pour l'input x_t
            dx_t = (df_raw @ self.parameters['W_f'].T +
                   di_raw @ self.parameters['W_i'].T +
                   do_raw @ self.parameters['W_o'].T +
                   dc_tilde_raw @ self.parameters['W_c'].T)
            dx[:, t, :] = dx_t
            
            # Gradient pour l'état caché précédent
            dh_prev = (df_raw @ self.parameters['U_f'].T +
                      di_raw @ self.parameters['U_i'].T +
                      do_raw @ self.parameters['U_o'].T +
                      dc_tilde_raw @ self.parameters['U_c'].T)
            dh_next = dh_prev
        
        # Stocker les gradients
        for gate in ['f', 'i', 'o', 'c']:
            self.gradients[f'W_{gate}'] = gradients[f'W_{gate}']
            self.gradients[f'U_{gate}'] = gradients[f'U_{gate}']
            self.gradients[f'b_{gate}'] = gradients[f'b_{gate}']
        
        return dx
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config


class GRU(Layer):
    """
    Couche GRU (Gated Recurrent Unit).
    
    Formules GRU:
        z_t = σ(x_t @ W_z + h_{t-1} @ U_z + b_z)  # Update gate
        r_t = σ(x_t @ W_r + h_{t-1} @ U_r + b_r)  # Reset gate
        
        h̃_t = tanh(x_t @ W_h + (r_t * h_{t-1}) @ U_h + b_h)  # Candidate
        h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t               # Hidden state
    
    Args:
        units: Nombre de cellules GRU
        return_sequences: Si True, retourne toutes les sorties
        return_state: Si True, retourne aussi le dernier état
        name: Nom de la couche
    """
    
    def __init__(self,
                 units: int,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 name: Optional[str] = None):
        super().__init__(name)
        
        self.units = units
        self.return_sequences = return_sequences
        self.return_state = return_state
        
        # Fonctions d'activation
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        
        # Caches
        self.x_cache = []
        self.h_cache = []
        self.z_cache = []  # Update gate
        self.r_cache = []  # Reset gate
        self.h_tilde_cache = []  # Candidate
    
    def initialize(self, input_shape: tuple) -> tuple:
        if len(input_shape) != 3:
            raise ValueError(f"GRU expects 3D input, got {len(input_shape)}D")
        
        batch_size, timesteps, input_dim = input_shape
        
        # Initialisation des poids
        limit = np.sqrt(2.0 / (input_dim + self.units))
        
        # Poids pour les 3 transformations (z, r, h)
        for gate in ['z', 'r', 'h']:
            self.parameters[f'W_{gate}'] = np.random.randn(input_dim, self.units) * limit
            self.parameters[f'U_{gate}'] = np.random.randn(self.units, self.units) * limit
            self.parameters[f'b_{gate}'] = np.zeros((1, self.units))
            
            self.gradients[f'W_{gate}'] = np.zeros_like(self.parameters[f'W_{gate}'])
            self.gradients[f'U_{gate}'] = np.zeros_like(self.parameters[f'U_{gate}'])
            self.gradients[f'b_{gate}'] = np.zeros_like(self.parameters[f'b_{gate}'])
        
        # Shape de sortie
        if self.return_sequences:
            self.output_shape = (batch_size, timesteps, self.units)
        else:
            self.output_shape = (batch_size, self.units)
            
        return self.output_shape
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, timesteps, input_dim = x.shape
        
        # Réinitialiser les caches
        self.x_cache = []
        self.h_cache = [np.zeros((batch_size, self.units))]  # h_0
        self.z_cache = []
        self.r_cache = []
        self.h_tilde_cache = []
        
        outputs = []
        
        for t in range(timesteps):
            x_t = x[:, t, :]
            self.x_cache.append(x_t)
            
            h_prev = self.h_cache[-1]
            
            # Update gate
            z_t = self.sigmoid.forward(
                x_t @ self.parameters['W_z'] + 
                h_prev @ self.parameters['U_z'] + 
                self.parameters['b_z']
            )
            self.z_cache.append(z_t)
            
            # Reset gate
            r_t = self.sigmoid.forward(
                x_t @ self.parameters['W_r'] + 
                h_prev @ self.parameters['U_r'] + 
                self.parameters['b_r']
            )
            self.r_cache.append(r_t)
            
            # Candidate hidden state
            h_tilde_t = self.tanh.forward(
                x_t @ self.parameters['W_h'] + 
                (r_t * h_prev) @ self.parameters['U_h'] + 
                self.parameters['b_h']
            )
            self.h_tilde_cache.append(h_tilde_t)
            
            # Hidden state
            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
            self.h_cache.append(h_t)
            
            if self.return_sequences:
                outputs.append(h_t)
        
        # Préparer la sortie
        if self.return_sequences:
            self.output = np.stack(outputs, axis=1)
        else:
            self.output = self.h_cache[-1]
        
        if self.return_state:
            return self.output, self.h_cache[-1]
        else:
            return self.output
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        batch_size, timesteps, input_dim = self.x_cache[0].shape[0], len(self.x_cache), self.x_cache[0].shape[1]
        
        # Initialiser les gradients
        gradients = {}
        for gate in ['z', 'r', 'h']:
            gradients[f'W_{gate}'] = np.zeros_like(self.parameters[f'W_{gate}'])
            gradients[f'U_{gate}'] = np.zeros_like(self.parameters[f'U_{gate}'])
            gradients[f'b_{gate}'] = np.zeros_like(self.parameters[f'b_{gate}'])
        
        dx = np.zeros((batch_size, timesteps, input_dim))
        
        # Initialiser le gradient de l'état caché
        dh_next = np.zeros((batch_size, self.units))
        
        # Backward à travers le temps
        for t in reversed(range(timesteps)):
            x_t = self.x_cache[t]
            h_prev = self.h_cache[t]
            h_t = self.h_cache[t + 1]
            
            z_t = self.z_cache[t]
            r_t = self.r_cache[t]
            h_tilde_t = self.h_tilde_cache[t]
            
            # Calculer dh_total
            if self.return_sequences:
                dh_total = dout[:, t, :] + dh_next
            else:
                if t == timesteps - 1:
                    dh_total = dout + dh_next
                else:
                    dh_total = dh_next
            
            # Gradients pour le candidat h_tilde
            dh_tilde = dh_total * z_t
            dh_tilde_raw = self.tanh.backward(dh_tilde)
            
            # Gradients pour la porte update (z)
            dz = dh_total * (h_tilde_t - h_prev)
            dz_raw = self.sigmoid.backward(dz)
            
            # Gradients pour la porte reset (r)
            dr = dh_tilde_raw @ self.parameters['U_h'].T * h_prev
            dr_raw = self.sigmoid.backward(dr)
            
            # Gradients pour l'état caché précédent
            dh_prev_part1 = dh_total * (1 - z_t)
            dh_prev_part2 = dh_tilde_raw @ self.parameters['U_h'].T * r_t
            dh_prev_part3 = dz_raw @ self.parameters['U_z'].T
            dh_prev_part4 = dr_raw @ self.parameters['U_r'].T
            
            dh_prev = dh_prev_part1 + dh_prev_part2 + dh_prev_part3 + dh_prev_part4
            dh_next = dh_prev
            
            # Mettre à jour les gradients des poids
            # Update gate
            gradients['W_z'] += x_t.T @ dz_raw / batch_size
            gradients['U_z'] += h_prev.T @ dz_raw / batch_size
            gradients['b_z'] += np.sum(dz_raw, axis=0, keepdims=True) / batch_size
            
            # Reset gate
            gradients['W_r'] += x_t.T @ dr_raw / batch_size
            gradients['U_r'] += h_prev.T @ dr_raw / batch_size
            gradients['b_r'] += np.sum(dr_raw, axis=0, keepdims=True) / batch_size
            
            # Candidate
            gradients['W_h'] += x_t.T @ dh_tilde_raw / batch_size
            gradients['U_h'] += (r_t * h_prev).T @ dh_tilde_raw / batch_size
            gradients['b_h'] += np.sum(dh_tilde_raw, axis=0, keepdims=True) / batch_size
            
            # Gradient pour l'input x_t
            dx_t = (dz_raw @ self.parameters['W_z'].T +
                   dr_raw @ self.parameters['W_r'].T +
                   dh_tilde_raw @ self.parameters['W_h'].T)
            dx[:, t, :] = dx_t
        
        # Stocker les gradients
        for gate in ['z', 'r', 'h']:
            self.gradients[f'W_{gate}'] = gradients[f'W_{gate}']
            self.gradients[f'U_{gate}'] = gradients[f'U_{gate}']
            self.gradients[f'b_{gate}'] = gradients[f'b_{gate}']
        
        return dx
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'units': self.units,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state
        })
        return config