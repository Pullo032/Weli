"""
Implémentations d'attention pour Weli (NumPy uniquement).

Inclus :
- MultiHeadAttention : attention multi-têtes générique (Q/K/V).
- SelfAttention : alias utilisant la même entrée pour Q/K/V.

Notes :
- Fonctionne sur des tenseurs de forme (batch, seq_len, d_model).
- Pas d'embedding ni de masquage avancé pour l'instant.
- Compatible avec la classe Layer (forward/backward + gradients).
"""

import numpy as np
from typing import Optional, Tuple
from .base import Layer


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


class MultiHeadAttention(Layer):
    """
    Attention multi-têtes (Q, K, V) avec NumPy.

    Args:
        d_model: dimension du modèle (features d'entrée)
        num_heads: nombre de têtes (d_model doit être divisible par num_heads)
        dropout: ignoré pour l'instant (placeholder)
        name: nom de la couche
    """

    def __init__(self, d_model: int, num_heads: int = 8,
                 dropout: float = 0.0, name: Optional[str] = None):
        super().__init__(name)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        self.dropout = dropout  # non utilisé

        # Poids
        self.parameters = {
            "W_q": None,
            "b_q": None,
            "W_k": None,
            "b_k": None,
            "W_v": None,
            "b_v": None,
            "W_o": None,
            "b_o": None,
        }
        self.gradients = {k: None for k in self.parameters}

        # Cache pour backward
        self._cache = {}

    def initialize(self, input_shape: Tuple[int, int]) -> Tuple[int, int]:
        """
        input_shape: (seq_len, d_model)
        """
        seq_len, dim = input_shape
        if dim != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {dim}")

        # Init Xavier/Glorot simple
        scale = np.sqrt(2.0 / (self.d_model + self.d_model))
        def init_w():
            return np.random.randn(self.d_model, self.d_model) * scale

        self.parameters["W_q"] = init_w()
        self.parameters["W_k"] = init_w()
        self.parameters["W_v"] = init_w()
        self.parameters["W_o"] = init_w()
        self.parameters["b_q"] = np.zeros((self.d_model,))
        self.parameters["b_k"] = np.zeros((self.d_model,))
        self.parameters["b_v"] = np.zeros((self.d_model,))
        self.parameters["b_o"] = np.zeros((self.d_model,))

        self.output_shape = input_shape
        return self.output_shape

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, seq, d_model) -> (batch, num_heads, seq, depth)
        """
        b, t, _ = x.shape
        x = x.reshape(b, t, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, num_heads, seq, depth) -> (batch, seq, d_model)
        """
        b, h, t, d = x.shape
        x = x.transpose(0, 2, 1, 3).reshape(b, t, h * d)
        return x

    def forward(self, query: np.ndarray, key: Optional[np.ndarray] = None,
                value: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Args:
            query: (batch, seq_q, d_model)
            key:   (batch, seq_k, d_model) ou None -> query
            value: (batch, seq_v, d_model) ou None -> key
            mask:  (batch, 1, 1, seq_k) ou (batch, 1, seq_q, seq_k) optionnel
        Returns:
            out: (batch, seq_q, d_model)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        W_q, b_q = self.parameters["W_q"], self.parameters["b_q"]
        W_k, b_k = self.parameters["W_k"], self.parameters["b_k"]
        W_v, b_v = self.parameters["W_v"], self.parameters["b_v"]
        W_o, b_o = self.parameters["W_o"], self.parameters["b_o"]

        # Linéaires
        Q = np.matmul(query, W_q) + b_q   # (b, tq, d_model)
        K = np.matmul(key,   W_k) + b_k   # (b, tk, d_model)
        V = np.matmul(value, W_v) + b_v   # (b, tv, d_model)

        # Split heads
        Qh = self._split_heads(Q)  # (b, h, tq, depth)
        Kh = self._split_heads(K)  # (b, h, tk, depth)
        Vh = self._split_heads(V)  # (b, h, tv, depth)

        # Scores
        scores = np.matmul(Qh, Kh.transpose(0, 1, 3, 2)) / np.sqrt(self.depth)  # (b,h,tq,tk)
        if mask is not None:
            scores = scores + (mask * -1e9)
        attn = _softmax(scores, axis=-1)  # (b,h,tq,tk)

        # Contexte
        context = np.matmul(attn, Vh)  # (b,h,tq,depth)
        context = self._combine_heads(context)  # (b,tq,d_model)

        out = np.matmul(context, W_o) + b_o  # (b,tq,d_model)

        # Cache pour backward
        self._cache = {
            "query": query, "key": key, "value": value,
            "Q": Q, "K": K, "V": V,
            "Qh": Qh, "Kh": Kh, "Vh": Vh,
            "attn": attn, "scores": scores, "mask": mask,
            "context": context
        }
        return out

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward passe.
        Args:
            dout: (batch, seq_q, d_model)
        Returns:
            dquery, dkey, dvalue
        """
        cache = self._cache
        Qh, Kh, Vh = cache["Qh"], cache["Kh"], cache["Vh"]
        attn, context = cache["attn"], cache["context"]
        Q, K, V = cache["Q"], cache["K"], cache["V"]
        query, key, value = cache["query"], cache["key"], cache["value"]

        W_q, W_k, W_v, W_o = (
            self.parameters["W_q"], self.parameters["W_k"],
            self.parameters["W_v"], self.parameters["W_o"]
        )

        b, h, tq, depth = Qh.shape

        # Grad sur W_o et b_o
        dW_o = np.matmul(context.reshape(-1, self.d_model).T, dout.reshape(-1, self.d_model))
        db_o = np.sum(dout, axis=(0, 1))

        dcontext = np.matmul(dout, W_o.T)  # (b,tq,d_model)
        dcontext_h = dcontext.reshape(b, tq, h, depth).transpose(0, 2, 1, 3)  # (b,h,tq,depth)

        # dAttn et dVh
        dattn = np.matmul(dcontext_h, Vh.transpose(0, 1, 3, 2))  # (b,h,tq,tk)
        dVh = np.matmul(attn.transpose(0, 1, 3, 2), dcontext_h).transpose(0, 1, 3, 2)  # (b,h,tv,depth)

        # Softmax grad: attn * (dattn - sum(dattn*attn))
        sum_dattn = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - sum_dattn)  # (b,h,tq,tk)

        dscores /= np.sqrt(self.depth)

        # Grad Qh et Kh
        dQh = np.matmul(dscores, Kh)  # (b,h,tq,depth)
        dKh = np.matmul(dscores.transpose(0, 1, 3, 2), Qh)  # (b,h,tk,depth)

        # Grad Vh déjà calculé

        # Combine heads back
        def combine(x):
            return x.transpose(0, 2, 1, 3).reshape(query.shape[0], -1, self.d_model)

        dQ = combine(dQh)
        dK = combine(dKh)
        dV = combine(dVh)

        # Grad sur poids Q/K/V et inputs
        dW_q = np.matmul(query.reshape(-1, self.d_model).T, dQ.reshape(-1, self.d_model))
        db_q = np.sum(dQ, axis=(0, 1))
        dW_k = np.matmul(key.reshape(-1, self.d_model).T, dK.reshape(-1, self.d_model))
        db_k = np.sum(dK, axis=(0, 1))
        dW_v = np.matmul(value.reshape(-1, self.d_model).T, dV.reshape(-1, self.d_model))
        db_v = np.sum(dV, axis=(0, 1))

        dquery = np.matmul(dQ, W_q.T)
        dkey = np.matmul(dK, W_k.T)
        dvalue = np.matmul(dV, W_v.T)

        # Stocker gradients
        self.gradients["W_q"] = dW_q
        self.gradients["b_q"] = db_q
        self.gradients["W_k"] = dW_k
        self.gradients["b_k"] = db_k
        self.gradients["W_v"] = dW_v
        self.gradients["b_v"] = db_v
        self.gradients["W_o"] = dW_o
        self.gradients["b_o"] = db_o

        return dquery, dkey, dvalue


class SelfAttention(MultiHeadAttention):
    """
    Self-attention (Q=K=V) en héritant de MultiHeadAttention.
    """

    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return super().forward(x, x, x, mask)
