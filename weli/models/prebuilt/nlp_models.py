"""
Modèles NLP pré-construits pour Weli.

Implémentations légères basées sur les couches disponibles (GRU/LSTM/Dense).
Pas d'Embedding ni d'Attention ici : ces modèles supposent des features
numériques déjà préparées (ex. embeddings pré-calculés).
"""

from typing import Tuple, Optional

from weli.models.functional import Functional, Input
from weli.layers import GRU, LSTM, Dropout, Dense


def text_classifier_gru(
    input_shape: Tuple[int, int] = (100, 128),
    hidden_units: int = 128,
    num_classes: int = 2,
    dropout_rate: float = 0.1,
    name: Optional[str] = "GRUTextClassifier",
) -> Functional:
    """
    Classifieur texte simple basé sur un GRU.

    Args:
        input_shape: (timesteps, features) — features déjà vectorisées/embedding
        hidden_units: taille de l'état caché GRU
        num_classes: nombre de classes de sortie
        dropout_rate: taux de dropout après le GRU
        name: nom du modèle
    """
    inputs = Input(shape=input_shape, name="input")
    x = GRU(hidden_units, name="gru")(inputs)
    x = Dropout(dropout_rate, name="dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    return Functional(inputs=inputs, outputs=outputs, name=name)


def text_classifier_lstm(
    input_shape: Tuple[int, int] = (100, 128),
    hidden_units: int = 128,
    num_classes: int = 2,
    dropout_rate: float = 0.1,
    name: Optional[str] = "LSTMTextClassifier",
) -> Functional:
    """
    Classifieur texte simple basé sur un LSTM.

    Args:
        input_shape: (timesteps, features) — features déjà vectorisées/embedding
        hidden_units: taille de l'état caché LSTM
        num_classes: nombre de classes de sortie
        dropout_rate: taux de dropout après le LSTM
        name: nom du modèle
    """
    inputs = Input(shape=input_shape, name="input")
    x = LSTM(hidden_units, name="lstm")(inputs)
    x = Dropout(dropout_rate, name="dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)
    return Functional(inputs=inputs, outputs=outputs, name=name)


__all__ = ["text_classifier_gru", "text_classifier_lstm"]
