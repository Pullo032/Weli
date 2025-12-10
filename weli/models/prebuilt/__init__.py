"""
Modèles pré-construits pour Weli.
"""

from .vision_models import resnet18
from .nlp_models import text_classifier_gru, text_classifier_lstm
from .gan_models import generator_mlp, discriminator_mlp, gan_mlp

__all__ = [
    "resnet18",
    "text_classifier_gru",
    "text_classifier_lstm",
    "generator_mlp",
    "discriminator_mlp",
    "gan_mlp",
]
