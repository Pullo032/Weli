"""
Modèles pré-construits pour Weli.

Cette sous-API expose des architectures de référence prêtes à l'emploi pour
les tâches de vision, NLP et génération adversaire. Elles sont conçues pour
être utilisables directement et pour servir de base à des variantes
personnalisées.
"""

from .vision_models import resnet18
from .nlp_models import text_classifier_gru, text_classifier_lstm
from .gan_models import generator_mlp, discriminator_mlp, gan_mlp

PREBUILT_MODELS = {
    "vision": {
        "resnet18": resnet18,
    },
    "nlp": {
        "text_classifier_gru": text_classifier_gru,
        "text_classifier_lstm": text_classifier_lstm,
    },
    "gan": {
        "generator_mlp": generator_mlp,
        "discriminator_mlp": discriminator_mlp,
        "gan_mlp": gan_mlp,
    },
}

__all__ = [
    "PREBUILT_MODELS",
    "resnet18",
    "text_classifier_gru",
    "text_classifier_lstm",
    "generator_mlp",
    "discriminator_mlp",
    "gan_mlp",
]
