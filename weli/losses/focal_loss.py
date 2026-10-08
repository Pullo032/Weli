"""Focal Loss pour Weli.

Cette implémentation est exposée à travers la classe ``FocalLoss`` définie dans
``weli.losses.advanced``. Le module découpe explicitement l’API publique pour les
utilisateurs qui souhaitent importer la classe depuis son fichier dédié.
"""

from .advanced import FocalLoss

__all__ = ["FocalLoss"]

