"""
Optimiseurs pour Weli.
"""

# Import de base (à compléter quand SGD sera implémenté)
# from .sgd import SGD

# Pour l'instant, exporter une classe vide pour éviter les erreurs d'import
try:
    from .sgd import SGD
except ImportError:
    # Si SGD n'est pas encore implémenté, créer une classe placeholder
    class SGD:
        """Placeholder pour SGD optimizer"""
        def __init__(self, lr=0.01):
            self.lr = lr

__all__ = ['SGD']

