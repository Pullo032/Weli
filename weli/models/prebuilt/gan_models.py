"""
Modèles GAN pré-construits pour Weli.

Implémentation d'un GAN MLP simple pour images aplaties (ex: 28x28=784).
Sans couches de convolution transpose (non disponibles ici), le générateur
et le discriminateur sont des MLP. Les images sont donc manipulées en vecteur
flatten (l'utilisateur peut reshaper en sortie/entrée si besoin).
"""

from typing import Tuple, Optional

from weli.models.functional import Functional, Input
from weli.layers import Dense, Dropout
from weli.layers.activation import LeakyReLU, Tanh, Sigmoid


def generator_mlp(
    noise_dim: int = 100,
    img_dim: int = 784,
    hidden_units: Tuple[int, int] = (256, 128),
    name: Optional[str] = "GAN_Generator"
) -> Functional:
    """
    Générateur MLP pour GAN produisant un vecteur image aplati.

    Args:
        noise_dim: dimension du vecteur bruit (z)
        img_dim: dimension de la sortie (flattened image)
        hidden_units: tailles des couches cachées
        name: nom du modèle
    """
    inputs = Input(shape=(noise_dim,), name="z")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = Dense(units, name=f"g_dense_{i}")(x)
        x = LeakyReLU(alpha=0.2)(x)
    x = Dense(img_dim, name="g_out")(x)
    outputs = Tanh(name="g_tanh")(x)  # sortie dans [-1, 1]
    return Functional(inputs=inputs, outputs=outputs, name=name)


def discriminator_mlp(
    img_dim: int = 784,
    hidden_units: Tuple[int, int] = (128, 64),
    dropout_rate: float = 0.3,
    name: Optional[str] = "GAN_Discriminator"
) -> Functional:
    """
    Discriminateur MLP pour GAN prenant un vecteur image aplati.

    Args:
        img_dim: dimension de l'entrée (flattened image)
        hidden_units: tailles des couches cachées
        dropout_rate: taux de dropout
        name: nom du modèle
    """
    inputs = Input(shape=(img_dim,), name="img_flat")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = Dense(units, name=f"d_dense_{i}")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(dropout_rate, name=f"d_dropout_{i}")(x)
    x = Dense(1, name="d_out")(x)
    outputs = Sigmoid(name="d_sigmoid")(x)  # probabilité réel/faux
    return Functional(inputs=inputs, outputs=outputs, name=name)


def gan_mlp(
    noise_dim: int = 100,
    img_dim: int = 784,
    gen_hidden: Tuple[int, int] = (256, 128),
    disc_hidden: Tuple[int, int] = (128, 64),
    disc_dropout: float = 0.3,
) -> Tuple[Functional, Functional]:
    """
    Crée un couple (generator, discriminator) pour un GAN MLP.

    Args:
        noise_dim: dimension du bruit
        img_dim: dimension de l'image aplatie
        gen_hidden: tailles des couches cachées du générateur
        disc_hidden: tailles des couches cachées du discriminateur
        disc_dropout: dropout du discriminateur

    Returns:
        (generator, discriminator)
    """
    gen = generator_mlp(noise_dim=noise_dim, img_dim=img_dim,
                        hidden_units=gen_hidden, name="Generator")
    disc = discriminator_mlp(img_dim=img_dim, hidden_units=disc_hidden,
                             dropout_rate=disc_dropout, name="Discriminator")
    return gen, disc


__all__ = ["generator_mlp", "discriminator_mlp", "gan_mlp"]
