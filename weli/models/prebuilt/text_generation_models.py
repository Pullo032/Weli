"""
Generateurs conditionnes par du texte, construits avec les couches natives de Weli.

Ces architectures pedagogiques prennent des sequences d'embeddings numeriques
prepares par l'utilisateur et produisent des pixels aplatis dans [-1, 1].
"""

from typing import Optional, Tuple

from weli.layers import GRU, Tanh
from weli.models.functional import Dense, Functional, Input, LayerNode


def _validate_positive_shape(shape: Tuple[int, ...], dimensions: int, label: str) -> None:
    if (
        not isinstance(shape, tuple)
        or len(shape) != dimensions
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape)
    ):
        raise ValueError(
            f"{label} must be a tuple of {dimensions} positive integers."
        )


def _text_generator(
    text_shape: Tuple[int, int],
    output_size: int,
    hidden_units: int,
    name: Optional[str],
) -> Functional:
    _validate_positive_shape(text_shape, 2, "text_shape")
    if not isinstance(hidden_units, int) or isinstance(hidden_units, bool) or hidden_units <= 0:
        raise ValueError("hidden_units must be a positive integer.")
    if not isinstance(output_size, int) or isinstance(output_size, bool) or output_size <= 0:
        raise ValueError("output_size must be a positive integer.")

    text = Input(shape=text_shape, name="text_embeddings")
    context = LayerNode(GRU(hidden_units, name="text_encoder"))(text)
    hidden = Dense(hidden_units, activation="relu", name="generator_hidden")(context)
    pixels = Dense(output_size, name="generated_pixels")(hidden)
    outputs = LayerNode(Tanh(name="pixel_range"))(pixels)
    return Functional(inputs=text, outputs=outputs, name=name)


def text_to_image_mlp(
    text_shape: Tuple[int, int] = (16, 64),
    image_shape: Tuple[int, int, int] = (32, 32, 3),
    hidden_units: int = 128,
    name: Optional[str] = "TextToImageMLP",
) -> Functional:
    """
    Construit un petit generateur d'image conditionne par des embeddings textuels.

    Args:
        text_shape: (longueur, dimension_embedding), apres vectorisation du texte
        image_shape: (hauteur, largeur, canaux) de l'image cible
        hidden_units: taille de l'encodeur GRU et de la couche cachee
        name: nom du modele

    Returns:
        Modele produisant des pixels aplatis, a remodeler en image_shape.
        Les valeurs de sortie sont dans [-1, 1].
    """
    _validate_positive_shape(image_shape, 3, "image_shape")
    image_size = image_shape[0] * image_shape[1] * image_shape[2]
    return _text_generator(text_shape, image_size, hidden_units, name)


def text_to_video_mlp(
    text_shape: Tuple[int, int] = (16, 64),
    num_frames: int = 8,
    frame_shape: Tuple[int, int, int] = (32, 32, 3),
    hidden_units: int = 128,
    name: Optional[str] = "TextToVideoMLP",
) -> Functional:
    """
    Construit un petit generateur video conditionne par des embeddings textuels.

    Args:
        text_shape: (longueur, dimension_embedding), apres vectorisation du texte
        num_frames: nombre d'images de la video
        frame_shape: (hauteur, largeur, canaux) de chaque image
        hidden_units: taille de l'encodeur GRU et de la couche cachee
        name: nom du modele

    Returns:
        Modele produisant une sequence d'images aplatie, a remodeler en
        (num_frames,) + frame_shape. Les valeurs de sortie sont dans [-1, 1].
    """
    _validate_positive_shape(frame_shape, 3, "frame_shape")
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer.")
    frame_size = frame_shape[0] * frame_shape[1] * frame_shape[2]
    return _text_generator(
        text_shape,
        num_frames * frame_size,
        hidden_units,
        name,
    )


__all__ = ["text_to_image_mlp", "text_to_video_mlp"]
