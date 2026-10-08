"""
Modèles de vision pré-construits pour Weli.

Implémentation minimale de ResNet18 en utilisant l'API fonctionnelle.
Cette version reste simple (Conv-BN-ReLU + skip connections) et utilise
Flatten + Dense au lieu d'un global average pooling (non implémenté ici).
"""

from typing import Tuple, Optional

from weli.models.functional import (
    Functional, Input, add,
    Conv2D, BatchNorm2D, ReLU, MaxPool2D, Flatten, Dense
)


def _resnet_block(x, filters: int, stride: int, name: str, use_conv_shortcut: bool):
    """
    Bloc résiduel basique (2x conv 3x3) avec option de downsampling.
    """
    shortcut = x

    # Branche principale
    x = Conv2D(filters, kernel_size=3, strides=stride, padding='same', name=f"{name}_conv1")(x)
    x = BatchNorm2D(name=f"{name}_bn1")(x)
    x = ReLU()(x)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', name=f"{name}_conv2")(x)
    x = BatchNorm2D(name=f"{name}_bn2")(x)

    # Branche shortcut si dimensions changent
    if use_conv_shortcut or stride != 1:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same', name=f"{name}_conv_short")(shortcut)
        shortcut = BatchNorm2D(name=f"{name}_bn_short")(shortcut)

    # Fusion + activation
    x = add([x, shortcut], name=f"{name}_add")
    x = ReLU()(x)
    return x


def resnet18(input_shape: Tuple[int, int, int] = (224, 224, 3),
             num_classes: int = 1000,
             name: Optional[str] = "ResNet18") -> Functional:
    """
    Construit un modèle ResNet18 (version simplifiée).

    Args:
        input_shape: Shape de l'image d'entrée (H, W, C)
        num_classes: Nombre de classes de sortie
        name: Nom du modèle

    Returns:
        Functional: modèle ResNet18 prêt à compiler/entraîner.
    """
    inputs = Input(shape=input_shape, name="input")

    # Stem
    x = Conv2D(64, kernel_size=7, strides=2, padding='same', name="conv1")(inputs)
    x = BatchNorm2D(name="bn1")(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same', name="pool1")(x)

    # Blocks
    x = _resnet_block(x, 64, stride=1, name="block1_1", use_conv_shortcut=False)
    x = _resnet_block(x, 64, stride=1, name="block1_2", use_conv_shortcut=False)

    x = _resnet_block(x, 128, stride=2, name="block2_1", use_conv_shortcut=True)
    x = _resnet_block(x, 128, stride=1, name="block2_2", use_conv_shortcut=False)

    x = _resnet_block(x, 256, stride=2, name="block3_1", use_conv_shortcut=True)
    x = _resnet_block(x, 256, stride=1, name="block3_2", use_conv_shortcut=False)

    x = _resnet_block(x, 512, stride=2, name="block4_1", use_conv_shortcut=True)
    x = _resnet_block(x, 512, stride=1, name="block4_2", use_conv_shortcut=False)

    # Tête de classification (pas de global average pooling dispo, on Flatten)
    x = Flatten(name="flatten")(x)
    outputs = Dense(num_classes, activation="softmax", name="fc")(x)

    model = Functional(inputs=inputs, outputs=outputs, name=name)
    return model


__all__ = ["resnet18"]

