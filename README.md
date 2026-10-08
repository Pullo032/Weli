# Weli

**Weli** est un framework de deep learning en Python, construit autour de NumPy. Il propose une API pédagogique pour assembler, entraîner, évaluer et sauvegarder des réseaux de neurones.

> Weli est un projet éducatif et léger. Il ne remplace pas les frameworks de production comme PyTorch ou TensorFlow.

## Sommaire

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Principales API](#principales-api)
- [Sauvegarder et recharger un modèle](#sauvegarder-et-recharger-un-modèle)
- [Exemples et documentation](#exemples-et-documentation)
- [Dépannage](README/depannage.md)
- [Structure du dépôt](#structure-du-dépôt)
- [Limites connues](#limites-connues)
- [Tests](#tests)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Fonctionnalités

- Modèles `Sequential`, modèles fonctionnels et conteneurs pour composer plusieurs modèles.
- Couches denses, convolutives, récurrentes et d'attention, ainsi que des activations, du pooling, du dropout et de la normalisation.
- Fonctions de perte pour la régression, la classification et l'apprentissage par similarité.
- Optimiseurs SGD, RMSprop et Adam.
- Entraînement par mini-batch, validation, évaluation et prédiction avec des tableaux NumPy.
- Modèles de référence préconstruits pour la vision, le NLP et les GAN.
- Outils de visualisation et de sauvegarde.
- Génération de code C pour un sous-ensemble de modèles séquentiels.

## Installation

Weli requiert Python 3.7 ou une version ultérieure. Après la publication de Weli sur PyPI, installez-le avec ses dépendances d'exécution :

```bash
python -m pip install weli-ml
```

Pour travailler sur le code source, clonez le dépôt et installez le package en mode éditable :

```bash
git clone https://github.com/Pullo032/Weli.git
cd Weli
python -m venv .venv
```

Activez l'environnement virtuel, puis installez Weli :

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# Installation du package et des dépendances de test
python -m pip install -e ".[test]"
```

L'installation en mode éditable convient au développement depuis un clone du dépôt. Pour installer depuis les sources sans le mode éditable, utilisez `python -m pip install .`.

## Démarrage rapide

Cet exemple crée et entraîne un petit modèle de régression, l'évalue, puis produit des prédictions. Les dimensions de `input_shape` excluent la dimension du batch.

```python
import numpy as np

from weli.layers import Dense
from weli.losses import MSE
from weli.models import Sequential, load_model
from weli.optimizers import Adam

np.random.seed(42)
x = np.random.randn(256, 4)
y = (2 * x[:, :1] - 0.5 * x[:, 1:2] + 0.1)

x_train, y_train = x[:200], y[:200]
x_val, y_val = x[200:], y[200:]

model = Sequential([
    Dense(16, activation="relu"),
    Dense(1),
])
model.compile(
    input_shape=(4,),
    loss_fn=MSE(),
    optimizer=Adam(),
)

model.summary()
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val, y_val),
)

loss, accuracy = model.evaluate(x_val, y_val)
predictions = model.predict(x_val[:5])

model.save("regression.weli")
restored_model = load_model("regression.weli", compile=False)
restored_predictions = restored_model.predict(x_val[:5])
```

`history` est un dictionnaire contenant les métriques d'entraînement, notamment `train_loss`, `train_acc`, `val_loss` et `val_acc`. Pour une tâche de régression, la métrique d'accuracy n'est pas significative.

## Principales API

Les composants peuvent être importés depuis leur module ou, pour les symboles principaux, directement depuis `weli`.

### Modèles

```python
from weli.models import Sequential, Functional, ModelContainer, Parallel
```

- `Sequential` empile des couches dans un ordre linéaire.
- `Functional` permet de construire des graphes avec des branches et des opérations de fusion.
- `ModelContainer` compose des modèles en séquence.
- `Parallel` exécute et fusionne des branches parallèles.

Pour entraîner un modèle séquentiel, configurez-le avec `compile(input_shape=..., loss_fn=..., optimizer=...)`, puis appelez `fit(...)`. `evaluate(...)` renvoie la perte et l'accuracy calculées par le modèle ; `predict(...)` renvoie les sorties du réseau.

### Couches

```python
from weli.layers import (
    Dense, Conv2D, MaxPool2D, Flatten,
    ReLU, Sigmoid, Tanh, Softmax,
    Dropout, BatchNorm1D, BatchNorm2D,
    SimpleRNN, LSTM, GRU,
    MultiHeadAttention, SelfAttention,
)
```

La disponibilité d'une couche ne signifie pas que toutes les combinaisons d'architectures sont compatibles. Consultez les exemples et les guides avant d'assembler des architectures avancées.

### Fonctions de perte

```python
from weli.losses import MSE, MAE, HuberLoss, CrossEntropy, BinaryCrossEntropy
```

Le package inclut notamment `MSE`, `MAE`, `HuberLoss`, `MSLE`, `CrossEntropy`, `BinaryCrossEntropy`, `CategoricalCrossEntropy`, `SparseCategoricalCrossEntropy`, `HingeLoss`, `SquaredHingeLoss`, `KLDivergence`, `PoissonLoss`, `CosineSimilarityLoss`, `LogCoshLoss`, `DiceLoss`, `FocalLoss`, `TripletLoss`, `ContrastiveLoss` et `WassersteinLoss`.

`CrossEntropy` accepte des indices de classe entiers ou des labels one-hot pour une sortie multi-classes. Respectez les formes et les conventions de labels attendues par la perte choisie.

### Optimiseurs

```python
from weli.optimizers import SGD, RMSprop, Adam

optimizer = SGD(lr=0.01, momentum=0.9)
```

Les optimiseurs sont fournis au modèle via `compile()` ou `fit()`.

### Modèles préconstruits

```python
from weli.models import (
    resnet18,
    text_classifier_gru,
    text_classifier_lstm,
    generator_mlp,
    discriminator_mlp,
    gan_mlp,
)
```

Ces architectures sont des modèles de référence destinés à servir de point de départ à l'apprentissage et à l'expérimentation.

## Sauvegarder et recharger un modèle

Un modèle séquentiel peut être sauvegardé et rechargé comme suit :

```python
from weli.models import load_model, save_model

save_model(model, "model.weli")
loaded_model = load_model("model.weli", compile=False)
```

La méthode `model.save(path)` est également disponible. La sauvegarde complète utilise par défaut le format Weli si aucune extension prise en charge n'est fournie ; la sauvegarde des seuls poids prend par défaut le format NPZ. Les formats et options disponibles sont documentés dans le guide des [modèles](weli/models/README.md).

**Sécurité :** certains formats de sauvegarde complets reposent sur la sérialisation Python. Ne chargez jamais un fichier de modèle provenant d'une source non fiable.

## Exemples et documentation

- [Documentation détaillée](README/README.md) : guides d'installation, modèles, couches, pertes, optimiseurs, entraînement, exemples et dépannage.
- [Référence complète de l'API](README/api.md) : symboles exportés, conventions des données et outils disponibles.
- [Guide des modèles](weli/models/README.md) : API séquentielle et fonctionnelle, sérialisation et exports.
- [Exemple de régression](examples/regression.py) et [guide des exemples](examples/README.md).
- [Application de documentation](Frontend/weli-frontend/) : site de documentation Weli.

Les guides Markdown sont également inclus dans l'archive source. La
[documentation du dépôt sur GitHub](https://github.com/Pullo032/Weli/tree/main/README)
est consultable sans installer le paquet.

## Structure du dépôt

```text
Weli/
├── weli/
│   ├── backend/       # Backend de calcul
│   ├── layers/        # Couches et opérations
│   ├── losses/        # Fonctions de perte
│   ├── models/        # Modèles, conteneurs, sauvegarde et export
│   ├── optimizers/    # Optimiseurs
│   └── utils/         # Utilitaires
├── examples/          # Exemples exécutables
├── testes/            # Tests automatisés
├── Frontend/          # Site React de documentation
├── README/            # Guides de documentation
├── requirements.txt   # Dépendances d'exécution
├── setup.py           # Métadonnées et configuration du package
└── pyproject.toml     # Configuration du système de build
```

## Limites connues

- Weli vise l'apprentissage et l'expérimentation ; ses performances et sa couverture ne sont pas celles d'un framework de deep learning de production.
- `generate_c_code()` ne prend actuellement en charge que les modèles `Sequential` constitués de couches `Dense` et d'activations ReLU, Sigmoid, Tanh ou Softmax.
- Les exports ONNX, TensorFlow Lite et CoreML ne sont pas implémentés ; leurs fonctions lèvent `NotImplementedError`.
- Les modèles préconstruits sont des architectures de référence, et non un catalogue complet de modèles pré-entraînés.

## Tests

Depuis la racine du dépôt, installez les dépendances de test (`python -m pip install -e ".[test]"`), puis lancez :

```bash
python -m pytest testes -v
```

## Contribuer

Les contributions sont les bienvenues. Pour proposer une évolution ou signaler un problème, ouvrez une [issue](https://github.com/Pullo032/Weli/issues) ou une pull request sur le dépôt.

Avant de soumettre une contribution, exécutez les tests concernés et décrivez les changements ainsi que les éventuelles limites.

## Licence

Weli est distribué sous licence MIT. Consultez le fichier [LICENSE](LICENSE) pour les conditions complètes.
