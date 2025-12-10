# Weli - Framework de Deep Learning en Python

Weli est un framework de deep learning écrit en Python, conçu pour être simple, intuitif et éducatif. Il permet de créer et d'entraîner des réseaux de neurones avec une API similaire à Keras/TensorFlow.

## 🚀 Fonctionnalités

- **Modèles flexibles** : Sequential, Functional API, et modèles conteneurs
- **Couches complètes** : Dense, Conv2D, RNN, LSTM, GRU, Dropout, BatchNorm, etc.
- **Fonctions de perte** : MSE, MAE, CrossEntropy, et bien d'autres
- **Optimiseurs** : SGD, Adam (à venir)
- **Visualisation** : Architecture, historique d'entraînement, activations
- **Export** : ONNX, TensorFlow Lite, CoreML, code C
- **Sauvegarde/Chargement** : Formats .weli, .json, .npz

## 📦 Installation

### Depuis les sources

```bash
# Cloner le repository
git clone <url-du-repo>
cd Weli

# Créer un environnement virtuel
python -m venv envweli
source envweli/bin/activate  # Sur Windows: envweli\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Installer le package
pip install -e .
```

## 🎯 Utilisation rapide

```python
from weli.models import Sequential
from weli.layers import Dense, ReLU, Softmax
from weli.losses import CrossEntropy
from weli.optimizers import SGD

# Créer un modèle
model = Sequential([
    Dense(128, activation='relu', input_dim=784),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(
    input_shape=(784,),
    loss_fn=CrossEntropy(),
    optimizer=SGD(lr=0.01)
)

# Afficher le résumé
model.summary()

# Entraîner
history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val)
)

# Évaluer
loss, accuracy = model.evaluate(x_test, y_test)

# Prédire
predictions = model.predict(x_new)

# Sauvegarder
model.save('my_model.weli')
```

## 📁 Structure du projet

```
Weli/
├── weli/                 # Package principal
│   ├── models/          # Modèles (Sequential, Functional, etc.)
│   ├── layers/          # Couches du réseau
│   ├── losses/          # Fonctions de perte
│   ├── optimizers/      # Optimiseurs
│   ├── backend/         # Backend de calcul (NumPy)
│   └── utils/           # Utilitaires
├── examples/            # Exemples d'utilisation
├── testes/              # Tests
├── requirements.txt     # Dépendances
├── setup.py            # Configuration d'installation
└── README.md           # Ce fichier
```

## 📚 Documentation

Pour plus de détails sur les modèles, consultez `weli/models/REDME.md`.

## 🧪 Tests

```bash
# Exécuter les tests
python -m pytest testes/ -v
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📝 Licence

Voir le fichier LICENSE pour plus de détails.

## 🔗 Dépendances principales

- NumPy : Calculs numériques
- Matplotlib : Visualisation
- (Autres dépendances dans requirements.txt)

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

