# Modèles Weli

Weli propose des modèles séquentiels et une API fonctionnelle basée sur NumPy.
Les shapes fournies à `compile()` et `Input()` excluent la dimension du batch.

## Modèle séquentiel

```python
from weli.layers import Dense, ReLU
from weli.losses import MSE
from weli.models import Sequential, save_model, load_model
from weli.optimizers import Adam

model = Sequential([Dense(16), ReLU(), Dense(1)])
model.compile(input_shape=(4,), loss_fn=MSE(), optimizer=Adam())
history = model.fit(x_train, y_train, epochs=20)
predictions = model.predict(x_test)

save_model(model, "model.weli")
restored = load_model("model.weli", compile=False)
```

`Sequential.load(path)` est également disponible et vérifie que le fichier
contient bien un modèle séquentiel. Les formats complets `.weli`, `.pkl` et
`.model` utilisent la sérialisation Python; ne chargez pas de tels fichiers
provenant d'une source non fiable. Les modèles JSON sont destinés aux
configurations et poids sérialisables.

## API fonctionnelle

```python
from weli.models import Functional, Input, add, Dense

inputs = Input(shape=(8,), name="features")
branch = Dense(8, name="projection")(inputs)
outputs = add([inputs, branch], name="residual_add")
model = Functional(inputs, outputs)
```

Les graphes fonctionnels prennent en charge la sérialisation de couches
enregistrées et des opérations `add`, `concatenate` et `multiply`.
Les graphes à couches personnalisées nécessitent leur enregistrement et une
configuration désérialisable.

## Exports

- `generate_c_code()` génère une fonction d'inférence C pour les modèles
  composés de couches Dense et d'activations ReLU, Sigmoid, Tanh ou Softmax.
- ONNX, TensorFlow Lite et CoreML ne sont pas implémentés. Leurs fonctions
  d'export lèvent `NotImplementedError` et ne créent pas de faux fichier.

Exécutez les tests depuis la racine du dépôt avec `python -m pytest testes -v`.
