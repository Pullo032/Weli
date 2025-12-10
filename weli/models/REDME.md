🏗️ Structure des fichiers
1. model.py - Classe de base Model 🎯

Rôle : Interface commune que tous les modèles doivent implémenter.

Fonctionnalités principales :

    ✅ forward() / backward() - Propagation avant/arrière

    ✅ fit() / evaluate() / predict() - API d'entraînement

    ✅ save() / load() - Sérialisation

    ✅ Gestion automatique des gradients

    ✅ Historique d'entraînement

Méthodes clés à implémenter :

def forward(self, x, training=True):
    """Propagation avant à travers toutes les couches"""

def backward(self, dout):
    """Rétropropagation du gradient"""

def train_step(self, x_batch, y_batch, loss_fn, optimizer):
    """Une étape d'entraînement"""

def get_parameters(self):
    """Récupère tous les paramètres"""

def summary(self):
    """Affiche un résumé du modèle""

  2. sequential.py - Modèle Sequential 📚

Rôle : Empilement linéaire de couches (le plus courant).

Caractéristiques :

    ✅ Simple et intuitif

    ✅ Empilement dans l'ordre

    ✅ Compatible avec 80% des architectures

Exemple d'utilisation :
python

model = Sequential([
    Dense(128, activation='relu', input_dim=784),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(input_shape=(784,))
model.summary()

Méthodes spécifiques :
python

add(layer)           # Ajoute une couche à la fin
compile()           # Prépare l'entraînement
__getitem__(index)  # Accès aux couches par index

3. functional.py - API Fonctionnelle 🎨

Rôle : Création de modèles complexes avec graphes de calcul.

Pourquoi ? Pour les architectures :

    Multi-inputs / Multi-outputs

    Skip connections (ResNet)

    Branchements parallèles (Inception)

    Modèles multi-tâches

Composants :

    Input() - Nœud d'entrée

    LayerNode - Enveloppe des couches

    add(), concatenate(), multiply() - Opérations de fusion

    Functional - Classe de modèle

Exemple (ResNet-like) :
python

input_layer = Input(shape=(256,))
x = Dense(128)(input_layer)
x = ReLU()(x)
x = Dense(256)(x)
output = add([input_layer, x])  # Skip connection!

model = Functional(inputs=input_layer, outputs=output)

Algorithme clé : Tri topologique pour déterminer l'ordre d'exécution.
4. container.py - Modèles conteneurs 🧩

Rôle : Combiner plusieurs modèles.

Classes :

    ModelContainer - Regroupe plusieurs modèles

    Parallel - Exécute en parallèle avec fusion

Exemple :
python

# Deux modèles en parallèle
branch1 = Sequential([Dense(64), ReLU()])
branch2 = Sequential([Dense(64), Tanh()])

model = Parallel([branch1, branch2], merge_op='concat')

**5. registry.py - Système de Registry 📝

Rôle : Gérer la sérialisation/désérialisation.

Fonctionnalités :

    ✅ Enregistrement des classes

    ✅ Factory pour création dynamique

    ✅ Support des objets personnalisés

Utilisation :
python

registry.register_model('MyModel', MyModelClass)
registry.register_layer('MyLayer', MyLayerClass)

# Désérialisation
model = registry.deserialize_model(config)

Pourquoi nécessaire ? Pour sauvegarder/charger des modèles avec leur architecture.
**6. save_load.py - Sauvegarde/Chargement 💾

Rôle : Persistance des modèles.

Formats supportés :

    .json - Architecture seulement

    .npz - Poids seulement

    .pkl / .weli - Modèle complet

Fonctions principales :
python

save_model(model, 'model.weli')           # Sauvegarde complète
load_model('model.weli')                  # Chargement
save_model_architecture(model, 'arch.json')  # Architecture seule
load_weights(model, 'weights.npz')        # Poids seulement

Structure des fichiers :
json

{
  "model_config": {...},
  "parameters": {...},
  "metadata": {
    "weli_version": "0.1.0",
    "model_class": "Sequential"
  }
}

**7. export.py - Export vers autres formats 🚀

Rôle : Interopérabilité avec d'autres frameworks.

Formats supportés :

    ONNX (Open Neural Network Exchange)

    TensorFlow Lite (pour mobile)

    CoreML (pour iOS)

    Code C (pour l'embarqué)

    CSV (pour analyse)

Exemple :
python

export_to_onnx(model, 'model.onnx')
generate_c_code(model, 'model.c')
export_parameters_csv(model, 'parameters/')

Limitation actuelle : Certains exports nécessitent des packages externes.
**8. visualization.py - Visualisation 👁️

Rôle : Comprendre et déboguer les modèles.

Fonctionnalités :
python

plot_model_architecture(model)           # Diagramme visuel
plot_training_history(history)           # Courbes d'apprentissage
plot_layer_activations(model, input_data) # Activations
visualize_weights_distribution(model)    # Distribution des poids
generate_model_summary_table(model)      # Résumé texte
export_architecture_to_html(model)       # HTML interactif

Sorties générées :

    Images PNG (pour rapports)

    HTML interactif (pour documentation)

    Tableaux texte (pour terminal)

    Graphiques statistiques

**9. utils.py - Utilitaires 🛠️

Rôle : Fonctions auxiliaires.

Fonctions :
python

count_parameters(model)        # Compte les paramètres
print_model_summary(model)     # Résumé détaillé
get_layer_by_name(model, name) # Trouve une couche par nom
get_layer_output(model, x, idx)# Sortie d'une couche spécifique
model_to_json(model)           # Conversion JSON

🔗 Flux de travail typique
1. Création du modèle
python

from weli.models import Sequential
from weli.layers import Dense, Dropout
from weli.nn.loss import CrossEntropy
from weli.nn.optimizer import Adam

# Option A: Sequential
model = Sequential([
    Dense(128, activation='relu', input_dim=784),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Option B: Functional (pour architectures complexes)

2. Compilation
python

model.compile(
    input_shape=(784,),
    loss_fn=CrossEntropy(),
    optimizer=Adam(lr=0.001)
)

model.summary()  # Vérifier l'architecture

3. Entraînement
python

history = model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(x_val, y_val),
    verbose=1
)

4. Évaluation
python

loss, accuracy = model.evaluate(x_test, y_test)
predictions = model.predict(x_new)

5. Sauvegarde
python

model.save('my_model.weli')          # Modèle complet
model.save_weights('weights.npz')    # Poids seulement

6. Chargement
python

from weli.models import load_model
model = load_model('my_model.weli')

7. Visualisation
python

from weli.models.visualization import plot_model_architecture
plot_model_architecture(model, filename='architecture.png')

🎯 Bonnes pratiques
Pour les débutants :

    Commencez avec Sequential

    Utilisez model.summary() souvent

    Sauvegardez les poids régulièrement

    Visualisez l'entraînement avec plot_training_history()

Pour les experts :

    Utilisez l'API fonctionnelle pour les architectures complexes

    Créez des blocs réutilisables

    Exportez en ONNX pour la production

    Utilisez le registry pour les objets personnalisés

Debugging :
python

# 1. Vérifier l'architecture
model.summary()

# 2. Vérifier les shapes
print(f"Input shape: {model._input_shape}")
print(f"Output shape: {model._output_shape}")

# 3. Vérifier les paramètres
params = model.get_parameters()
print(f"Total parameters: {sum(p.size for p in params.values())}")

# 4. Visualiser les activations
plot_layer_activations(model, sample_input)

⚠️ Points d'attention
Erreurs courantes :

    Oublier compile() avant fit()

    Mauvaises shapes d'input

    Gradients non réinitialisés entre les batches

    Mode train/test non respecté

Vérifications :
python

# Avant l'entraînement
assert model.initialized, "Model not initialized"
assert hasattr(model, '_loss_fn'), "Loss function not set"
assert hasattr(model, '_optimizer'), "Optimizer not set"

# Pendant l'entraînement
assert model.training == True, "Should be in training mode"

🚀 Prochaines étapes possibles
Améliorations futures :

    Distributed training - Entraînement distribué

    Mixed precision - Float16/Float32 mixte

    Pruning/Quantization - Optimisation pour l'inférence

    AutoML - Recherche automatique d'architecture

    Explainable AI - Visualisation des décisions

Intégrations :

    TensorBoard pour le monitoring

    ONNX Runtime pour l'inférence

    Hugging Face Hub pour le partage

    MLflow pour le MLOps

📚 Ressources internes
Dépendances :

    weli.layers - Toutes les couches

    weli.nn.loss - Fonctions de perte

    weli.nn.optimizer - Optimiseurs

    weli.utils - Utilitaires généraux

Tests :
bash

# Exécuter les tests des modèles
python -m pytest tests/test_models.py -v

# Tests spécifiques
python -m pytest tests/test_models.py::TestSequential -v
python -m pytest tests/test_models.py::TestFunctional -v

🎉 Conclusion

Le module models fournit une API complète, flexible et intuitive pour créer n'importe quel type de modèle de deep learning avec Weli. Que vous soyez débutant ou expert, vous trouverez les outils nécessaires pour vos projets.