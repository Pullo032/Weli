# Exemple de régression

Ce programme autonome montre le flux de travail de base avec Weli : créer
des données NumPy, construire un modèle séquentiel, choisir une perte et un
optimiseur, puis entraîner le modèle et consulter quelques prédictions.
Les données sont synthétiques ; aucun fichier de données externe n'est
nécessaire.

## Exécution

```bash
python -m examples.regression
```

Lancez la commande depuis la racine du dépôt. Le script affiche la perte
d'entraînement finale et les prédictions de trois exemples.

## Comprendre l'exemple

La matrice `x_train` contient deux caractéristiques par ligne. La cible est
calculée par la formule `y = 2 * x[:, 0] - 0.5 * x[:, 1] + 1` et a une
colonne, soit une sortie numérique par exemple. Le modèle contient une couche
cachée de huit unités avec ReLU et une couche de sortie dense.

`input_shape=(2,)` décrit les caractéristiques d'un exemple, sans le nombre
d'exemples du lot. `MSE()` mesure l'erreur de régression et `Adam(lr=0.01)`
met à jour les paramètres à chaque mini-lot.

Pour modifier l'exemple, gardez `input_shape`, la dernière couche et la forme
des cibles cohérentes avec votre problème. Le
[guide de régression](../README/exemples.md) présente le même flux de travail
avec davantage d'explications ; le [guide d'entraînement](../README/entrainement.md)
détaille validation et métriques.