# Apprentissage auto-supervisé sur des séries temporelles

Extrait non autonome d'un projet de recherche appliquée sur l'apprentissage de représentations pour des séries temporelles multivariées. Les données réelles et les intégrations propres à Vivent, notamment l'accès aux sources de données, ont été retirées : le code présenté n'est donc pas exécutable tel quel.

## Contexte et contribution

J'ai contribué à la conception, à l'implémentation et aux tests des composants présentés ici, sous la supervision de **Julien Roulet**. Le travail sur `data_streamer/` a été mené en collaboration avec Julien Roulet et **Léo Riedtman**.

Les principales contributions visibles dans cet extrait sont :

- un pipeline de préparation des séries : alignement temporel, fenêtrage, extraction de caractéristiques et chargement par lots ;
- des transformations et normalisations adaptées aux séries temporelles ;
- une architecture modulaire pour assembler encodeurs, têtes, fonctions de coût et méthodes d'apprentissage ;
- des méthodes auto-supervisées et des composants pour des tâches en aval ;
- les scripts d'entraînement, d'évaluation et d'inférence, ainsi que des tests unitaires.

## En quelques minutes

Pour voir l'essentiel :

1. `data_streamer/window_processor.py` : préparation et fenêtrage des signaux ;
2. `model/method/methods.py` : méthodes d'apprentissage sur les représentations.

Le parcours complet est montré dans `train.py` (entraînement et validation), `validation.py` (évaluation) et `test.py` (inférence sur le split de test tenu à l'écart).

## Frontière avec les systèmes privés

Les données, configurations d'expériences réelles, paramètres finaux et poids de modèles ne sont pas inclus. Le code d'accès à la source de données n'est pas publié : les fonctions de haut niveau reçoivent une implémentation de `DataSource` par injection, au lieu de se connecter elles-mêmes à un service privé. Cette source fournit les segments des découpages `train`, `val` et `test`, ainsi que les valeurs brutes à traiter. `test.py` utilise explicitement le découpage tenu à l'écart.


## Exécution et limites

Ce dépôt est un extrait de code, pas une application autonome : les scripts réutilisent le contexte de paquet et les utilitaires du projet d'origine, dont le chargeur de configuration. Pour entraîner ou évaluer, il faut appeler les fonctions avec une configuration et une implémentation autorisée de `DataSource` fournies par l'environnement hôte. Aucun accès à une base de données ni aucune donnée réelle ne sont fournis ici.

Les dépendances Python listées dans `requirements.txt` s'installent avec `python -m pip install -r requirements.txt`. Selon la configuration matérielle, PyTorch peut nécessiter une installation adaptée au GPU.
