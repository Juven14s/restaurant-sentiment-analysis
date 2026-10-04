# Classification de sentiments multi-aspects

Projet de fouille de textes : classification d’avis de restaurants en français
selon trois aspects, **Prix**, **Cuisine** et **Service**.
Chaque aspect reçoit une étiquette : `Positive`, `Négative`, `Neutre` ou `NE`
(non exprimée). Dans ce projet, `Neutre` désigne une opinion mixte, avec des
éléments positifs et négatifs.

## Approches

- **CamemBERT** (`almanach/camembert-base`), entraîné avec trois têtes de
  classification indépendantes. C’est la méthode activée par défaut.
- **LLM via Ollama** : classification sans entraînement avec `gemma3:1b`.
- Un module expérimental de paraphrases via `qwen3:1.7b` est présent,
  mais l’augmentation est désactivée par défaut.

## Installation

Python 3.10 ou supérieur est requis (Python 3.12 conseillé).
Depuis la racine du dépôt :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sous Windows, activez l’environnement avec `.venv\Scripts\activate`.
Le premier lancement de CamemBERT nécessite une connexion pour télécharger
le modèle. L’entraînement peut être long sur CPU.

## Données

Placez les fichiers du cours `ftdataset_train.tsv` et `ftdataset_val.tsv` dans
`data/`. Consultez [le format des données](data/README.md).
Ces données sont exclues de Git ; leur redistribution doit être autorisée par
leur source avant de les publier.

## Exécution

Depuis la racine du dépôt :

```bash
python src/runproject.py
```

Pour une vérification rapide sur un sous-ensemble (un entraînement reste effectué) :

```bash
python src/runproject.py --n_train=32 --n_test=16 --n_runs=1
```

Pour demander un GPU :

```bash
python src/runproject.py --device=0 --n_runs=1
```

La valeur `--device=-1` sélectionne le CPU. L’implémentation actuelle utilise
le GPU CUDA par défaut lorsque `device` est positif ou nul.

Pour utiliser Ollama, définissez `METHOD = 'LLM'` dans
`src/classifier_wrapper.py`, lancez le serveur Ollama local puis téléchargez
le modèle avec `ollama pull gemma3:1b`. L’URL par défaut est
`http://localhost:11434/v1` ; elle peut être changée avec `--ollama_url`.

## Résultats et limites

Le rapport initial annonce environ **85 % d’exactitude moyenne** sur les trois
aspects. Ce résultat provient du travail initial et n’a pas été reproduit lors
de la préparation du dépôt. Le programme affiche l’exactitude par aspect et
leur moyenne (`macro_acc`). Si aucun jeu de test n’est fourni, il utilise le
jeu de validation, également employé durant l’entraînement.

La configuration décrite utilise trois époques, une longueur maximale de
256 tokens, un batch de 16, du dropout (0,3) et du label smoothing (0,1).
Le [rapport original](docs/rapport-original.txt) détaille les expériences.

## Organisation

```text
src/          Classifieurs, configuration et script d’exécution
data/        Emplacement des jeux de données locaux
docs/        Rapport initial
```

Auteur : Rassem DJIMADOUM.
