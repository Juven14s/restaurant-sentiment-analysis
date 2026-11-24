# Analyse de sentiments multi-aspects : CamemBERT et évaluation de LLM

Projet de classification d'avis de restaurants en français selon trois aspects :
**Prix, Cuisine et Service**. Pour chaque aspect, le modèle prédit `Positive`,
`Négative`, `Neutre` ou `NE`. Dans ce corpus, `Neutre` désigne une opinion **mixte,
positive et négative**, et `NE` une opinion non exprimée.

Le projet comprend un **travail académique initial avec CamemBERT**, dont le
rapport indique environ **85 % d'exactitude moyenne en validation**, puis un
**prolongement sur la qualité des données et l'évaluation de modèles NLP/LLM**,
réalisé sur CPU sans nouvel entraînement neuronal.

Consulter le [rapport académique initial](docs/rapport-original.txt), le
[rapport de l'étude CPU, en anglais](docs/light_research_report.md), le
[guide de reproduction en français](docs/light_study.md) et
[l'état des expériences](docs/experiments.md).

## Projet initial : classification avec CamemBERT

Le modèle initial adapte **CamemBERT** (`almanach/camembert-base`) à la
classification multi-aspects : représentation du premier token, trois têtes
linéaires indépendantes, dropout de 0,3 et lissage des étiquettes de 0,1.
Le rapport décrit trois époques d'entraînement, une longueur maximale de
256 tokens et un taux d'apprentissage de 3e-5.

**Résultats de validation indiqués dans le rapport initial :**

| Mesure | Exactitude approximative |
|---|---:|
| Prix | 85 % |
| Cuisine | 86 % |
| Service | 84 % |
| **Moyenne des trois aspects** | **85 %** |

Ce résultat reste celui du projet académique. Il n'a pas été remesuré lors de
la mise au propre du dépôt. Le code et le rapport d'origine sont conservés dans
[legacy/](legacy/) et [docs/rapport-original.txt](docs/rapport-original.txt).

Les expériences suivantes utilisent un **nouveau découpage après déduplication**
et des métriques supplémentaires. Leurs scores ne sont donc pas directement
comparables aux 85 % initiaux. Aucun nouvel entraînement complet de CamemBERT
n'a été effectué avec ce protocole ; les résultats TF-IDF et Qwen ci-dessous
ne remplacent pas le résultat CamemBERT.

## Prolongement réalisé sur CPU

Cette étude examine ce que l'on peut améliorer dans les données, les métriques
et l'analyse des erreurs **sans disposer de GPU**. Qwen conserve ses poids
préentraînés ; seule la référence linéaire est ajustée sur le jeu d'entraînement.

- Validation et déduplication de 4 609 lignes sources pour conserver **4 598 avis**,
  répartis en **3 678 / 460 / 460** exemples d'entraînement, de validation et de test,
  sans texte normalisé commun entre les partitions.
- Comparaison des références majoritaire et TF-IDF avec **Qwen3-0.6B en inférence
  sur CPU**, sur un échantillon fixé de 40 avis de validation et un jeu diagnostic
  original de 80 exemples.
- Revue de **50 erreurs de validation** : opinions mixtes, portée des aspects,
  opinions absentes et ambiguïtés possibles d'annotation.
- Audit de **12 paraphrases issues uniquement de l'entraînement** ; aucune n'a
  servi à entraîner un modèle.
- Alignement strict des prédictions par identifiant, manifestes de provenance,
  **44 tests automatisés** et intégration continue avec GitHub Actions.

Les revues qualitatives et les annotations du jeu diagnostic ont été produites
par l'assistant et nécessitent une validation humaine indépendante. Les avis
du corpus initial et les paires de paraphrases ne sont pas redistribués.

## Résultats du nouveau protocole

Sur les **460 avis de validation**, TF-IDF obtient **75,94 % d'exactitude moyenne
par aspect** et **0,4449 de macro-F1**. Pour la classe d'opinion mixte, le F1 vaut
seulement 0,00 pour Prix, 0,05 pour Cuisine et 0,00 pour Service : l'exactitude
moyenne masque les difficultés sur les classes rares.

Chaque comparaison ci-dessous porte sur les mêmes exemples pour les différentes
méthodes. L'exactitude et le macro-F1 sont moyennés sur les trois aspects.

| Jeu d'évaluation | Méthode | Exactitude moyenne | Macro-F1 | Conformité du schéma |
|---|---|---:|---:|---:|
| Échantillon de validation, 40 avis | TF-IDF | 72,50 % | 0,4032 | 100 % |
| Échantillon de validation, 40 avis | Qwen, JSON strict | 0,00 % | 0,0000 | 0 % |
| Échantillon de validation, 40 avis | Qwen, analyse après retrait des balises | 47,50 % | 0,2689 | 95 % |
| Jeu diagnostic, 80 exemples | TF-IDF | 46,25 % | 0,2775 | 100 % |
| Jeu diagnostic, 80 exemples | Qwen, JSON strict | 0,00 % | 0,0000 | 0 % |
| Jeu diagnostic, 80 exemples | Qwen, analyse après retrait des balises | 62,92 % | 0,4129 | 95 % |

Qwen a entouré toutes ses réponses de balises Markdown, ce qui enfreint le
contrat imposant du JSON seul. Les scores stricts à zéro correspondent à cet
échec de format, et non à une mesure isolée de compréhension des sentiments.
Une **analyse de format séparée** retire une seule paire complète de balises
JSON des mêmes générations, sans corriger les étiquettes ni exclure les sorties
invalides. Cette règle a été choisie après inspection de la validation ; le
parseur de production reste strict.

Même après ce retrait, Qwen reste derrière TF-IDF sur les avis réels de
l'échantillon de validation. Son score supérieur sur le jeu diagnostic ne
prouve pas une supériorité générale.

Ce jeu couvre huit difficultés, dont la négation, les opinions mixtes, la
séparation des aspects et les textes ressemblant à des instructions. Il a été
rédigé par l'assistant et ne constitue pas un benchmark externe validé de façon
indépendante. Les [règles d'annotation](benchmarks/fr_aspect_challenge_v1/README.md),
les [résultats publics](reports/light_study/) et le
[rapport complet](docs/light_research_report.md) détaillent les scores par
catégorie et les limites de l'étude.

## Reproduire l'étude sur CPU

Python 3.12 est recommandé. Depuis la racine du dépôt :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[baseline]'
```

Sous Windows, activer l'environnement avec `.venv\Scripts\activate`.

Placer les deux fichiers TSV originaux dans `data/raw/`, en suivant
[data/README.md](data/README.md). Le corpus privé est nécessaire pour réajuster
les modèles de référence et reproduire les analyses de validation.

```bash
sentiment prepare \
  --input data/raw/ftdataset_train.tsv data/raw/ftdataset_val.tsv \
  --output-dir data/processed-v1 --seed 42 --split-by text

python scripts/run_light_study.py --data-dir data/processed-v1
```

Le script exécute les modèles de référence sur CPU, fixe l'échantillon de
40 avis, prépare une feuille privée de revue de 50 erreurs et évalue le jeu
public de 80 exemples. Les sorties sont placées dans `outputs/light-study-v1/`.
L'annotation qualitative demande une revue séparée : le script ne reproduit pas
automatiquement les jugements publiés.

Pour ajouter **l'inférence Qwen à poids inchangés**, installer les dépendances
ML. Sous Linux/Windows, installer d'abord la version CPU de PyTorch :

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[training]'
python scripts/run_light_study.py --data-dir data/processed-v1 --run-llm
```

Le modèle public est téléchargé au premier lancement. La recette fixe la
révision du modèle, le prompt et le décodage, avec quatre threads CPU par défaut.
Sur un Intel i7-11370H, l'inférence a pris **231 secondes pour les 40 avis** et
**417 secondes pour les 80 exemples**, hors installation et téléchargement.
Ces mesures proviennent d'une seule exécution et ne garantissent pas une durée
sur une autre machine. Le script ne lance aucun entraînement neuronal.
Le [guide en français](docs/light_study.md) décrit l'audit de paraphrases et
les commandes d'analyse ciblée.

## Contrôles sur les données et l'évaluation

La préparation valide les étiquettes, normalise les textes pour détecter les
doublons, retire les répétitions concordantes et exclut toutes les occurrences
des textes portant des annotations contradictoires. Les partitions figées
conservent les empreintes des sources, les effectifs par classe et les contrôles
de chevauchement ; elles refusent une modification des données existantes.

Le découpage par texte autorise la présence d'un même restaurant dans plusieurs
partitions. Il n'est pas stratifié et ne détecte pas les quasi-doublons
sémantiques. Une expérience sur des restaurants inconnus est possible avec
`--split-by restaurant`, dans un nouveau répertoire de sortie. Un corpus de
cours déjà étudié ne constitue pas un test externe entièrement inédit.
L'étude CPU utilise la validation pour le développement et ne réévalue pas le
test du cours pour choisir ses prompts ou ses réglages.

L'évaluation fournit l'exactitude par aspect, le macro-F1 sur quatre classes,
les effectifs par classe, les matrices de confusion, l'exactitude simultanée
sur les trois aspects et la conformité du schéma. Les identifiants manquants,
supplémentaires ou dupliqués sont refusés. Une sortie non conforme compte comme
une erreur sur les trois aspects et reste dans le dénominateur. Les manifestes
consignent les empreintes des données, les graines aléatoires, les révisions du
code et du modèle, les réglages, les versions, le matériel et les durées.
Voir le [protocole expérimental](docs/protocol.md).

## Entraînements supplémentaires facultatifs

Le dépôt contient aussi une nouvelle implémentation d'entraînement CamemBERT et
une adaptation Qwen par LoRA, avec QLoRA facultatif, sélection du meilleur point
de sauvegarde sur la validation et reprise d'entraînement. Ces procédures sont
**disponibles et testées sur des modèles miniatures** ; leurs performances sur
les modèles préentraînés complets restent à mesurer avec le nouveau protocole.
Cela ne remet pas en cause l'expérience CamemBERT du projet initial.

Pour un futur entraînement sur GPU, utiliser le
[notebook Colab](https://colab.research.google.com/github/Juven14s/restaurant-sentiment-analysis/blob/main/notebooks/colab_training.ipynb)
et son [guide en français](docs/colab.md). Il commence par un essai de 32 exemples,
conserve les sauvegardes dans Drive et laisse l'évaluation finale sur test
désactivée jusqu'à son activation explicite. Colab est facultatif et indépendant
de l'étude CPU présentée ci-dessus.

## Tests et organisation du dépôt

```bash
python -m unittest discover -s tests -v
```

La suite comprend 44 tests. Les contrôles de base n'ont besoin ni de GPU ni de
téléchargement de modèle ; les tests ML facultatifs sont ignorés si leurs
dépendances sont absentes. Des modèles locaux miniatures à poids aléatoires
permettent de tester de vrais entraînements CPU, le rechargement des sauvegardes
et la reprise LoRA. Ils vérifient l'intégration logicielle, pas la qualité des
modèles préentraînés ni la compatibilité CUDA.

```text
src/restaurant_sentiment/  Données, modèles, prompts, analyses, entraînement et CLI
scripts/                  Étude CPU et audit de paraphrases issues de l'entraînement
benchmarks/               Jeu diagnostic public, étiquettes et justifications
reports/                  Scores publiés et diagnostics agrégés
tests/                    Contrôles du protocole et intégrations hors ligne
docs/                     Rapports, protocole et guides de reproduction
notebooks/                Procédure facultative d'entraînement Colab
legacy/                   Code académique initial archivé
data/                     Sources privées et partitions locales, ignorées par Git
outputs/                  Prédictions et sauvegardes locales, ignorées par Git
```

Les gains d'augmentation de données, DPO et l'exécution sur cluster ou plusieurs
GPU restent des pistes futures. Auteur : Rassem DJIMADOUM.
