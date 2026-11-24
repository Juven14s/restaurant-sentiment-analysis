# Étude CPU sans entraînement neuronal

Cette voie prépare des preuves expérimentales sur les données, l'évaluation et
les erreurs sans relancer CamemBERT ni entraîner un LLM. TF-IDF est réajusté sur
le train uniquement, ce qui prend peu de temps. Qwen est utilisé en inférence,
avec ses poids préentraînés inchangés.

## Reproduire les analyses légères

Avec les splits du protocole v1 déjà préparés :

```bash
python -m pip install -e '.[baseline]'
python scripts/run_light_study.py --data-dir data/processed-v1
```

Le script produit dans `outputs/light-study-v1/` :

- les résultats TF-IDF et majoritaire sur la validation ;
- un échantillon de 40 avis choisi par hash sans regarder les étiquettes ;
- une feuille de revue de 50 erreurs, réparties par motifs d'aspects erronés ;
- les scores du benchmark public de 80 avis originaux, par catégorie.

Pour ajouter l'inférence CPU Qwen3-0.6B :

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[training]'
python scripts/run_light_study.py --data-dir data/processed-v1 --run-llm
```

Le modèle public est téléchargé au premier lancement. La recette fixe sa
révision, le prompt existant, le décodage glouton, la longueur d'entrée 512 et
96 tokens de sortie. Le script ne contient aucun appel d'entraînement neuronal.
`--cpu-threads 4` limite le nombre de threads PyTorch ; ajustez selon votre CPU.
La première commande ci-dessus installe PyTorch CPU sous Linux/Windows et évite
de télécharger les dépendances CUDA pour un ordinateur sans GPU.

Pour relancer une analyse précise :

```bash
sentiment baseline --data-dir data/processed-v1 --method tfidf --split validation \
  --output-dir outputs/validation/tfidf
sentiment diagnose --references data/processed-v1/validation.jsonl \
  --predictions outputs/validation/tfidf/predictions.jsonl --output-dir outputs/validation/analysis
```

Les feuilles CSV contiennent les avis privés et restent ignorées par Git.
Une feuille déjà annotée n'est pas écrasée. Les statistiques lexicales sont des
indices de surface, pas des diagnostics sémantiques ou causaux automatiques.
La revue qualitative publiée a été réalisée par l'assistant sur 50 erreurs ;
elle demande une confirmation humaine, notamment pour les annotations discutables.
Les étiquettes originales restent inchangées.

## Contrat JSON et analyse de format

Le parseur de production exige du JSON seul, avec les trois clés et les quatre
étiquettes exactes. Une réponse entourée de balises Markdown échoue ce contrat.
L'étude conserve le score strict et présente une **ablation séparée** sur les
mêmes générations : retirer une seule paire complète de balises Markdown JSON,
sans corriger les étiquettes, inventer des champs ou éliminer des exemples.

```bash
sentiment format-ablation --references data/validation-probe-v1/probe.jsonl \
  --predictions outputs/validation-probe/qwen/predictions.jsonl \
  --output-dir outputs/validation-probe/qwen/ablation
```

Cette normalisation a été décidée après observation des sorties de validation.
Elle n'est pas présentée comme une amélioration préenregistrée sur un test final.
Le test du cours n'est pas utilisé pour choisir des réglages dans cette étude.

## Petit audit de paraphrases

Un second script génère douze paraphrases issues du train uniquement, avec une
répartition de trois sources par classe Cuisine :

```bash
python scripts/audit_paraphrases.py --train-path data/processed-v1/train.jsonl \
  --output-dir outputs/paraphrase-audit-new
```

Le résultat reste privé dans `private_pairs.json` et demande une revue. L'audit
réalisé a trouvé cinq préservations sémantiques, quatre échecs et trois cas
incertains ; deux paires restent candidates après vérification des sources et
exclusion des copies. Ces décisions viennent d'une seule revue par l'assistant,
sans validation humaine indépendante. Aucune paraphrase n'a servi à entraîner
un modèle et aucun gain de performance n'est revendiqué.

## Lecture des résultats

Consultez [le rapport en anglais](light_research_report.md) et les rapports JSON
dans `reports/light_study/`. Le benchmark et ses étiquettes sont publiés dans
`benchmarks/fr_aspect_challenge_v1/`. Il s'agit d'un stress test rédigé par un
assistant, sans validation humaine indépendante, et non d'un benchmark externe
certifié. Aucune performance SFT, DPO, multi-GPU ou cluster n'est revendiquée.
