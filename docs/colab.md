# Lancer les expériences dans Google Colab

[Ouvrir le notebook](https://colab.research.google.com/github/Juven14s/restaurant-sentiment-analysis/blob/main/notebooks/colab_training.ipynb)

## Premier essai

1. Choisissez **Exécution → Modifier le type d'exécution → GPU**. Un GPU n'est
   pas toujours disponible en offre gratuite ; le notebook vérifie sa présence.
2. Conservez `SMOKE_TEST = True`, `SEED = 42` et `USE_4BIT = False`.
3. Exécutez les cellules dans l'ordre. Autorisez le montage de votre Drive.
4. À la demande, envoyez vos fichiers locaux `ftdataset_train.tsv` et
   `ftdataset_val.tsv`. Dans le workspace d'origine ils se trouvent dans
   `ftproject/data/`. Les fichiers ne sont pas récupérés depuis GitHub.
5. Vérifiez les effectifs des splits, les logs et le statut `completed` dans
   les deux `run_manifest.json`. Le mode smoke limite le train à 32 avis, utilise
   toute la validation et ne lance aucune évaluation sur le test.

Les tests intégrés exécutent aussi de vrais entraînements sur des modèles
miniatures CPU, sans téléchargement. Ils ne remplacent pas l'essai sur le GPU
Colab ni la mesure de qualité des modèles pré-entraînés.

## Expérience complète

Après vérification du fonctionnement, passez `SMOKE_TEST = False` et relancez
dans l'ordre. Le notebook crée des runs complets distincts et réutilise les mêmes
splits. Les recettes de départ sont explicites dans `training_recipe`.

L'entraînement sélectionne un checkpoint avec la perte de validation. Après
lecture de cette validation et gel des choix, activez `RUN_TEST_EVALUATION = True`.
L'étape finale évalue classe majoritaire, TF-IDF, CamemBERT, Qwen sans adaptation puis
Qwen avec LoRA. Les deux Qwen partent de la même révision enregistrée.
Figez les choix de modèle et d'hyperparamètres avant de consulter ces résultats.

Un run peut prendre un temps variable selon le GPU, les limites de la session
et les téléchargements. Relevez le temps du premier essai avant de programmer
plusieurs expériences. Pour une étude multi-seeds, changez uniquement `SEED`
à 43 puis 44 ; la seed de préparation reste toujours à 42.

## Sauvegarde et reprise

Tout est conservé dans `Mon Drive/restaurant-sentiment-v1/` :

```text
code_revision.json        Commit du dépôt utilisé pour cette expérience
model_revisions.json      Révisions exactes des modèles Hugging Face
source_data/              TSV privés
data-text-seed42/          Splits gelés et manifeste
smoke_runs/               Runs pilotes, sans résultats test
full_runs/                Modèles, checkpoints, prédictions et métriques finales
baselines/                Résultats majoritaire et TF-IDF
```

Après une coupure, ouvrez le même notebook et exécutez de nouveau les cellules
depuis le début avec les mêmes réglages. Elles réutilisent le commit et les
révisions verrouillés, puis cherchent le dernier `checkpoint-*` contenant les
états du Trainer, de l'optimiseur, du scheduler et du générateur aléatoire.
Le travail depuis ce dernier checkpoint sera refait. `final/` sert à l'inférence,
pas à la reprise de l'optimiseur.

Conservez le **même chemin Drive** : les checkpoints du Trainer référencent le
répertoire de run d'origine. Évitez de déplacer un run interrompu. Un run déjà
terminé n'est pas réentraîné. Une modification des données ou de la configuration
bloque la reprise pour éviter de mélanger des expériences.

Si la session s'arrête avant le premier checkpoint complet, conservez les logs
et changez `RUN_TAG` pour recommencer dans un nouveau run. Si vous changez les
hyperparamètres ou activez QLoRA, changez également `RUN_TAG`. Pour changer les
sources ou le protocole, utilisez un nouvel `EXPERIMENT_NAME`.

## Mémoire GPU

Le notebook commence par LoRA standard sur Qwen3-0.6B, batch 1, séquences 512 et
accumulation 8. La compatibilité et la mémoire réelles restent à vérifier sur
le GPU attribué. Pour CamemBERT, réduisez le batch dans `training_recipe` en cas
de mémoire insuffisante. Pour Qwen, `USE_4BIT = True` installe bitsandbytes et
active QLoRA ; choisissez un nouveau `RUN_TAG`. Les séquences plus courtes
changent les entrées observées : documentez ce choix dans la comparaison.

La recette choisit FP16 lorsque le GPU ne prend pas en charge BF16. Aucun essai
CUDA ou 4-bit n'a été exécuté lors de la préparation locale du dépôt.

## Partager les résultats

Conservez les métriques, les réglages et les révisions. Vous pouvez partager
les rapports agrégés après vérification. Les TSV, checkpoints, prompts contenant
des avis et prédictions brutes restent privés par défaut. Le notebook public
est enregistré sans sorties d'exécution ni identifiants d'accès.

[Limites officielles de Colab](https://research.google.com/colaboratory/faq.html)
