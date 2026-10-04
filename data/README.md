# Données

Les jeux de données du cours ne sont pas inclus dans le dépôt.
Copiez ici `ftdataset_train.tsv` et `ftdataset_val.tsv` depuis vos données locales.
Vous pouvez également fournir `ftdataset_test.tsv`. En son absence, le programme
évalue sur le jeu de validation, ce qui ne constitue pas une évaluation indépendante.

Les fichiers TSV doivent contenir les colonnes `Avis`, `Prix`, `Cuisine`, `Service`.
Les étiquettes attendues sont `Positive`, `Négative`, `Neutre` et `NE`.
