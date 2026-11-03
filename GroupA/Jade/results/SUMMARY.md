# Prédiction du QIT à partir de la HRV néonatale : résumé
*Groupe A · Jade · généré le 03/11/2026 par `01_prediction_qit.ipynb` · mode rapide (1 répétition), chiffres provisoires*

## Données et méthode
- 189 enfants de la cohorte AuBE avec un enregistrement M0 et des scores WPPSI-III complets ; 25 indices HRV linéaires, 45 non linéaires, 6 covariables cliniques.
- Modèles : Elastic Net, LightGBM, Random Forest, SVR, XGBoost ; sélection de variables dans les plis (forêt, PCA 90 %) ; hyperparamètres réglés par validation croisée interne.
- Évaluation : plis communs de l'équipe (`evaluate_model.py`), cibles QIT, QIV, QIP, NCL.

## Résultats (meilleur modèle par cible)
| Cible | Modèle | MAE ± écart-type | R² | MAE de la moyenne | Gain vs moyenne [IC 95 %] |
|---|---|---|---|---|---|
| QIT | Elastic Net · toutes · Clinique seul | 10.22 ± 0.70 | 0.013 | 10.39 | +0.17 [0.00 ; 0.35] (démontré) |
| QIV | Elastic Net · toutes · Clinique seul | 11.56 ± 1.65 | -0.006 | 11.66 | +0.10 [-0.11 ; 0.31] (non démontré) |
| QIP | SVR · PCA 90 % · Clinique + HRV | 9.73 ± 1.28 | -0.015 | 9.76 | +0.02 [-0.03 ; 0.08] (non démontré) |
| NCL | Elastic Net · toutes · Clinique seul | 11.70 ± 1.86 | -0.043 | 11.68 | -0.02 [-0.25 ; 0.21] (non démontré) |

- QIT : test de permutation p = 0.048 ; intervalle conforme à 90% : ± 21.3 points (couverture observée 90%, contre ± 20.9 autour du QIT moyen).
- Apport de la HRV au modèle clinique (Elastic Net, QIT) : MAE +0.17 points.
- Référence : rapport de Samuel Fornet, MAE ≈ 10,3 et R² ≤ 0 en validation croisée imbriquée.

## Limites
- Petit effectif (≈ 180 enfants) pour ≈ 200 variables ; pas de cohorte externe de validation.
- Signal faible : une association n'est pas une cause, et la prédiction individuelle reste imprécise.

## Suite
- Variables tirées des données brutes (fichiers RR, signal ECG avec le groupe B) ; familles d'indices séparées.
- Elastic Net bayésien / processus gaussien ; classification QIT < 90 (AUC, kappa).
