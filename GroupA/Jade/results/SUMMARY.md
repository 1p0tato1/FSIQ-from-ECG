# Prédiction du QI à 3 ans à partir de la HRV néonatale : résumé
*Groupe A · Jade · généré le 04/11/2026 par `01_prediction_qit.ipynb` · 10 × 5 plis*

## Conclusion
**Aucun modèle ne fait mieux que prédire le score moyen, sur aucune cible.** La HRV retrouve pourtant bien la prématurité (AUC 0.79) : les données et la méthode savent voir un lien quand il existe.

## Données et méthode
- 189 enfants de la cohorte AuBE (enregistrement à la naissance, scores WPPSI-III complets) ; 25 indices HRV linéaires, 45 non linéaires, 6 covariables cliniques.
- Modèles : Elastic Net, LightGBM, Random Forest, SVR, XGBoost ; réduction des variables dans chaque pli ; évaluation sur les plis communs de l'équipe (10 × 5, `evaluate_model.py`).

## Régression (meilleur modèle par cible)
| Cible | Modèle | MAE ± écart-type | R² | MAE de la moyenne | Gain vs moyenne [IC 95 %] |
|---|---|---|---|---|---|
| QIT | Elastic Net · toutes · Clinique seul | 10.33 ± 0.69 | -0.007 | 10.39 | +0.06 [-0.12 ; 0.23] (non démontré) |
| QIV | Elastic Net · toutes · Clinique seul | 11.68 ± 1.20 | -0.010 | 11.66 | -0.02 [-0.22 ; 0.16] (non démontré) |
| QIP | Elastic Net · toutes · Clinique seul | 9.75 ± 0.94 | -0.017 | 9.75 | +0.00 [-0.16 ; 0.17] (non démontré) |
| NCL | Elastic Net · toutes · Clinique seul | 11.64 ± 1.23 | -0.021 | 11.62 | -0.01 [-0.15 ; 0.13] (non démontré) |

- Ajouter la HRV au modèle clinique (Elastic Net, QIT) : la MAE passe de 10.33 à 10.47 (moins bien).
- Signal dans les variables : la plus forte corrélation avec le QIT vaut 0.16, contre 0.17 en moyenne par hasard (jusqu'à 0.21) : rien ne dépasse le hasard, pas de lien à apprendre.
- Limite de détection (simulation) : avec ces enfants, la méthode voit un lien dès qu'il explique environ 20% de la variance du QI ; si la HRV en explique une part, elle est plus petite.
- Borne (indices choisis à l'avance : RMSSD (TD24h), HF (FT24h), SDNN (TD24h), 1re composante HRV) : aucun lien démontré ; aucun n'explique plus de 6% de la variance du QIT (6% au-delà des covariables cliniques), intervalles simultanés à 95 % ; un lien de 6% aurait été détecté 8 fois sur 10.
- Intervalle de prédiction à 90% (prédiction conforme) : ± 21.3 points, contre ± 20.9 autour du QIT moyen.

## Contrôle positif, classification, hypothèses
- Contrôle positif : HRV seule → prématurité, AUC 0.79 [0.67 ; 0.93].
- Classification QIT < 90 (40 enfants) : meilleure AUC 0.50 [0.39 ; 0.61] (Logistique · Clinique + HRV), exactitude équilibrée 0.51, kappa 0.02.
- H1 · HRV à M0 (189 enfants) : gain -0.13 [-0.22 ; -0.04] (non soutenue).
- H1 · HRV à M6 (176 enfants) : gain -0.07 [-0.32 ; 0.17] (non soutenue).
- H1 · HRV à M12 (177 enfants) : gain -0.03 [-0.07 ; 0.01] (non soutenue).
- H1 · HRV à M18 (173 enfants) : gain +0.01 [-0.14 ; 0.17] (non soutenue).
- H1 · HRV à M24 (182 enfants) : gain -0.14 [-0.24 ; -0.05] (non soutenue).
- H2 · évolution M0 → M12 (175 enfants) : gain -0.05 [-0.19 ; 0.08] (non soutenue).
- H2 · évolution M0 → M24 (180 enfants) : gain -0.10 [-0.26 ; 0.05] (non soutenue).
- Avec 7 hypothèses testées, une hypothèse soutenue isolée peut être due au hasard : à confirmer.

## Limites
- Environ 190 enfants ; pas de cohorte externe ; le QI à 3 ans dépend de nombreux facteurs non mesurés ici (environnement familial, niveau socio-économique).
- Une association n'est pas une cause : un modèle prédictif ne dit pas comment améliorer le QI d'un enfant.

## Suite
- Variables tirées des données brutes (fichiers RR, signal ECG) : travail du groupe B.
- Confirmer sur une autre cohorte toute hypothèse soutenue ci-dessus.
