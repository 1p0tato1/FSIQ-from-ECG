# Prédiction du QI à 3 ans à partir de la HRV néonatale : résumé
*Groupe A · Jade · généré le 26/11/2026 par `01_prediction_qit.ipynb` · 10 × 5 plis*

## Conclusion
**Aucun modèle ne fait mieux que prédire le score moyen, sur aucune cible.** La HRV retrouve pourtant la prématurité (AUC 0.79 [0.67 ; 0.93]) : les données et la méthode savent voir un lien quand il existe.

## Données et méthode
- 189 enfants de la cohorte AuBE (enregistrement à la naissance, scores WPPSI-III complets) ; 25 indices HRV linéaires, 45 non linéaires, 6 covariables cliniques.
- Modèles : Arbre de décision, Elastic Net, Kernel ridge, Kernel ridge multi-noyaux, LightGBM, Random Forest, Régression linéaire, SVR, XGBoost ; réduction des variables dans chaque pli ; mêmes plis pour tous les modèles (10 × 5), évaluation par `evaluate_model.py` (outil commun de l'équipe).

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

## Tous les modèles (QIT)
MAE et R² : moyenne ± écart-type sur les plis ; gain sur la moyenne avec son IC à 95 % (`evaluate_model.py`).

| modèle | MAE | R² | gain vs moyenne [IC 95 %] | gain démontré |
|---|---|---|---|---|
| Elastic Net · toutes · Clinique seul | 10.33 ± 0.69 | -0.007 ± 0.041 | +0.06 [-0.12 ; 0.23] | non |
| Kernel ridge multi-noyaux · toutes · Clinique + HRV | 10.38 ± 0.73 | -0.020 ± 0.064 | +0.02 [-0.30 ; 0.32] | non |
| score moyen (référence) | 10.39 ± 0.66 | -0.005 ± 0.004 |  |  |
| Elastic Net · toutes · Clinique + HRV | 10.47 ± 0.72 | -0.025 ± 0.045 | -0.08 [-0.14 ; -0.02] | non, moins bien |
| SVR · PCA 90 % · Clinique + HRV | 10.51 ± 0.73 | -0.034 ± 0.065 | -0.12 [-0.22 ; -0.02] | non, moins bien |
| Random Forest · toutes · Clinique + HRV | 10.59 ± 0.68 | -0.047 ± 0.031 | -0.19 [-0.33 ; -0.05] | non, moins bien |
| XGBoost · sél. RF · Clinique + HRV | 10.61 ± 0.77 | -0.058 ± 0.066 | -0.22 [-0.44 ; -0.01] | non, moins bien |
| Kernel ridge · toutes · Clinique + HRV | 10.62 ± 0.75 | -0.068 ± 0.106 | -0.23 [-0.37 ; -0.09] | non, moins bien |
| Régression linéaire · pas à pas · Clinique + HRV | 10.89 ± 0.80 | -0.121 ± 0.115 | -0.50 [-0.75 ; -0.24] | non, moins bien |
| LightGBM · sél. RF · Clinique + HRV | 10.93 ± 0.86 | -0.126 ± 0.105 | -0.54 [-0.86 ; -0.23] | non, moins bien |
| Arbre de décision · toutes · Clinique + HRV | 11.14 ± 0.97 | -0.165 ± 0.150 | -0.74 [-1.03 ; -0.46] | non, moins bien |

## Méthodes ajoutées après les retours des enseignants (QIT)
- Régression linéaire · pas à pas · Clinique + HRV : MAE 10.89, gain -0.50 [-0.75 ; -0.24] (moins bien que la moyenne).
- Arbre de décision · toutes · Clinique + HRV : MAE 11.14, gain -0.74 [-1.03 ; -0.46] (moins bien que la moyenne).
- Kernel ridge · toutes · Clinique + HRV : MAE 10.62, gain -0.23 [-0.37 ; -0.09] (moins bien que la moyenne).
- Kernel ridge multi-noyaux · toutes · Clinique + HRV : MAE 10.38, gain +0.02 [-0.30 ; 0.32] (non démontré).
- Poids du kernel ridge multi-noyaux (médiane sur les plis) : clinique 1.00, HRV linéaire 0.00, HRV non linéaire 0.00.
- Classement des variables par un arbre de décision (importance d'impureté et SHAP) : pour le QIT, variables au-dessus du hasard : aucune ; accord des top 5 entre plis 0.08 (stable au-dessus de 0.15). Pour la prématurité : nl|alpha2 (DFADay), lin|pcNN20 (TD24h) ; accord 0.22 (stable au-dessus de 0.28).
- Réglages de chaque modèle (fixés ou cherchés, et comment) : section 3 du notebook ; réglages retenus pli par pli : `results/reglages_retenus_QIT.md`.

## Contrôle positif, classification, hypothèses
- Contrôle positif : HRV seule → prématurité, AUC 0.79 [0.67 ; 0.93] (lien démontré (IC > 0,5) ; discrimination acceptable ; accord faible).
- Classification QIT < 90 (40 enfants) : meilleure AUC 0.50 [0.39 ; 0.61] (Logistique · Clinique + HRV), exactitude équilibrée 0.50, kappa 0.00 (aucun lien démontré (l'IC contient 0,5) ; discrimination faible ; accord très faible).
- H1 · HRV à M0 (189 enfants) : gain -0.13 [-0.22 ; -0.04] (non soutenue : moins bien que la moyenne).
- H1 · HRV à M6 (176 enfants) : gain -0.07 [-0.32 ; 0.17] (non soutenue).
- H1 · HRV à M12 (177 enfants) : gain -0.03 [-0.07 ; 0.01] (non soutenue).
- H1 · HRV à M18 (173 enfants) : gain +0.01 [-0.14 ; 0.17] (non soutenue).
- H1 · HRV à M24 (182 enfants) : gain -0.14 [-0.24 ; -0.05] (non soutenue : moins bien que la moyenne).
- H2 · évolution M0 → M12 (175 enfants) : gain -0.05 [-0.19 ; 0.08] (non soutenue).
- H2 · évolution M0 → M24 (180 enfants) : gain -0.10 [-0.26 ; 0.05] (non soutenue).
- Avec 7 hypothèses testées, une hypothèse soutenue isolée peut être due au hasard : à confirmer.

## Limites
- 189 enfants ; pas de cohorte externe ; le QI à 3 ans dépend de nombreux facteurs non mesurés ici (environnement familial, niveau socio-économique).
- Une association n'est pas une cause : un modèle prédictif ne dit pas comment améliorer le QI d'un enfant.

## Suite
- Variables tirées des données brutes (fichiers RR, signal ECG) : travail du groupe B.
- Confirmer sur une autre cohorte toute hypothèse soutenue ci-dessus.
