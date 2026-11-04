"""Vérifie le test de la section 4.5 (outils_qit.bornes_correlation) sur des données SIMULÉES dont on connaît la
vérité : 189 enfants, 4 indices (3 très corrélés entre eux, comme RMSSD, HF et SDNN, plus 1 indépendant).

1. aucun lien         : le test doit annoncer un lien à tort au plus 5 fois sur 100 ;
2. lien au seuil      : un lien qui explique la part de variance annoncée comme détectable doit être vu ≈ 8 fois sur 10,
                        et l'intervalle doit contenir la vraie corrélation au moins 95 fois sur 100 ;
3. confusion          : la prématurité baisse à la fois la HRV et le QI ; la corrélation brute voit un lien, la
                        corrélation ajustée doit l'éliminer.

Lancer depuis n'importe où dans le dépôt : python GroupA/Jade/verifier_bornes.py (environ 1 minute).
Aucune donnée réelle n'est utilisée.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ICI = Path(__file__).resolve().parent
RACINE = next(d for d in [ICI, *ICI.parents] if (d / "evaluate_model.py").exists())
sys.path[:0] = [str(RACINE), str(ICI)]
import outils_qit as outils  # noqa: E402

N, M = 189, 4
R2_SEUIL = outils.variance_detectable(N, M)
R_SEUIL = np.sqrt(R2_SEUIL)


def un_essai(scenario, graine):
    g = np.random.default_rng(graine)
    prema, sexe = g.binomial(1, 0.3, N), g.binomial(1, 0.5, N)
    vagal = g.normal(size=N) - 1.0 * prema                       # la prématurité baisse le tonus vagal
    X = pd.DataFrame({"RMSSD": np.exp(vagal + 0.4 * g.normal(size=N)),       # asymétriques et très corrélés
                      "HF": np.exp(2 * vagal + 0.6 * g.normal(size=N)),
                      "SDNN": np.exp(0.7 * vagal + 0.7 * g.normal(size=N)),
                      "autre": g.normal(size=N)})
    bruit = g.normal(size=N)
    if scenario == "aucun lien":
        y = 100 + 15 * bruit
    elif scenario == "lien au seuil":                          # le QI dépend de RMSSD
        z = X["RMSSD"].rank().to_numpy()
        z = (z - z.mean()) / z.std()
        y = 100 + 15 * (R_SEUIL * z + np.sqrt(1 - R2_SEUIL) * bruit)
    else:                                                      # confusion par la prématurité
        y = 100 + 15 * bruit - 12 * prema
    b = outils.bornes_correlation(X, pd.Series(y), pd.DataFrame({"prema": prema, "sexe": sexe.astype(float)}))
    return {"faux positif brut": (b["démontré"] == "oui").any(),
            "faux positif ajusté": (b["démontré ajusté"] == "oui").any(),
            "RMSSD détecté": b.loc["RMSSD", "démontré"] == "oui",
            "IC couvre la vraie valeur": b.loc["RMSSD", "IC bas"] <= 0.96 * R_SEUIL <= b.loc["RMSSD", "IC haut"]}


if __name__ == "__main__":
    print(f"Seuil annoncé comme détectable : {R2_SEUIL:.1%} de variance (corrélation {R_SEUIL:.2f})")
    for scenario, n_essais in [("aucun lien", 4000), ("lien au seuil", 2000), ("confusion", 2000)]:
        r = pd.DataFrame(Parallel(n_jobs=-1)(delayed(un_essai)(scenario, s) for s in range(n_essais))).mean()
        print(f"\n== {scenario} ({n_essais} jeux simulés)")
        if scenario == "aucun lien":
            print(f"lien annoncé à tort : brut {r['faux positif brut']:.1%}, ajusté {r['faux positif ajusté']:.1%}"
                  "  (attendu : au plus 5 %)")
        elif scenario == "lien au seuil":
            print(f"lien détecté : {r['RMSSD détecté']:.1%}  (attendu : environ 80 %)")
            print(f"intervalle qui contient la vraie corrélation : {r['IC couvre la vraie valeur']:.1%}"
                  "  (attendu : au moins 95 %)")
        else:
            print(f"lien annoncé à tort : brut {r['faux positif brut']:.1%} (la confusion crée un lien brut), "
                  f"ajusté {r['faux positif ajusté']:.1%}  (attendu : au plus 5 %)")
