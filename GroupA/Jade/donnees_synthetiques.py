"""Fausses données au format exact des 3 fichiers du projet, pour essayer 01_prediction_qit.ipynb sans les vraies.

Même forme que les vrais exports : mêmes noms de colonnes (avec l'accent illisible « Num�ro »), ligne d'unités,
cases « - », passage écrit « MO », colonnes Day/Night vides à M0, deux lignes d'en-tête dans le WPPSI, lignes en
double… Les valeurs sont tirées au hasard : AUCUNE donnée réelle. Deux liens sont plantés exprès, pour que les
analyses aient quelque chose à trouver : un lien fort entre la HRV et la prématurité, et un lien faible entre la HRV
et le QI.

    python donnees_synthetiques.py                 -> écrit dans data/synthetique/ (ignoré par git)
    python donnees_synthetiques.py autre/dossier   -> écrit ailleurs

Dans le notebook, DONNEES_SYNTHETIQUES = True les crée automatiquement si besoin.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def generer(dossier):
    """Écrit HRV_Lineaire_Full.csv, HRV_NonLineaire_Full.csv et Résultats WPPSI III.csv dans `dossier`."""
    out = Path(dossier)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 200
    ids = np.arange(1, n + 1)
    prema = rng.binomial(1, 0.3, n)
    latent = rng.normal(size=n)
    qit = np.clip(np.round(100 + 5 * latent - 6 * prema + rng.normal(0, 11, n)), 55, 140)
    X = "�"   # accent illisible, écrit tel quel dans les vrais fichiers

    META = [f"Num{X}ro", "Monogramme", "DDN", "Sexe", "Tabac", f"Diab{X}te", "HTA", "Maturation Pulm", f"Pr{X}matur{X}",
            "SFA", f"R{X}a {X} la naissance", "Pathologie neurologique", "Sortie du service", f"D{X}pression: prise d'IRS",
            "Filename", "Date", "Recorder", "Passage", "Total duration", "Day duration", "Night duration",
            "Night beginning", "Night end", "nb RR total", "nb RR total day", "nb RR total night", "nb RR analyzed total",
            "nb RR analyzed day", "nb RR analyzed night", "nb RR excluded total", "nb RR excluded day",
            "nb RR excluded night", "nb RR corrected total", " nb RR corrected day", "nb RR corrected night"]
    TD = ["Mean RR", "Mean HR", "NN20", "pcNN20", "NN30", "pcNN30", "NN50", "pcNN50", "SDNN", "RMSSD", "SDANN", "SDNNIDX"]
    GI = ["Tri ind", "TINN", "X", "Y", "N", "M"]
    INDICES_LIN = ([f"{i} (TD{p})" for p in ["24h", "Day", "Night"] for i in TD]
                   + [f"{i} (FT24h)" for i in ["Ptot", "ULF", "VLF", "LF", "HF"]]
                   + [f"{i} (FTDay)" for i in ["Ptot", "VLF", "LF", "HF", "LFnu", "HFnu"]] + ["LF_HF (24h)"]
                   + [f"{i} (FTNight)" for i in ["Ptot", "VLF", "LF", "HF", "LFnu", "HFnu", "LF/HF"]]
                   + [f"{i} (GI{p})" for p in ["24h", "Day", "Night"] for i in GI]
                   + [f"{i} (IncNight)" for i in ["TIpsd", "VLFIpsd", "%VLFIpsd", "LFIpsd", "%LFIpsd", "HFIpsd", "%HFIpsd"]])
    P3 = ["24h", "Day", "Night"]
    META_NL = META[:15] + ["date", "Recorder", "Passage", "Total duration", "Day duration", "Night duration",
                           "Night beginning", "Night end", "nb RR total", "nb analyzed RR total", "% analyzed RR total",
                           "nb RR total Day", "nb analyzed RR day", "% analyzed RR day", "nb RR total Night",
                           "nb analyzed RR night", "% analyzed RR night"]
    INDICES_NL = ([f"{i} (PCP{p})" for p in P3 for i in [f"Centro{X}d", "SD1", "SD2", "SD1SD2" if p == "24h" else "SD1/SD2",
                                                          "SD1nu", "SD2nu"]]
                  + ["nb VPC (Turb24h)", "TO (Turb24h)", "TS (Turb24h)"]
                  + [f"{i} (DFA{p})" for p in P3 for i in ["alpha1", "alpha2", "H"]]
                  + [f"{i} ({p})" for i in ["Higuchi", "Katz", "Hurst"] for p in P3] + ["Beta 1f (Slope24h)"]
                  + [f"{i} (MT{p})" for i in ["Skewness", "Kurtosis"] for p in P3] + [f"Lyapunov (LLE{p})" for p in P3]
                  + ["AC (ACDC24h)", "DC (ACDC24h)"]
                  + [f"{i} (Symb{p})" for p in P3 for i in ["0V", "0V%", "1V", "1V%", "2LV", "2LV%", "2UV", "2UV%", "MP", "MP%"]]
                  + [f"{i}(EMD{p})" for p in P3 for i in ["pLF1", "pLF2", "pHF1", "pHF2", "IMAI1", "IMAI2"]]
                  + [f"{i} (Entropy{p})" for i in ["AppEn", "SampEn"] for p in P3]
                  + [f"{i} (Entropy{p})" for p in P3 for i in ["SE", "CE", "CCE", "NCCE", "rho"]]
                  + [f"LZC (Entropy{p})" for p in P3])

    def indices(noms, idx):
        cols, commun = {}, rng.normal(size=len(idx))
        for j, nom in enumerate(noms):
            z = (0.7 * commun + 0.5 * rng.normal(size=len(idx)) + (0.25 * latent[idx] if j % 7 == 0 else 0)
                 + (0.9 * prema[idx] if j % 4 == 1 else 0))          # la prématurité change vraiment la HRV
            v = 10 ** rng.uniform(-1, 4) * np.exp(0.6 * z)
            v[rng.random(len(idx)) < 0.02] = np.nan
            cols[nom] = v
        return cols

    def ecrire_tsv(chemin, df, unites):
        with open(chemin, "w", encoding="utf-8", newline="") as f:
            f.write("\t".join(df.columns) + "\n" + "\t".join(unites) + "\n")
            df.to_csv(f, sep="\t", decimal=",", index=False, header=False)

    lin_parts, nl_parts = [], []
    for passage in ["MO", "M6", "M12", "M18", "M24"]:   # « MO » avec la lettre O, comme dans le vrai fichier
        idx = (ids if passage == "MO" else ids[rng.random(n) < 0.7]) - 1
        if passage == "MO":
            idx = np.concatenate([idx[10:], idx[:3]])   # 10 enfants sans M0, 3 avec deux enregistrements M0
        k = len(idx)
        tot = rng.integers(80000, 150000, k)
        ana = (tot * rng.uniform(0.4, 1.0, k)).astype(int)
        meta = {META[0]: [f"AUBE-{i + 1:03d}" for i in idx], "Monogramme": "XX", "DDN": "12/03/2014",
                "Sexe": rng.choice(["M", "F"], k), "Tabac": rng.binomial(1, 0.2, k), META[5]: rng.binomial(1, 0.05, k),
                "HTA": rng.binomial(1, 0.1, k), "Maturation Pulm": rng.binomial(1, 0.2, k), META[8]: prema[idx],
                "SFA": rng.binomial(1, 0.1, k), META[10]: rng.binomial(1, 0.15, k),
                "Pathologie neurologique": rng.binomial(1, 0.05, k), "Sortie du service": rng.choice(["Mat", "Neonat"], k),
                META[13]: rng.binomial(1, 0.05, k), "Filename": [f"aube{i + 1:03d}_{passage}.rr" for i in idx],
                "Date": "14/03/2014", "Recorder": (rng.random(k) < 0.02).astype(float), "Passage": passage,
                "Total duration": "19:54:00", "Day duration": "12:00:00", "Night duration": "07:54:00",
                "Night beginning": "22:00", "Night end": "07:00", "nb RR total": tot, "nb RR analyzed total": ana}
        meta = {c: meta.get(c, rng.integers(100, 9000, k)) for c in META}
        lin_parts.append(pd.concat([pd.DataFrame(meta), pd.DataFrame(indices(INDICES_LIN, idx))], axis=1))
        nl_meta = {**{c: meta[c] for c in META[:15]}, META[0]: [f"aube {i + 1}" for i in idx], "date": meta["Date"],
                   "Recorder": meta["Recorder"], "Passage": passage, "nb RR total": tot, "nb analyzed RR total": ana,
                   "% analyzed RR total": np.round(100 * ana / tot, 1)}
        nl_meta = {c: nl_meta.get(c, rng.integers(100, 9000, k)) for c in META_NL}
        nl_parts.append(pd.concat([pd.DataFrame(nl_meta), pd.DataFrame(indices(INDICES_NL, idx))], axis=1))

    def comme_le_vrai_fichier(df, periodes_vides_m0):
        """Colonnes de certaines périodes vides à M0, nombres de RR vides, 3 colonnes vides en fin de ligne."""
        m0 = df.index[df["Passage"] == "MO"]
        df.loc[m0, [c for c in df.columns if c.endswith(periodes_vides_m0)]] = np.nan
        df[[c for c in df.columns if c.startswith("nb RR total")]] = np.nan
        for nom in ["", " ", "  "]:
            df[nom] = np.nan
        return df

    lin = comme_le_vrai_fichier(pd.concat(lin_parts, ignore_index=True), ("Day)", "Night)"))
    lin.iloc[5, lin.columns.get_loc("LF/HF (FTNight)")] = np.inf
    unites_lin = ["", "", "", "M/F", "0/1", "0/1", "0/1", "0/1", "0/1", "HIV 3-4", "LMPV", "Dilatation ventr",
                  "Exam neuro normal", f"Labilit{X} cardiaq", "0/1", "Unit", "-", "-", "Hz", "-", ""]
    unites_lin += ["-"] * (len(META) - len(unites_lin)) + rng.choice(["ms", "bpm", "-", "%", "ms^2"], len(INDICES_LIN)).tolist()
    unites_lin += ["", "", ""]
    ecrire_tsv(out / "HRV_Lineaire_Full.csv", lin, unites_lin)
    # lignes « vides » remplies de « - » (enfant sans enregistrement), comme dans le vrai fichier
    with open(out / "HRV_Lineaire_Full.csv", "a", encoding="utf-8", newline="") as f:
        for i in rng.choice(ids, 120):
            f.write(f"AUBE-{i:03d}\tXX\t12/03/2014\t" + "\t".join(["-"] * (lin.shape[1] - 3)) + "\n")
    nl = comme_le_vrai_fichier(pd.concat(nl_parts, ignore_index=True), ("Night)",))
    unites_nl = unites_lin[:14] + ["-", "-", "-", "Hz", "HH:MM:SS", "", "HH:MM:SS", "HH:MM:SS", "HH:MM:SS"]
    unites_nl += ["n", "n", "%"] * 3 + rng.choice(["ms", "-", "%", "n", "ms^2", "ms/beat"], len(INDICES_NL)).tolist() + ["", "", ""]
    ecrire_tsv(out / "HRV_NonLineaire_Full.csv", nl, unites_nl)

    # WPPSI : 2 lignes d'en-tête, Windows-1252, « ; »
    w_ids = ids[rng.random(n) < 0.9]
    k = w_ids - 1
    qiv, qip, ncl = (np.round(qit[k] + rng.normal(0, s, len(k))) for s in (6, 6, 8))

    def pct(q):
        return np.clip(np.round(100 * (q - 55) / 90), 1, 99)

    blocs = [("Note composite", [qiv, qip, qit[k], ncl]), ("Rang percentile", [pct(qiv), pct(qip), pct(qit[k]), pct(ncl)]),
             ("Intervalle de confiance (95%)", [[f"{int(a) - 6}-{int(a) + 6}" for a in v] for v in [qiv, qip, qit[k], ncl]])]
    haut = ["", "", "", "", "", "NS Verb.", "", "NS Perf.", ""]
    bas = ["N°", "", "DDN", "Age", "COM", "INF", "DIM", "CUB", "AOB"]
    ddn_w = ["12/03/2014"] * len(k)
    ddn_w[20], ddn_w[21] = "01/01/2015", "02/02/2015"   # 2 enfants dont la date de naissance ne correspond pas
    cols = [pd.Series(w_ids, dtype=object), pd.Series([""] * len(k)), pd.Series(ddn_w),
            pd.Series(["2a10m"] * len(k))] + [pd.Series(rng.integers(1, 19, len(k))) for _ in range(5)]
    for nom_bloc, valeurs in blocs:
        for j, (echelle, v) in enumerate(zip(["QIV", "QIP", "QIT", "NCL"], valeurs)):
            haut.append(nom_bloc if j == 0 else "")
            bas.append(echelle)
            cols.append(pd.Series(v))
    corps = pd.concat(cols, axis=1)
    corps.columns = range(corps.shape[1])
    extra = corps.iloc[[0, 1]].copy()   # lignes en double : une partielle, une contradictoire
    extra.iloc[0, 9:] = np.nan
    extra.iloc[1, 11] = 77
    corps = pd.concat([corps, extra], ignore_index=True)
    corps.iloc[2, 0] = f"0{corps.iloc[2, 0]}"
    with open(out / "Résultats WPPSI III.csv", "w", encoding="cp1252", newline="") as f:
        f.write(";".join(haut) + "\n" + ";".join(bas) + "\n")
        corps.to_csv(f, sep=";", decimal=",", index=False, header=False)
    print(f"Données synthétiques écrites dans {out}")


generate = generer   # nom anglais (notebook anglais)


if __name__ == "__main__":
    ici = Path(__file__).resolve().parent
    racine = next(d for d in [ici, *ici.parents] if (d / "evaluate_model.py").exists())
    generer(sys.argv[1] if len(sys.argv) > 1 else racine / "data" / "synthetique")
