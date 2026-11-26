"""
Pipeline deep learning - ÉTAPE 3 : CNN gelé, puis entraînement de l'agrégateur + MLP
avec recherche d'hyperparamètres.

1. Le CNN 1D (gelé, il n'apprend plus) transforme chaque fenêtre de 30 s en 128
   caractéristiques. C'est calculé UNE SEULE FOIS et enregistré (caracteristiques_cnn.npz).
2. On entraîne seulement l'agrégateur (LSTM / GRU / attention / moyenne) + le MLP sur
   ces caractéristiques. C'est très rapide, d'où la possibilité de tester beaucoup de
   réglages.
3. Recherche d'hyperparamètres par VALIDATION CROISÉE IMBRIQUÉE :
     - boucle externe (5 plis) : bébés de test jamais vus pendant les choix ;
     - boucle interne (3 plis) : choix de la meilleure configuration sur les seuls bébés
       d'entraînement du pli externe.
   On n'a donc pas de résultats trop optimistes dus au choix des réglages.
4. Comparaisons : référence (moyenne) et régression Ridge sur les caractéristiques
   moyennes du CNN, avec les mêmes plis.

Installation : pip install torch numpy pandas scikit-learn scipy tqdm
Lancement :    python dl_3_tete_gelee.py      (après dl_1_preparation.py)
"""

import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# ============ PARAMÈTRES À MODIFIER ============
DOSSIER_SEGMENTS = "segments_ecg"
FICHIER_QI = "Résultats WPPSI III.csv"
GROUPE_CIBLE = "Note composite"
CIBLE = "QIT"

# CNN gelé : "aleatoire" (poids aléatoires, aucune fuite d'information) ou chemin .pt
# d'un CNN pré-entraîné SANS le QI. Éviter modeles_dl/pli_*.pt : ces CNN ont vu le QI
# de bébés qui sont en test dans d'autres plis (fuite d'information).
CNN_GELE = "aleatoire"
FICHIER_CARACT = "caracteristiques_cnn.npz"   # cache des caractéristiques
CANAUX_BASE = 16
N_FEATURES_CNN = 128
TAILLE_PAQUET_CNN = 64

N_PLIS_EXTERNES = 5
N_PLIS_INTERNES = 3
N_CONFIGS = 12                  # configurations tirées au hasard (+ la configuration d'origine)
EPOCHS_MAX = 150
PATIENCE = 20
BEBES_PAR_LOT = 16
GRAINE = 0

# Espace de recherche des hyperparamètres
ESPACE = {
    "agregation": ["lstm_dernier", "lstm_moyenne", "gru_moyenne", "attention", "moyenne"],
    "taille_cachee": [16, 32, 64],
    "dropout": [0.0, 0.2, 0.5],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-4, 1e-2],
    "activation_mlp": ["relu", "gelu"],
    "neurones_mlp": [16, 32],
}
# Configuration du premier essai (pour comparaison), toujours incluse
CONFIG_ORIGINE = {"agregation": "lstm_dernier", "taille_cachee": 64, "dropout": 0.3,
                  "lr": 1e-3, "weight_decay": 1e-2, "activation_mlp": "relu",
                  "neurones_mlp": 32}
# ===============================================

APPAREIL = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fixer_graine(g):
    random.seed(g)
    np.random.seed(g)
    torch.manual_seed(g)


def par_paquets(liste, taille):
    for k in range(0, len(liste), taille):
        yield liste[k:k + taille]


# ---------------------------------------------------------------------------
# QI (fichier WPPSI-III, en-tête sur 2 lignes)
# ---------------------------------------------------------------------------
def charger_qi():
    for encodage in ("utf-8-sig", "cp1252"):
        try:
            brut = pd.read_csv(FICHIER_QI, sep="\t", header=None, dtype=str, encoding=encodage)
            break
        except UnicodeDecodeError:
            continue
    groupes = brut.iloc[0].ffill().fillna("").str.strip()
    noms = brut.iloc[1].fillna("").str.strip()
    df = brut.iloc[2:].reset_index(drop=True)
    df.columns = [f"{g} - {n}" if g else n for g, n in zip(groupes, noms)]
    nom_cible = f"{GROUPE_CIBLE} - {CIBLE}"
    if nom_cible not in df.columns:
        raise KeyError(f"Colonne '{nom_cible}' introuvable.")
    ids = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    qi = pd.to_numeric(df[nom_cible].str.replace(",", ".", regex=False), errors="coerce")
    ok = ids.notna() & qi.notna()
    return dict(zip(ids[ok].astype(int), qi[ok].astype(float)))


# ---------------------------------------------------------------------------
# 1. CNN gelé et extraction des caractéristiques
# ---------------------------------------------------------------------------
class BlocResiduel(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 7, stride, 3, bias=False)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 7, 1, 3, bias=False)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.raccourci = (nn.Identity() if c_in == c_out and stride == 1 else
                          nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride, bias=False),
                                        nn.GroupNorm(8, c_out)))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + self.raccourci(x))


class ResNet1D(nn.Module):
    def __init__(self, c=CANAUX_BASE, n_out=N_FEATURES_CNN):
        super().__init__()
        self.entree = nn.Sequential(
            nn.Conv1d(1, c, 15, 2, 7, bias=False), nn.GroupNorm(8, c), nn.ReLU(),
            nn.MaxPool1d(3, 2, 1))
        self.blocs = nn.Sequential(
            BlocResiduel(c, c, 1), BlocResiduel(c, 2 * c, 2),
            BlocResiduel(2 * c, 4 * c, 2), BlocResiduel(4 * c, 8 * c, 2))
        self.sortie = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                    nn.Dropout(0.2), nn.Linear(8 * c, n_out))

    def forward(self, x):
        return self.sortie(self.blocs(self.entree(x)))


def extraire_caracteristiques():
    cache = Path(FICHIER_CARACT)
    if cache.exists():
        d = np.load(cache)
        if str(d["source_cnn"]) == str(CNN_GELE):
            print(f"Caractéristiques lues dans {cache} (CNN : {CNN_GELE})")
            return d["X"], d["bebe"]
        print("Le cache a été calculé avec un autre CNN : nouveau calcul.")

    fixer_graine(GRAINE)
    cnn = ResNet1D().to(APPAREIL)
    if CNN_GELE != "aleatoire":
        etat = torch.load(CNN_GELE, map_location=APPAREIL)
        if any(k.startswith("cnn.") for k in etat):        # modèle complet -> partie CNN
            etat = {k[4:]: v for k, v in etat.items() if k.startswith("cnn.")}
        cnn.load_state_dict(etat)
        if "modeles_dl" in str(CNN_GELE):
            print("ATTENTION : ce CNN a été entraîné sur le QI d'une partie des bébés "
                  "-> fuite d'information, résultats trop optimistes.")
    cnn.eval()

    index = pd.read_csv(Path(DOSSIER_SEGMENTS) / "index.csv")
    X, bebe = [], []
    with torch.no_grad(), torch.autocast(APPAREIL.type, dtype=torch.float16,
                                         enabled=APPAREIL.type == "cuda"):
        for _, ligne in tqdm(index.iterrows(), total=len(index), desc="Extraction CNN",
                             unit="segment"):
            s = torch.from_numpy(np.load(ligne["chemin"]).astype(np.float32))
            s = s.unsqueeze(1).to(APPAREIL)
            f = torch.cat([cnn(p) for p in s.split(TAILLE_PAQUET_CNN)]).float().cpu().numpy()
            X.append(f.astype(np.float16))
            bebe.append(int(ligne["id"]))
    X, bebe = np.stack(X), np.array(bebe)
    np.savez(cache, X=X, bebe=bebe, source_cnn=str(CNN_GELE))
    print(f"Caractéristiques enregistrées : {X.shape} (segments, fenêtres, caractéristiques)")
    return X, bebe


# ---------------------------------------------------------------------------
# 2. Agrégateur + MLP (la partie entraînée)
# ---------------------------------------------------------------------------
class TeteQI(nn.Module):
    """(segments, fenêtres, 128) -> un QI (standardisé) par segment."""

    def __init__(self, cfg):
        super().__init__()
        self.agr = cfg["agregation"]
        h = cfg["taille_cachee"]
        self.norm = nn.LayerNorm(N_FEATURES_CNN)
        if self.agr.startswith("lstm"):
            self.rnn = nn.LSTM(N_FEATURES_CNN, h, batch_first=True)
        elif self.agr.startswith("gru"):
            self.rnn = nn.GRU(N_FEATURES_CNN, h, batch_first=True)
        elif self.agr == "attention":
            self.score = nn.Sequential(nn.Linear(N_FEATURES_CNN, 64), nn.Tanh(), nn.Linear(64, 1))
            self.proj = nn.Sequential(nn.Linear(N_FEATURES_CNN, h), nn.GELU())
        elif self.agr == "moyenne":
            self.proj = nn.Sequential(nn.Linear(N_FEATURES_CNN, h), nn.GELU())
        else:
            raise ValueError(self.agr)
        act = nn.ReLU() if cfg["activation_mlp"] == "relu" else nn.GELU()
        self.mlp = nn.Sequential(nn.Dropout(cfg["dropout"]), nn.Linear(h, cfg["neurones_mlp"]),
                                 act, nn.Dropout(cfg["dropout"]),
                                 nn.Linear(cfg["neurones_mlp"], 1))

    def forward(self, x):
        x = self.norm(x)
        if self.agr == "lstm_dernier":
            _, (h, _) = self.rnn(x)
            z = h[-1]
        elif self.agr in ("lstm_moyenne", "gru_moyenne"):
            sorties, _ = self.rnn(x)
            z = sorties.mean(dim=1)
        elif self.agr == "attention":
            poids = torch.softmax(self.score(x), dim=1)
            z = self.proj((poids * x).sum(dim=1))
        else:
            z = self.proj(x.mean(dim=1))
        return self.mlp(z).squeeze(-1)


def lot_de_segments(ids_lot, segs_de):
    """Indices des segments des bébés du lot + numéro du bébé (0..n-1) pour chaque segment."""
    segs = np.concatenate([segs_de[i] for i in ids_lot])
    groupe = np.concatenate([np.full(len(segs_de[i]), k) for k, i in enumerate(ids_lot)])
    return (torch.from_numpy(segs).to(APPAREIL),
            torch.from_numpy(groupe).long().to(APPAREIL))


def moyenne_par_bebe(p, groupe, n):
    """ŷ_i = moyenne des ŷ_ij des segments du bébé i."""
    somme = torch.zeros(n, device=p.device).index_add_(0, groupe, p)
    nombre = torch.zeros(n, device=p.device).index_add_(0, groupe, torch.ones_like(p))
    return somme / nombre


def predire(tete, X, ids, segs_de, moy, et):
    tete.eval()
    sorties = []
    with torch.no_grad():
        for lot in par_paquets(list(ids), 32):
            s, g = lot_de_segments(lot, segs_de)
            sorties.append(moyenne_par_bebe(tete(X[s]), g, len(lot)).cpu().numpy())
    return np.concatenate(sorties) * et + moy


def entrainer(cfg, X, segs_de, qi, ids_tr, ids_va, graine):
    """Entraîne la tête ; arrêt précoce sur ids_va. Renvoie (tête, moy, et, MAE val, epochs)."""
    fixer_graine(graine)
    y_tr = np.array([qi[i] for i in ids_tr])
    y_va = np.array([qi[i] for i in ids_va])
    moy, et = y_tr.mean(), y_tr.std() + 1e-8
    tete = TeteQI(cfg).to(APPAREIL)
    opt = torch.optim.AdamW(tete.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    meilleur, etat, attente, epoch = np.inf, None, 0, 0
    for epoch in range(1, EPOCHS_MAX + 1):
        tete.train()
        for lot in par_paquets(list(np.random.permutation(ids_tr)), BEBES_PAR_LOT):
            s, g = lot_de_segments(lot, segs_de)
            cible = torch.tensor([(qi[i] - moy) / et for i in lot], dtype=torch.float32,
                                 device=APPAREIL)
            p = moyenne_par_bebe(tete(X[s]), g, len(lot))
            perte = ((p - cible) ** 2).mean()
            opt.zero_grad()
            perte.backward()
            nn.utils.clip_grad_norm_(tete.parameters(), 1.0)
            opt.step()
        mae = np.mean(np.abs(predire(tete, X, ids_va, segs_de, moy, et) - y_va))
        if mae < meilleur - 1e-4:
            meilleur, etat, attente = mae, {k: v.clone() for k, v in tete.state_dict().items()}, 0
        else:
            attente += 1
            if attente >= PATIENCE:
                break
    if etat is not None:                     # None seulement si l'entraînement a divergé (NaN)
        tete.load_state_dict(etat)
    return tete, moy, et, meilleur, epoch


def tirer_configs():
    toutes = [dict(zip(ESPACE, v)) for v in itertools.product(*ESPACE.values())]
    rng = random.Random(GRAINE)
    choisies = rng.sample(toutes, min(N_CONFIGS, len(toutes)))
    return [CONFIG_ORIGINE] + [c for c in choisies if c != CONFIG_ORIGINE]


def metriques(y, p, nom):
    y, p = np.asarray(y), np.asarray(p)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = spearmanr(y, p).correlation if np.std(p) > 1e-6 else np.nan
    print(f"  {nom:24s} MAE {mae:6.2f} | RMSE {rmse:6.2f} | R² {r2:6.2f} | Spearman {rho:5.2f} "
          f"| à ±15 : {100 * np.mean(np.abs(y - p) <= 15):5.1f} % | écart-type préd. {np.std(p):5.2f}")


# ---------------------------------------------------------------------------
# 3. Programme principal
# ---------------------------------------------------------------------------
def main():
    print(f"Calcul sur : {APPAREIL}")
    X_np, bebe_seg = extraire_caracteristiques()
    qi = charger_qi()
    X = torch.from_numpy(X_np.astype(np.float32)).to(APPAREIL)
    segs_de = {int(i): np.where(bebe_seg == i)[0] for i in np.unique(bebe_seg)}
    ids = np.array(sorted(set(segs_de) & set(qi)))
    y = np.array([qi[i] for i in ids])
    print(f"{len(ids)} bébés avec ECG et QI, {X.shape[0]} segments de {X.shape[1]} fenêtres")

    # Caractéristiques moyennes par bébé (pour la Ridge)
    F = np.stack([X_np[segs_de[i]].astype(np.float32).mean(axis=(0, 1)) for i in ids])
    print(f"Variabilité des caractéristiques CNN entre bébés : écart-type moyen {F.std(0).mean():.4f}")

    configs = tirer_configs()
    print(f"{len(configs)} configurations testées par validation croisée imbriquée "
          f"({N_PLIS_EXTERNES} x {N_PLIS_INTERNES} plis)")

    journal, resultats = [], []
    externes = KFold(N_PLIS_EXTERNES, shuffle=True, random_state=GRAINE)
    debut = time.time()
    barre = tqdm(total=N_PLIS_EXTERNES * (len(configs) * N_PLIS_INTERNES + 1),
                 desc="Entraînements", unit="modèle", dynamic_ncols=True)
    for num, (i_tr, i_te) in enumerate(externes.split(ids), start=1):
        ids_tr, ids_te = ids[i_tr], ids[i_te]

        # --- Choix des hyperparamètres sur les seuls bébés d'entraînement ---
        internes = KFold(N_PLIS_INTERNES, shuffle=True, random_state=GRAINE + num)
        scores = []
        for c, cfg in enumerate(configs):
            maes, epochs = [], []
            for j_tr, j_va in internes.split(ids_tr):
                _, _, _, mae, ep = entrainer(cfg, X, segs_de, qi, ids_tr[j_tr], ids_tr[j_va],
                                             GRAINE + c)
                maes.append(mae)
                epochs.append(ep)
                barre.update()
            scores.append(np.mean(maes))
            journal.append({"pli": num, "config": c, **cfg,
                            "MAE_validation_interne": np.mean(maes),
                            "epochs_moyen": np.mean(epochs)})
        meilleure = int(np.argmin(scores))
        cfg = configs[meilleure]
        tqdm.write(f"Pli {num} : meilleure configuration n°{meilleure} "
                   f"(MAE interne {scores[meilleure]:.2f}) : {cfg}")

        # --- Entraînement final sur tout l'entraînement du pli, évaluation sur le test ---
        ids_fit, ids_es = train_test_split(ids_tr, test_size=0.15, random_state=GRAINE)
        tete, moy, et, _, _ = entrainer(cfg, X, segs_de, qi, ids_fit, ids_es, GRAINE)
        barre.update()
        p_tete = predire(tete, X, ids_te, segs_de, moy, et)

        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30)))
        p_ridge = ridge.fit(F[i_tr], y[i_tr]).predict(F[i_te])

        for k, i in enumerate(ids_te):
            resultats.append({"id": int(i), "pli": num, "QI_reel": qi[i],
                              "QI_tete": p_tete[k], "QI_ridge": p_ridge[k],
                              "QI_reference": y[i_tr].mean(), "config": meilleure})
    barre.close()

    journal = pd.DataFrame(journal)
    journal.to_csv("recherche_hyperparametres.csv", index=False)
    res = pd.DataFrame(resultats).sort_values("id")
    res.to_csv("predictions_tete_gelee.csv", index=False)

    print(f"\nTerminé en {(time.time() - debut) / 60:.1f} min")
    print("\nInfluence de chaque hyperparamètre (MAE de validation interne moyenne) :")
    for h in ESPACE:
        moyennes = journal.groupby(h)["MAE_validation_interne"].mean().round(2)
        print(f"  {h:15s} " + " | ".join(f"{k}: {v}" for k, v in moyennes.items()))

    print("\nRésultats sur les bébés de test (validation croisée externe) :")
    metriques(res["QI_reel"], res["QI_tete"], "Tête (CNN gelé)")
    metriques(res["QI_reel"], res["QI_ridge"], "Ridge (caract. moyennes)")
    metriques(res["QI_reel"], res["QI_reference"], "Référence (moyenne)")
    print("\nFichiers : predictions_tete_gelee.csv, recherche_hyperparametres.csv")


if __name__ == "__main__":
    main()
