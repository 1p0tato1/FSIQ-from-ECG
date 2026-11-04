"""
Pipeline deep learning - ÉTAPE 2 : modèle et entraînement (prédiction du QI à 2 ans).

Pour chaque segment j du bébé i  (matrice nb_fenêtres x 6000) :
    1D-ResNet  : chaque fenêtre de 30 s       -> 128 caractéristiques
    LSTM / GRU / Transformer : les fenêtres   -> 1 vecteur de 64
    MLP        : -> QI prédit pour le segment, ŷ_ij
Puis  ŷ_i = moyenne des ŷ_ij sur les S segments du bébé
et    perte = Σ_i (ŷ_i - y_i)²

Évaluation : validation croisée PAR BÉBÉ (un bébé n'est jamais à la fois en
entraînement et en test), avec arrêt précoce sur des bébés de validation.

Installation : pip install torch numpy pandas scikit-learn scipy
(pour un GPU NVIDIA, suivez https://pytorch.org/get-started/locally/)
Lancement :    python dl_2_entrainement.py      (lancer dl_1_preparation.py avant)
"""

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from torch.utils.checkpoint import checkpoint

# ============ PARAMÈTRES À MODIFIER ============
DOSSIER_SEGMENTS = "segments_ecg"          # sortie de dl_1_preparation.py
FICHIER_QI = "Résultats WPPSI III.csv"
GROUPE_CIBLE = "Note composite"            # groupe de colonnes (1re ligne d'en-tête)
CIBLE = "QIT"                              # QIT = QI total ; ou "QIV", "QIP", "NCL"

MODELE_SEQUENCE = "lstm"                   # "lstm", "gru" ou "transformer"
N_FEATURES_CNN = 128                       # sortie du 1D-ResNet par fenêtre
N_FEATURES_SEQ = 64                        # sortie du LSTM / GRU / Transformer
CANAUX_BASE = 16                           # largeur du ResNet (16 = petit réseau)
POIDS_PRE_ENTRAINES = None                 # chemin .pt d'un ResNet pré-entraîné (fine-tuning)

N_PLIS = 5                                 # validation croisée (par bébé)
PART_VALIDATION = 0.15                     # bébés d'entraînement réservés à l'arrêt précoce
EPOCHS_MAX = 60
PATIENCE = 10                              # arrêt si pas d'amélioration pendant N epochs
TAUX_APPRENTISSAGE = 1e-3
WEIGHT_DECAY = 1e-2
BEBES_PAR_PAS = 4                          # bébés par mise à jour (accumulation de gradient)
TAILLE_PAQUET_CNN = 60                     # fenêtres passées ensemble dans le CNN (mémoire)
AUGMENTATION = True                        # bruit + variation d'amplitude à l'entraînement
GRAINE = 0
# ===============================================

APPAREIL = torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")
AMP = APPAREIL.type == "cuda"              # calcul en demi-précision sur GPU NVIDIA


def fixer_graine(g):
    random.seed(g)
    np.random.seed(g)
    torch.manual_seed(g)


# ---------------------------------------------------------------------------
# Lecture du QI (fichier WPPSI-III)
# ---------------------------------------------------------------------------
def charger_qi():
    """Lit le fichier WPPSI : en-tête sur 2 lignes (groupe / colonne), numéro du bébé
    dans la 1re colonne. Renvoie {numéro du bébé: QI}."""
    for encodage in ("utf-8-sig", "cp1252"):
        try:
            brut = pd.read_csv(FICHIER_QI, sep="\t", header=None, dtype=str, encoding=encodage)
            break
        except UnicodeDecodeError:
            continue
    # Ligne 0 : groupes (cellules fusionnées -> on propage vers la droite) ; ligne 1 : noms
    groupes = brut.iloc[0].ffill().fillna("").str.strip()
    noms = brut.iloc[1].fillna("").str.strip()
    colonnes = [f"{g} - {n}" if g else n for g, n in zip(groupes, noms)]
    df = brut.iloc[2:].reset_index(drop=True)
    df.columns = colonnes

    nom_cible = f"{GROUPE_CIBLE} - {CIBLE}"
    if nom_cible not in df.columns:
        dispo = [c for c in colonnes if " - " in c and not c.endswith(" - ")]
        raise KeyError(f"Colonne '{nom_cible}' introuvable. Disponibles : {dispo}")
    ids = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    qi = pd.to_numeric(df[nom_cible].str.replace(",", ".", regex=False), errors="coerce")
    ok = ids.notna() & qi.notna()
    print(f"QI ({nom_cible}) : {ok.sum()} bébés avec une valeur, "
          f"{ids.notna().sum() - ok.sum()} sans valeur")
    return dict(zip(ids[ok].astype(int), qi[ok].astype(float)))


# ---------------------------------------------------------------------------
# Modèle
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
    """Une fenêtre de 30 s (1 x 6000) -> N_FEATURES_CNN caractéristiques."""

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


class Agregateur(nn.Module):
    """Suite des fenêtres (1, n, 128) -> un vecteur (1, 64)."""

    def __init__(self, type_seq=MODELE_SEQUENCE, d_in=N_FEATURES_CNN, d_out=N_FEATURES_SEQ):
        super().__init__()
        self.type = type_seq
        if type_seq == "lstm":
            self.rnn = nn.LSTM(d_in, d_out, batch_first=True)
        elif type_seq == "gru":
            self.rnn = nn.GRU(d_in, d_out, batch_first=True)
        elif type_seq == "transformer":
            self.proj = nn.Linear(d_in, d_out)
            couche = nn.TransformerEncoderLayer(d_out, nhead=4, dim_feedforward=2 * d_out,
                                                dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(couche, num_layers=2)
        else:
            raise ValueError("MODELE_SEQUENCE doit être 'lstm', 'gru' ou 'transformer'")

    @staticmethod
    def position(n, d, appareil):
        pos = torch.arange(n, device=appareil).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=appareil) * (-np.log(10000.0) / d))
        pe = torch.zeros(n, d, device=appareil)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe

    def forward(self, x):
        if self.type == "lstm":
            _, (h, _) = self.rnn(x)
            return h[-1]
        if self.type == "gru":
            _, h = self.rnn(x)
            return h[-1]
        x = self.proj(x)
        x = x + self.position(x.shape[1], x.shape[2], x.device)
        return self.transformer(x).mean(dim=1)


class ModeleQI(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ResNet1D()
        self.sequence = Agregateur()
        self.mlp = nn.Sequential(nn.Linear(N_FEATURES_SEQ, 32), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(32, 1))

    def predire_segment(self, fenetres):
        """fenetres : (nb_fenêtres, 6000) -> ŷ_ij (un nombre)."""
        x = fenetres.unsqueeze(1)                     # (n, 1, 6000)
        caract = []
        for paquet in x.split(TAILLE_PAQUET_CNN):
            if self.training and torch.is_grad_enabled():
                # recalcule les activations au lieu de les stocker : économise la mémoire
                caract.append(checkpoint(self.cnn, paquet, use_reentrant=False))
            else:
                caract.append(self.cnn(paquet))
        x1 = torch.cat(caract).unsqueeze(0)           # (1, n, 128)
        x2 = self.sequence(x1)                        # (1, 64)
        return self.mlp(x2).squeeze()                 # ŷ_ij

    def forward(self, segments):
        """segments : liste des S segments d'un bébé -> (ŷ_i, [ŷ_ij])."""
        preds = torch.stack([self.predire_segment(s) for s in segments])
        return preds.mean(), preds                    # ŷ_i = 1/S Σ ŷ_ij


# ---------------------------------------------------------------------------
# Données
# ---------------------------------------------------------------------------
def charger_segments(chemins, augmenter=False):
    segs = []
    for c in chemins:
        x = torch.from_numpy(np.load(c).astype(np.float32))
        if augmenter:
            x = x * torch.empty(x.shape[0], 1).uniform_(0.9, 1.1)   # amplitude
            x = x + 0.05 * torch.randn_like(x)                      # bruit
        segs.append(x.to(APPAREIL))
    return segs


# ---------------------------------------------------------------------------
# Entraînement / prédiction
# ---------------------------------------------------------------------------
def predire(modele, ids, segments_par_bebe, moy, et):
    modele.eval()
    res = {}
    with torch.no_grad(), torch.autocast(APPAREIL.type, dtype=torch.float16, enabled=AMP):
        for i in ids:
            y_chap, y_segs = modele(charger_segments(segments_par_bebe[i]))
            res[i] = (float(y_chap) * et + moy, (y_segs.float().cpu().numpy() * et + moy).tolist())
    return res


def entrainer(ids_train, ids_val, segments_par_bebe, qi):
    y_train = np.array([qi[i] for i in ids_train])
    moy, et = y_train.mean(), y_train.std() + 1e-8     # QI standardisé pour l'entraînement

    modele = ModeleQI().to(APPAREIL)
    if POIDS_PRE_ENTRAINES:
        poids = torch.load(POIDS_PRE_ENTRAINES, map_location=APPAREIL)
        manquants = modele.cnn.load_state_dict(poids, strict=False)
        print(f"    Poids pré-entraînés chargés ({manquants})")
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(APPAREIL.type, enabled=AMP)

    meilleur, meilleur_etat, attente = np.inf, None, 0
    for epoch in range(1, EPOCHS_MAX + 1):
        modele.train()
        ordre = list(ids_train)
        random.shuffle(ordre)
        opt.zero_grad()
        perte_totale = 0.0
        for k, i in enumerate(ordre):
            segs = charger_segments(segments_par_bebe[i], augmenter=AUGMENTATION)
            cible = torch.tensor((qi[i] - moy) / et, device=APPAREIL)
            with torch.autocast(APPAREIL.type, dtype=torch.float16, enabled=AMP):
                y_chap, _ = modele(segs)
            perte = (y_chap.float() - cible) ** 2      # (ŷ_i - y_i)²
            scaler.scale(perte / BEBES_PAR_PAS).backward()
            perte_totale += perte.item()
            if (k + 1) % BEBES_PAR_PAS == 0 or k + 1 == len(ordre):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

        preds_val = predire(modele, ids_val, segments_par_bebe, moy, et)
        mae_val = np.mean([abs(preds_val[i][0] - qi[i]) for i in ids_val])
        print(f"    epoch {epoch:3d}  perte entr. {perte_totale / len(ordre):.3f}  "
              f"MAE validation {mae_val:.2f}")
        if mae_val < meilleur - 1e-3:
            meilleur, meilleur_etat, attente = mae_val, copy.deepcopy(modele.state_dict()), 0
        else:
            attente += 1
            if attente >= PATIENCE:
                print(f"    arrêt précoce (meilleure MAE validation : {meilleur:.2f})")
                break
    modele.load_state_dict(meilleur_etat)
    return modele, moy, et


def metriques(y, p, nom):
    y, p = np.asarray(y), np.asarray(p)
    mae = np.mean(np.abs(y - p))
    rmse = np.sqrt(np.mean((y - p) ** 2))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rho = spearmanr(y, p).correlation if np.std(p) > 0 else np.nan
    pc15 = 100 * np.mean(np.abs(y - p) <= 15)
    print(f"  {nom:22s} MAE {mae:6.2f} | RMSE {rmse:6.2f} | R² {r2:6.2f} | "
          f"Spearman {rho:5.2f} | à ±15 pts : {pc15:5.1f} %")


def main():
    fixer_graine(GRAINE)
    print(f"Calcul sur : {APPAREIL}")

    index = pd.read_csv(Path(DOSSIER_SEGMENTS) / "index.csv")
    segments_par_bebe = index.groupby("id")["chemin"].apply(list).to_dict()
    qi = charger_qi()
    ids = np.array(sorted(set(segments_par_bebe) & set(qi)))
    sans_qi = sorted(set(segments_par_bebe) - set(qi))
    print(f"{len(ids)} bébés avec ECG et QI"
          + (f" ({len(sans_qi)} ECG sans QI ignorés : {sans_qi})" if sans_qi else ""))
    if len(ids) < 4:
        raise ValueError("Trop peu de bébés pour une validation croisée.")

    resultats = []
    plis = KFold(n_splits=min(N_PLIS, len(ids)), shuffle=True, random_state=GRAINE)
    debut = time.time()
    for num, (i_train, i_test) in enumerate(plis.split(ids), start=1):
        ids_train, ids_test = ids[i_train], ids[i_test]
        ids_train, ids_val = train_test_split(
            ids_train, test_size=max(1, round(PART_VALIDATION * len(ids_train))),
            random_state=GRAINE)
        print(f"\nPli {num} : {len(ids_train)} entraînement, {len(ids_val)} validation, "
              f"{len(ids_test)} test")
        modele, moy, et = entrainer(ids_train, ids_val, segments_par_bebe, qi)
        Path("modeles_dl").mkdir(exist_ok=True)
        torch.save(modele.state_dict(), f"modeles_dl/pli_{num}.pt")

        preds = predire(modele, ids_test, segments_par_bebe, moy, et)
        for i in ids_test:
            resultats.append({"id": int(i), "pli": num, "QI_reel": qi[i],
                              "QI_predit": preds[i][0],
                              "QI_reference": moy,   # prédire la moyenne d'entraînement
                              "QI_par_segment": "; ".join(f"{v:.1f}" for v in preds[i][1])})

    res = pd.DataFrame(resultats).sort_values("id")
    res.to_csv("predictions_dl.csv", index=False)
    print(f"\nTerminé en {(time.time() - debut) / 60:.1f} min. "
          f"Prédictions enregistrées dans predictions_dl.csv\n")
    print("Résultats (validation croisée par bébé) :")
    metriques(res["QI_reel"], res["QI_predit"], f"Modèle ({MODELE_SEQUENCE})")
    metriques(res["QI_reel"], res["QI_reference"], "Référence (moyenne)")

    # Plage de QI à 90 % (prédiction conforme à partir des erreurs hors entraînement)
    err = np.abs(res["QI_reel"] - res["QI_predit"]).values
    n = len(err)
    niveau = min(1.0, np.ceil((n + 1) * 0.9) / n)
    q = np.quantile(err, niveau, method="higher")
    print(f"\nPlage de prédiction à 90 % : QI prédit ± {q:.1f} points")


if __name__ == "__main__":
    main()
