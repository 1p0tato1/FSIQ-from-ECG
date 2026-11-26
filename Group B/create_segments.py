"""
Pipeline deep learning - ÉTAPE 1 : préparation des ECG.

Pour chaque bébé :
    ECG brut -> filtrage -> segments de durée T -> fenêtres de 30 s
avec T = durée du PLUS COURT enregistrement (calculée automatiquement).
Chaque segment est enregistré en .npy de forme (nb_fenêtres, 6000).

Installation (une seule fois) :  pip install numpy scipy pandas h5py
Lancement :                      python create_segments.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, resample_poly, sosfiltfilt

from pathlib import Path
BASE = Path(__file__).resolve().parent

# ============ PARAMÈTRES À MODIFIER ============
DOSSIER_MAT = BASE.parent / "data" / "M0-anonyme"            # dossier contenant les fichiers .mat (sous-dossiers inclus)
DOSSIER_SORTIE = BASE.parent / "data" / "segments_ecg"     # où enregistrer les segments préparés
FE = 200                            # fréquence d'échantillonnage voulue (Hz)
DUREE_FENETRE_S = 30                # durée d'une fenêtre (s)
FILTRE_HZ = (0.5, 40)               # filtre passe-bande (retire dérive et bruit)

# Enregistrements plus courts que ça : exclus (sinon ils imposent des segments
# minuscules à tous les bébés). Mettez 0 pour tout garder.
DUREE_MIN_ACCEPTEE_H = 2
# Pour forcer une durée de segment au lieu du minimum automatique (en heures) : ex. 3
DUREE_SEGMENT_FORCEE_H = None
# ===============================================


def numero_bebe(fichier):
    """'001-GAYlo-M0.mat' -> 1 (doit correspondre à la colonne Numéro des CSV)."""
    m = re.match(r"(\d+)", fichier.name)
    return int(m.group(1)) if m else None


def infos_sans_charger(chemin):
    """Renvoie (nb d'échantillons, Fe) sans charger tout l'ECG en mémoire."""
    try:
        tailles = {nom: forme for nom, forme, _ in scipy.io.whosmat(chemin)}
        n = int(np.prod(tailles["ECG"]))
        fe = scipy.io.loadmat(chemin, variable_names=["Fe"], squeeze_me=True)["Fe"]
    except (NotImplementedError, ValueError):
        import h5py  # fichiers MATLAB -v7.3
        with h5py.File(chemin, "r") as f:
            n = int(np.prod(f["ECG"].shape))
            fe = np.array(f["Fe"])
    return n, int(np.asarray(fe).ravel()[0])


def charger_ecg(chemin):
    try:
        ecg = scipy.io.loadmat(chemin, variable_names=["ECG"], squeeze_me=True)["ECG"]
    except NotImplementedError:
        import h5py
        with h5py.File(chemin, "r") as f:
            ecg = np.array(f["ECG"])
    return np.asarray(ecg, dtype=np.float64).ravel()


def pretraiter(ecg, fe):
    ecg = np.nan_to_num(ecg)
    if fe != FE:
        ecg = resample_poly(ecg, FE, fe)
    sos = butter(3, FILTRE_HZ, btype="band", fs=FE, output="sos")
    ecg = sosfiltfilt(sos, ecg)
    # Normalisation robuste par enregistrement (médiane / MAD), valeurs extrêmes écrêtées
    med = np.median(ecg)
    mad = 1.4826 * np.median(np.abs(ecg - med)) + 1e-8
    return np.clip((ecg - med) / mad, -20, 20).astype(np.float32)


def main():
    fichiers = sorted(p for p in Path(DOSSIER_MAT).rglob("*.mat") if not p.name.startswith("._"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier .mat dans '{DOSSIER_MAT}'")

    # --- 1. Durée de chaque enregistrement ---
    print("Durée des enregistrements :")
    infos = []
    for p in fichiers:
        n, fe = infos_sans_charger(p)
        duree_h = n / fe / 3600
        infos.append({"fichier": p, "id": numero_bebe(p), "fe": fe, "duree_h": duree_h})
        print(f"  {p.name:35s} {duree_h:6.2f} h  (Fe = {fe} Hz)")
    infos = pd.DataFrame(infos)

    exclus = infos[infos["duree_h"] < DUREE_MIN_ACCEPTEE_H]
    if len(exclus):
        print(f"\nExclus car < {DUREE_MIN_ACCEPTEE_H} h : {', '.join(p.name for p in exclus['fichier'])}")
    infos = infos[infos["duree_h"] >= DUREE_MIN_ACCEPTEE_H]
    sans_id = infos[infos["id"].isna()]
    if len(sans_id):
        print(f"Ignorés (pas de numéro au début du nom) : {', '.join(p.name for p in sans_id['fichier'])}")
        infos = infos[infos["id"].notna()]

    # --- 2. Durée des segments = plus court enregistrement ---
    duree_seg_h = DUREE_SEGMENT_FORCEE_H or infos["duree_h"].min()
    fen_par_seg = int(duree_seg_h * 3600 // DUREE_FENETRE_S)
    long_fen = FE * DUREE_FENETRE_S
    print(f"\nDurée des segments : {fen_par_seg * DUREE_FENETRE_S / 3600:.2f} h "
          f"-> chaque segment = {fen_par_seg} fenêtres x {long_fen} points")

    # --- 3. Découpage et enregistrement ---
    sortie = Path(DOSSIER_SORTIE)
    sortie.mkdir(exist_ok=True)
    index = []
    for _, ligne in infos.iterrows():
        ecg = pretraiter(charger_ecg(ligne["fichier"]), ligne["fe"])
        n_fen = len(ecg) // long_fen
        fenetres = ecg[: n_fen * long_fen].reshape(n_fen, long_fen)
        n_seg = n_fen // fen_par_seg          # seuls les segments complets sont gardés
        for j in range(n_seg):
            seg = fenetres[j * fen_par_seg:(j + 1) * fen_par_seg]
            chemin = sortie / f"{int(ligne['id']):03d}_seg{j + 1}.npy"
            np.save(chemin, seg.astype(np.float16))
            index.append({"id": int(ligne["id"]), "fichier": ligne["fichier"].name,
                          "segment": j + 1, "chemin": str(chemin)})
        print(f"  {ligne['fichier'].name} : {n_seg} segment(s)")

    index = pd.DataFrame(index)
    index.to_csv(sortie / "index.csv", index=False)
    print(f"\n{index['id'].nunique()} bébés, {len(index)} segments enregistrés dans '{sortie}'")
    if index["id"].duplicated().any() and index.groupby("id")["fichier"].nunique().max() > 1:
        print("Attention : certains bébés ont plusieurs fichiers .mat (M0, M1...) : "
              "leurs segments sont tous rattachés au même bébé.")


if __name__ == "__main__":
    main()
