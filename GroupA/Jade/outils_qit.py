"""Outils de 01_prediction_qit.ipynb (et de sa version anglaise 01_prediction_qit_en.ipynb) : lecture, nettoyage,
modèles, validation croisée, graphiques.

Le notebook montre les étapes et les résultats ; ce fichier contient le « comment ». Les sections suivent celles du
notebook. Tout ce qui apprend quelque chose (imputation, filtre, sélection, réglage, modèle) est dans un Pipeline
ajusté dans chaque pli, sur les seuls enfants d'entraînement.

Langue : choisir_langue("en") passe les messages, tableaux et graphiques en anglais. Les calculs sont exactement les
mêmes dans les deux langues. Les noms anglais des fonctions (pour le notebook anglais) sont en fin de fichier.

Après une modification de ce fichier : redémarrer le noyau du notebook.
"""
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from scipy.stats import norm, rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from evaluate_model import bootstrap, norm_id, run_cv

# ===================================================================================== Réglages fixes
# [À ADAPTER] pour d'autres fichiers : noms des colonnes (comparés sans accents, majuscules ni espaces).
COL_ID_HRV, COL_ID_WPPSI, COL_PASSAGE = "Numéro", "N°", "Passage"
COLS_SCORES = {e: f"Note composite | {e}" for e in ["QIV", "QIP", "QIT", "NCL"]}
COLS_CLINIQUES = ["Sexe", "Prématuré", "SFA", "Réa à la naissance", "Pathologie neurologique",
                  "Tabac", "Diabète", "HTA", "Maturation Pulm", "Dépression: prise d'IRS"]
# Jamais en variables d'entrée : identité, dates, horaires, durées, nombres de RR, scores WPPSI.
MOTIFS_EXCLUS = [r"numero", r"^n°", r"monogramme", r"ddn", r"filename", r"recorder", r"date", r"duration",
                 r"beginning", r"\bend\b", r"nb ?rr", r"analy[sz]ed", r"passage", r"sortie",
                 r"^(qit|qiv|qip|ncl|com|inf|dim|cub|aob)\b"]
MAX_MANQUANTS_COL, MAX_MANQUANTS_LIGNE = 0.30, 0.50   # variable / enfant retiré au-delà de 30 % / 50 % de manquants
SEUIL_QUASI_CONSTANTE = 0.95      # variable retirée si une même valeur concerne 95 % des enfants
MIN_CAS_RARES = 5                 # covariable binaire retirée si une classe a moins de 5 enfants
SEUIL_CORR = 0.95                 # une seule variable gardée par groupe corrélé au-delà de ce seuil
TOP_K = 30                        # variables gardées par la sélection « sél. RF »
VARIANCE_PCA = 0.90               # part de variance gardée par la sélection « PCA 90 % »
K_INTERNE = 5                     # plis internes pour régler les modèles

# ===================================================================================== Langue
LANGUE = "fr"
REFERENCE = "score moyen (référence)"
_EN = {   # noms des colonnes de résultats (et oui/non) en anglais
    "enfants": "children", "MAE moyenne": "MAE of the mean", "MAE modèle": "model MAE",
    "IC bas": "CI low", "IC haut": "CI high", "R² planté": "planted R²", "détecté": "detected",
    "AUC IC bas": "AUC CI low", "AUC IC haut": "AUC CI high", "sensibilité": "sensitivity",
    "spécificité": "specificity", "exactitude": "accuracy", "exactitude équilibrée": "balanced accuracy",
    "indice": "index", "ρ ajusté": "adjusted ρ", "IC bas ajusté": "adjusted CI low",
    "IC haut ajusté": "adjusted CI high", "variance max": "max variance", "variance max ajusté": "adjusted max variance",
    "démontré": "demonstrated", "démontré ajusté": "adjusted demonstrated", "vrai": "true", "moyenne": "mean",
    "oui": "yes", "non": "no",
}


def choisir_langue(code):
    """'fr' (par défaut) ou 'en' : langue des messages, tableaux et graphiques. Les calculs ne changent pas."""
    global LANGUE, REFERENCE
    if code not in ("fr", "en"):
        raise ValueError("langue / language : 'fr' ou / or 'en'")
    LANGUE = code
    REFERENCE = _t("score moyen (référence)", "mean score (baseline)")


def _t(fr, en):
    return en if LANGUE == "en" else fr


def _k(cle):
    """Nom d'une colonne de résultat dans la langue choisie."""
    return _EN.get(cle, cle) if LANGUE == "en" else cle


def _traduire(objet):
    """Résultat (dict ou DataFrame) avec ses noms de colonnes, et oui/non, dans la langue choisie."""
    if LANGUE != "en":
        return objet
    if isinstance(objet, dict):
        return {_k(k): (_k(v) if isinstance(v, str) else v) for k, v in objet.items()}
    t = objet.rename(columns=_k)
    if t.index.name:
        t = t.rename_axis(_k(t.index.name))
    return t.replace({"oui": "yes", "non": "no"})


# ===================================================================================== 1.1 Lecture
def normaliser_texte(s):
    """Minuscules, sans accents, espaces simplifiés : ' Numéro ' -> 'numero'."""
    s = unicodedata.normalize("NFKD", str(s))
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip().lower()


def _cle(s):
    return re.sub(r"[\s_.\-]", "", normaliser_texte(s))


def _meme_nom(colonne, nom):
    """Compare deux noms ; un caractère illisible « � » dans le nom de colonne vaut n'importe quel caractère."""
    motif = "".join("." if ch == "�" else re.escape(ch) for ch in _cle(colonne))
    return re.fullmatch(motif, _cle(nom)) is not None


def trouver_colonne(df, nom):
    """Vrai nom de la colonne qui correspond à `nom` (ou à la fin d'un nom « Bloc | Nom »), sinon None."""
    exactes = [c for c in df.columns if _meme_nom(c, nom)]
    fins = [c for c in df.columns if _meme_nom(str(c).split("|")[-1], nom)]
    return exactes[0] if exactes else fins[0] if len(fins) == 1 else None


UNITES = {"-", "n", "ms", "ms^2", "ms/beat", "bpm", "%", "hz", "hh:mm:ss", "0/1", "m/f", "unit"}
VALEURS_VIDES = {"", "-", "--", "na", "n/a", "nan", "?", "#n/a", "#div/0!", "#valeur!", "#value!"}


def lire_csv(chemin, lignes_entete=1):
    """Export Excel -> DataFrame : encodage et séparateur détectés ; deux lignes d'en-tête réunies en « Bloc | Nom » ;
    ligne d'unités ignorée ; cases vides (« », « - », « NaN »…) en valeurs manquantes ; nombres à virgule convertis."""
    brut = Path(chemin).read_bytes()
    try:
        encodage = "utf-8-sig"
        brut.decode(encodage)
    except UnicodeDecodeError:
        encodage = "cp1252"
    sep = max(["\t", ";", ","], key=brut.decode(encodage, errors="replace").splitlines()[0].count)
    df = pd.read_csv(chemin, sep=sep, encoding=encodage, header=list(range(lignes_entete)),
                     decimal="," if sep != "," else ".")
    if lignes_entete > 1:          # une cellule fusionnée d'Excel n'est remplie que sur sa première colonne
        noms, bloc = [], ""
        for haut, bas in df.columns:
            haut, bas = ("" if str(x).startswith("Unnamed") else str(x).strip() for x in (haut, bas))
            bloc = haut or bloc
            noms.append(f"{bloc} | {bas}" if bloc and bas else bloc or bas)
        df.columns = noms
    noms, vus = [], {}
    for c in df.columns:           # colonnes sans nom (tabulations en fin de ligne) ou en double
        nom = str(c).strip() or "sans_nom"
        vus[nom] = vus.get(nom, 0) + 1
        noms.append(nom if vus[nom] == 1 else f"{nom}.{vus[nom] - 1}")
    df.columns = noms
    valeurs = [normaliser_texte(v) for v in df.iloc[0] if pd.notna(v) and str(v).strip()]
    unites = bool(valeurs) and sum(v in UNITES for v in valeurs) >= 0.5 * len(valeurs)
    if unites:
        df = df.iloc[1:].reset_index(drop=True)
    for c in df.columns:           # en nombres si au moins 90 % des cases remplies s'y prêtent
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
            texte = df[c].astype("string").str.replace(r"[\s %]", "", regex=True).str.replace(",", ".")
            texte = texte.where(~texte.str.lower().isin(VALEURS_VIDES).fillna(False).astype(bool))
            nombres = pd.to_numeric(texte, errors="coerce").astype("float64")
            if nombres.notna().sum() >= 0.9 * texte.notna().sum():
                df[c] = nombres
    print(_t(f"{Path(chemin).name} : {df.shape[0]} lignes, {df.shape[1]} colonnes"
             + (", ligne d'unités ignorée" if unites else ""),
             f"{Path(chemin).name}: {df.shape[0]} rows, {df.shape[1]} columns" + (", units row skipped" if unites else "")))
    return df.copy()               # copie d'un seul bloc (plus rapide après les conversions colonne par colonne)


def _avec_id(df, nom_colonne, fichier):
    """Ajoute id_enfant (numéro harmonisé comme dans evaluate_model : « 012 », « 12.0 » -> « 12 ») et retire l'original."""
    col = trouver_colonne(df, nom_colonne)
    if col is None:
        raise KeyError(_t(f"{fichier} : colonne identifiant « {nom_colonne} » introuvable.",
                          f"{fichier}: ID column '{nom_colonne}' not found."))
    return df.assign(id_enfant=df[col].map(norm_id)).drop(columns=col).dropna(subset=["id_enfant"])


@dataclass
class Sources:
    lin: pd.DataFrame        # HRV linéaire, tous les passages
    nl: pd.DataFrame         # HRV non linéaire, tous les passages
    scores: pd.DataFrame     # une ligne par enfant : QIV, QIP, QIT, NCL complets
    wppsi: pd.DataFrame      # une ligne par enfant : le reste du fichier WPPSI (date de naissance…)


def charger(dossier, fichiers, lignes_entete_wppsi=2):
    """Lit les 3 fichiers et garde, pour le WPPSI, les enfants qui ont tous leurs scores (comme pulisci_righe.py)."""
    lin = _avec_id(lire_csv(dossier / fichiers["lin"]), COL_ID_HRV, fichiers["lin"])
    nl = _avec_id(lire_csv(dossier / fichiers["nl"]), COL_ID_HRV, fichiers["nl"])
    wppsi = _avec_id(lire_csv(dossier / fichiers["wppsi"], lignes_entete_wppsi), COL_ID_WPPSI, fichiers["wppsi"])
    n_communs = len(set(lin.id_enfant) & set(nl.id_enfant) & set(wppsi.id_enfant))
    print(_t(f"{n_communs} sujets présents dans les 3 fichiers", f"{n_communs} subjects present in all 3 files"))

    cols = {e: trouver_colonne(wppsi, nom) for e, nom in COLS_SCORES.items()}
    cols = {e: c for e, c in cols.items() if c is not None and pd.api.types.is_numeric_dtype(wppsi[c])}
    if "QIT" not in cols:
        raise ValueError(_t("QIT introuvable : vérifier COLS_SCORES dans outils_qit.py.",
                            "QIT not found: check COLS_SCORES in outils_qit.py."))
    scores = wppsi[["id_enfant", *cols.values()]].rename(columns={c: e for e, c in cols.items()})
    n_qit = scores.dropna(subset=["QIT"]).groupby("id_enfant")["QIT"].nunique()     # deux QIT différents : retiré
    scores = scores[~scores.id_enfant.isin(n_qit.index[n_qit > 1])].groupby("id_enfant").first()
    complets = scores.notna().all(axis=1) & scores.apply(lambda s: s.between(40, 160)).all(axis=1)
    print(_t(f"{(~complets).sum()} sujets sans tous leurs scores retirés ; {complets.sum()} enfants avec leurs scores",
             f"{(~complets).sum()} subjects without all their scores removed; {complets.sum()} children with scores"))
    return Sources(lin, nl, scores[complets], wppsi.groupby("id_enfant").first())


# ===================================================================================== 1.2 Un jeu par passage
def normaliser_passage(valeur):
    """'MO' (lettre O), 'm0', 'M00' -> 'M0' ; 'M06' -> 'M6'."""
    s = re.sub(r"\s", "", str(valeur)).upper()
    return re.sub(r"^M0+(?=\d)", "M", "M0" if s == "MO" else s)


def _par_periode(colonnes):
    """Nombre de colonnes par période : 'SDNN (TDNight)' -> Night, 'Higuchi (24h)' -> 24h, sinon autre."""
    periodes = [(m.group(1) if (m := re.search(r"(24h|Day|Night)\)$", c.split("|", 1)[-1])) else _t("autre", "other"))
                for c in colonnes]
    return pd.Series(periodes, dtype=object).value_counts().to_dict()


def _indices_hrv(brut, bloc, passage, bavard):
    """Un enregistrement par enfant à ce passage ; sépare les indices HRV (nombres) du reste (méta-données)."""
    df = brut[brut[trouver_colonne(brut, COL_PASSAGE)].map(normaliser_passage) == normaliser_passage(passage)]
    df = df.drop_duplicates("id_enfant").set_index("id_enfant")
    cliniques = {trouver_colonne(df, n) for n in COLS_CLINIQUES} - {None}
    exclues = [c for c in df.columns if c in cliniques or any(re.search(m, normaliser_texte(c)) for m in MOTIFS_EXCLUS)]
    vides = [c for c in df.columns if c not in exclues and df[c].isna().all()]
    indices = [c for c in df.columns if c not in exclues + vides and pd.api.types.is_numeric_dtype(df[c])]
    if bavard:
        print(_t(f"[{bloc}] {len(df)} enregistrements à {passage} : {len(indices)} indices HRV {_par_periode(indices)}, "
                 f"{len(vides)} colonnes vides à ce passage {_par_periode(vides)}",
                 f"[{bloc}] {len(df)} recordings at {passage}: {len(indices)} HRV indices {_par_periode(indices)}, "
                 f"{len(vides)} empty columns at this visit {_par_periode(vides)}"))
    return df[indices].add_prefix(f"{bloc}|"), df.drop(columns=indices)


def _date_naissance(source, index):
    col = trouver_colonne(source, "DDN")
    if col is None:
        return None
    return pd.to_datetime(source[col].astype("string").str.strip(), dayfirst=True, errors="coerce",
                          format="mixed").reindex(index)


def construire_donnees(sources, passage, bavard=False):
    """Indices HRV (lin|, nl|), covariables cliniques (clin|) et scores des enfants enregistrés à ce passage.
    Aucune étape ne regarde le QI : rien n'apprend sur la cible, on peut donc le faire avant les plis."""
    hrv_lin, meta_lin = _indices_hrv(sources.lin, "lin", passage, bavard)
    hrv_nl, meta_nl = _indices_hrv(sources.nl, "nl", passage, bavard)
    donnees = hrv_lin.join(hrv_nl, how="inner").join(sources.scores, how="inner")
    n_fusion = len(donnees)

    # même enfant dans les 3 fichiers : la date de naissance doit être la même partout
    ddn = pd.DataFrame({nom: d for nom, s in [("lin", meta_lin), ("nl", meta_nl), ("wppsi", sources.wppsi)]
                        if (d := _date_naissance(s, donnees.index)) is not None})
    ddn_differentes = (ddn.notna().sum(axis=1) >= 2) & (ddn.nunique(axis=1) > 1)
    donnees = donnees[~ddn_differentes]

    # covariables cliniques : texte à deux valeurs codé 0/1 (Sexe), classes trop rares retirées
    clinique, codes = pd.DataFrame(index=donnees.index), {}
    for nom in COLS_CLINIQUES:
        source = next((s for s in [meta_lin, meta_nl] if trouver_colonne(s, nom) is not None), None)
        if source is None:
            continue
        valeurs = source[trouver_colonne(source, nom)].reindex(donnees.index)
        if not pd.api.types.is_numeric_dtype(valeurs):
            valeurs = valeurs.astype("string").str.strip()
            niveaux = sorted(valeurs.dropna().unique())
            codes[nom] = {v: i for i, v in enumerate(niveaux)} if len(niveaux) == 2 else None
            valeurs = valeurs.map(codes[nom]).astype(float) if codes[nom] else pd.Series(np.nan, index=valeurs.index)
        clinique[f"clin|{nom}"] = valeurs
    effectifs = clinique.apply(lambda s: s.value_counts().min() if s.nunique() == 2 else 0 if s.nunique() < 2 else 99)
    rares = list(effectifs.index[effectifs < MIN_CAS_RARES])
    donnees = donnees.join(clinique.drop(columns=rares))

    # indices HRV : trop vides ou quasi constants retirés, puis enfants trop incomplets
    cols_hrv = list(hrv_lin.columns) + list(hrv_nl.columns)
    donnees[cols_hrv] = donnees[cols_hrv].replace([np.inf, -np.inf], np.nan)
    manquant = donnees[cols_hrv].isna().mean()
    dominant = donnees[cols_hrv].apply(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 1.0)
    trop_vides = manquant.index[manquant > MAX_MANQUANTS_COL]
    quasi_constantes = dominant.index[dominant >= SEUIL_QUASI_CONSTANTE].difference(trop_vides)
    donnees = donnees.drop(columns=trop_vides.union(quasi_constantes))
    restants = [c for c in cols_hrv if c in donnees.columns]
    incomplets = donnees[restants].isna().mean(axis=1) > MAX_MANQUANTS_LIGNE
    donnees = donnees[~incomplets]

    if bavard:
        print(_t(f"{n_fusion} enfants présents dans les 3 fichiers à {passage} ; retirés : {ddn_differentes.sum()} "
                 f"(date de naissance différente), {incomplets.sum()} (trop incomplets) -> {len(donnees)} enfants",
                 f"{n_fusion} children present in all 3 files at {passage}; removed: {ddn_differentes.sum()} "
                 f"(different birth date), {incomplets.sum()} (too incomplete) -> {len(donnees)} children"))
        print(_t(f"Covariables cliniques : {clinique.shape[1] - len(rares)} gardées, trop rares {[r[5:] for r in rares]}"
                 + "".join(f" ; {n} codé {c}" for n, c in codes.items() if c),
                 f"Clinical covariates: {clinique.shape[1] - len(rares)} kept, too rare {[r[5:] for r in rares]}"
                 + "".join(f"; {n} coded {c}" for n, c in codes.items() if c)))
        print(_t(f"Variables HRV : {len(trop_vides)} trop vides {_par_periode(trop_vides)}, {len(quasi_constantes)} quasi "
                 f"constantes, {len(restants)} gardées {_par_periode(restants)}",
                 f"HRV variables: {len(trop_vides)} too empty {_par_periode(trop_vides)}, {len(quasi_constantes)} "
                 f"near-constant, {len(restants)} kept {_par_periode(restants)}"))
    return donnees


# ===================================================================================== 1.3 Cohorte et plis
def creer_plis(donnees, fichier_cohorte, fichier_plis, t_repetitions, k_plis, graine):
    """cohort.csv (enfants, scores) et folds.csv (t × k plis stratifiés sur le QIT) au format de evaluate_model.py.
    Mêmes données et même graine -> mêmes plis à chaque lancement."""
    Path(fichier_cohorte).parent.mkdir(parents=True, exist_ok=True)
    scores = [c for c in COLS_SCORES if c in donnees.columns]
    donnees[scores].rename_axis("child_id").reset_index().to_csv(fichier_cohorte, index=False)
    tranches = pd.qcut(donnees["QIT"], 5, labels=False, duplicates="drop")
    plis = [pd.DataFrame({"child_id": donnees.index[te], "repeat": r, "fold": k})
            for r in range(1, t_repetitions + 1)
            for k, (_, te) in enumerate(StratifiedKFold(k_plis, shuffle=True, random_state=graine + r)
                                        .split(donnees, tranches), start=1)]
    pd.concat(plis).to_csv(fichier_plis, index=False)


def plis_communs(fichier_plis, repetitions):
    """(répétition, pli, enfants d'entraînement, enfants de test) de folds.csv, dans l'ordre de run_cv."""
    f = pd.read_csv(fichier_plis, dtype={"child_id": str}).assign(child_id=lambda d: d.child_id.map(norm_id))
    f = f[f["repeat"].isin(repetitions)]
    return [(r, k, g.loc[g.fold != k, "child_id"].tolist(), g.loc[g.fold == k, "child_id"].tolist())
            for r, g in f.groupby("repeat") for k in sorted(g["fold"].unique())]


def plis_stratifies(y, n_rep, k_plis=5, graine=42):
    """Plis propres à une analyse (autre groupe d'enfants ou autre cible), stratifiés : sur la classe si y est
    binaire, sur les quintiles sinon."""
    strates = y if y.nunique() <= 2 else pd.qcut(y, 5, labels=False, duplicates="drop")
    return [(r, k, list(y.index[tr]), list(y.index[te])) for r in range(1, n_rep + 1)
            for k, (tr, te) in enumerate(StratifiedKFold(k_plis, shuffle=True, random_state=graine + r)
                                         .split(y, strates), start=1)]


# ===================================================================================== 1.4 Signal univarié
def correlation_contre_hasard(X, y, n_melanges=50):
    """|Spearman| de chaque variable avec y, et maximum obtenu en mélangeant y (moyenne, 95e centile)."""
    rho = X.corrwith(y, method="spearman").abs().sort_values(ascending=False)
    tirage = np.random.default_rng(0)
    maxima = [X.corrwith(pd.Series(tirage.permutation(y.to_numpy()), index=y.index), method="spearman").abs().max()
              for _ in range(n_melanges)]
    return rho, np.mean(maxima), np.quantile(maxima, 0.95)


# ===================================================================================== 2. Feature selection
class FiltreCorrelation(BaseEstimator, TransformerMixin):
    """Garde une variable par groupe de variables très corrélées (|Spearman| > seuil), ajusté dans chaque pli."""

    def __init__(self, seuil=SEUIL_CORR):
        self.seuil = seuil

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.nan_to_num(np.corrcoef(rankdata(X, axis=0), rowvar=False))) if X.shape[1] > 1 else None
        garder = []
        for j in range(X.shape[1]):
            if not garder or corr[j, garder].max() <= self.seuil:
                garder.append(j)
        self.garder_ = np.array(garder)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.garder_]

    def get_feature_names_out(self, input_features=None):
        noms = input_features if input_features is not None else [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(noms, dtype=object)[self.garder_]


def faire_preprocesseur(colonnes):
    """HRV : médiane, filtre de corrélation, transformation vers une loi normale. Clinique : mode, mise à l'échelle."""
    hrv = [c for c in colonnes if not c.startswith("clin|")]
    clin = [c for c in colonnes if c.startswith("clin|")]
    etapes = []
    if hrv:
        etapes.append(("hrv", Pipeline([("imputation", SimpleImputer(strategy="median")),
                                        ("correlation", FiltreCorrelation()),
                                        ("loi_normale", QuantileTransformer(n_quantiles=50,
                                                                            output_distribution="normal"))]), hrv))
    if clin:
        etapes.append(("clinique", Pipeline([("imputation", SimpleImputer(strategy="most_frequent")),
                                             ("echelle", StandardScaler())]), clin))
    return ColumnTransformer(etapes, verbose_feature_names_out=False)


SELECTIONS = {
    "toutes": lambda: "passthrough",
    "sél. RF": lambda: SelectFromModel(RandomForestRegressor(n_estimators=200, max_features="sqrt", min_samples_leaf=5,
                                                             random_state=0, n_jobs=1),
                                       max_features=lambda X: min(TOP_K, X.shape[1]), threshold=-np.inf),
    "PCA 90 %": lambda: PCA(n_components=VARIANCE_PCA, svd_solver="full"),
}
# noms anglais des mêmes sélections (notebook anglais)
SELECTIONS.update({"all": SELECTIONS["toutes"], "RF selection": SELECTIONS["sél. RF"], "PCA 90%": SELECTIONS["PCA 90 %"]})
LOGISTIQUE = ("Logistique", "Logistic")

# ===================================================================================== 3. Models
ESPACES = {   # plan du mercredi
    "Random Forest": {"n_estimators": [200, 500], "max_depth": [3, 5, 7, None], "min_samples_leaf": [5, 10, 20],
                      "max_features": ["sqrt", "log2"]},
    "LightGBM": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, -1],
                 "min_child_samples": [10, 20], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0, 1, 10]},
    "XGBoost": {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7],
                "min_child_weight": [1, 5], "reg_alpha": [0, 0.1, 1], "reg_lambda": [1, 10], "subsample": [0.8, 1.0]},
    "SVR": {"C": [0.1, 1, 10, 100], "epsilon": [0.01, 0.1, 1], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]},
}
BASES = {     # n_jobs=1 : le parallélisme se fait sur les plis
    "Random Forest": lambda: RandomForestRegressor(random_state=0, n_jobs=1),
    "LightGBM": lambda: LGBMRegressor(colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=0,
                                      n_jobs=1, verbose=-1),
    "XGBoost": lambda: XGBRegressor(colsample_bytree=0.5, random_state=0, n_jobs=1),
    "SVR": lambda: SVR(),
}


def faire_modele(nom, n_iter):
    """Modèle réglé dans une boucle interne à 5 plis, sur les seuls enfants d'entraînement."""
    interne = KFold(K_INTERNE, shuffle=True, random_state=0)
    if nom == "Elastic Net":      # ElasticNetCV parcourt 30 pénalités × 5 mélanges L1/L2 d'un coup
        return ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], alphas=30, eps=1e-2, cv=interne, max_iter=10_000,
                            selection="random", random_state=0)
    return RandomizedSearchCV(BASES[nom](), ESPACES[nom], n_iter=n_iter, cv=interne,
                              scoring="neg_mean_absolute_error", random_state=0)


def _classification(modele):
    return modele in LOGISTIQUE if isinstance(modele, str) else is_classifier(modele)


def faire_pipeline(colonnes, modele="Elastic Net", selection="toutes", n_iter=12):
    """Préparation, sélection et modèle : un seul objet, ajusté en entier dans chaque pli.
    `modele` : un nom (Elastic Net, Random Forest, LightGBM, XGBoost, SVR, Logistique) ou n'importe quel modèle
    scikit-learn déjà réglé, par exemple Ridge(alpha=10) ; il est copié neuf dans chaque pli."""
    if not isinstance(modele, str):
        return Pipeline([("prep", faire_preprocesseur(colonnes)), ("selection", SELECTIONS[selection]()),
                         ("modele", clone(modele))])
    if modele in LOGISTIQUE:      # classification ; class_weight="balanced" : la classe rare compte autant
        return Pipeline([("prep", faire_preprocesseur(colonnes)),
                         ("modele", LogisticRegressionCV(Cs=10, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                                         scoring="roc_auc", class_weight="balanced", max_iter=5_000))])
    return Pipeline([("prep", faire_preprocesseur(colonnes)), ("selection", SELECTIONS[selection]()),
                     ("modele", faire_modele(modele, n_iter))])


# ===================================================================================== 4. Evaluation
def _un_pli(Xd, y, tr, te, modele, selection, n_iter):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = faire_pipeline(list(Xd.columns), modele, selection, n_iter).fit(Xd.loc[tr], y.loc[tr])
        return pipe.predict_proba(Xd.loc[te])[:, 1] if _classification(modele) else np.ravel(pipe.predict(Xd.loc[te]))


def predictions_croisees(Xd, y, plis, modele="Elastic Net", selection="toutes", n_iter=12, n_jobs=-1):
    """Prédictions hors pli : chaque enfant est prédit par un modèle qui ne l'a pas vu.
    y : une cible (Series) ou plusieurs (DataFrame, un modèle par cible). Renvoie, par cible, un tableau avec une
    ligne par répétition et une colonne par enfant. modele="Logistique" (ou un classifieur scikit-learn) :
    probabilité de la classe 1. `modele` peut être un nom ou un modèle scikit-learn (voir faire_pipeline)."""
    Y = y.to_frame() if isinstance(y, pd.Series) else y
    taches = [(c, r, tr, te) for c in Y.columns for r, _, tr, te in plis]
    sorties = Parallel(n_jobs=n_jobs)(delayed(_un_pli)(Xd, Y[c], tr, te, modele, selection, n_iter)
                                      for c, _, tr, te in taches)
    P = {c: pd.DataFrame(np.nan, index=sorted({p[0] for p in plis}), columns=Y.index) for c in Y.columns}
    for (c, r, _, te), s in zip(taches, sorties):
        P[c].loc[r, te] = s
    return next(iter(P.values())) if isinstance(y, pd.Series) else P


def ecrire_predictions(P, plis, fichier_plis, fichier_cohorte, fichier, repetitions):
    """Écrit les prédictions au format commun de l'équipe, en passant par run_cv de evaluate_model.py."""
    par_test = {tuple(te): {c: P[c].loc[r, te].to_numpy() for c in P} for r, _, _, te in plis}
    run_cv(lambda tr, te: par_test[tuple(te)], fichier_plis, fichier_cohorte, fichier, targets=list(P),
           repeats=list(repetitions), verbose=False)


def _mesures(vrai, pred):
    mse = mean_squared_error(vrai, pred)
    return {"MAE": mean_absolute_error(vrai, pred), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(vrai, pred)}


def mesures_par_pli(predictions, etiquettes, plis):
    """Une ligne par (modèle, cible, pli de test), référence comprise : MAE, MSE, RMSE, R² et gain sur la référence."""
    lignes = []
    for cible in etiquettes.columns:
        y = etiquettes[cible]
        for r, k, tr, te in plis:
            ref = _mesures(y[te], np.full(len(te), y[tr].mean()))
            lignes.append({"model": REFERENCE, "target": cible, "repeat": r, "fold": k, **ref, "gain": 0.0})
            for nom, P in predictions.items():
                m = _mesures(y[te], P[cible].loc[r, te])
                lignes.append({"model": nom, "target": cible, "repeat": r, "fold": k, **m, "gain": ref["MAE"] - m["MAE"]})
    return pd.DataFrame(lignes)


def comparaison_par_modele(par_pli):
    """Moyenne et écart-type sur les plis, colonnes du plan du jeudi (model_comparison.csv)."""
    t = par_pli.groupby(["model", "target"], sort=False)[["MAE", "MSE", "RMSE", "R2"]].agg(["mean", "std"])
    t.columns = [f"{m.lower()}_{s}" for m, s in t.columns]
    return t.reset_index()


def ic(bas, haut, chiffres=2):
    return _t(f"[{bas:.{chiffres}f} ; {haut:.{chiffres}f}]", f"[{bas:.{chiffres}f}, {haut:.{chiffres}f}]")


def tableau_resultats(comparaison, resume_equipe, cibles):
    """MAE et R² (moyenne ± écart-type sur les plis) et gain sur la moyenne avec son IC 95 % (evaluate_model)."""
    equipe = resume_equipe.replace({"mean_baseline": REFERENCE})
    t = comparaison.merge(equipe[["target", "model", "MAE_gain_vs_baseline", "gain_ci_low", "gain_ci_high"]],
                          on=["target", "model"], how="left")
    t = t.sort_values(["target", "mae_mean"], key=lambda s: s.map(cibles.index) if s.name == "target" else s)
    oui, non = _t("oui", "yes"), _t("non", "no")
    return pd.DataFrame({
        "MAE": [f"{m:.2f} ± {s:.2f}" for m, s in zip(t.mae_mean, t.mae_std)],
        "R²": [f"{m:.3f} ± {s:.3f}" for m, s in zip(t.r2_mean, t.r2_std)],
        _t("gain vs moyenne [IC 95 %]", "gain vs mean [95% CI]"):
            ["" if m == REFERENCE else f"{g:+.2f} {ic(a, b)}" for m, g, a, b
             in zip(t.model, t.MAE_gain_vs_baseline, t.gain_ci_low, t.gain_ci_high)],
        _t("gain démontré", "gain demonstrated"):
            ["" if m == REFERENCE else (oui if a > 0 else non) for m, a in zip(t.model, t.gain_ci_low)],
    }).set_index(pd.MultiIndex.from_frame(t[["target", "model"]], names=[_t("cible", "target"), _t("modèle", "model")]))


TEINTE_MIEUX = "#e3eefb"           # fond léger du côté « bon résultat » d'un seuil


def _cote_mieux(ax, seuil=0.0, axe="x"):
    """Teinte le côté « bon résultat » (au-delà de `seuil`) et trace le seuil ; laisse de la place de ce côté."""
    bas, haut = ax.get_xlim() if axe == "x" else ax.get_ylim()
    haut = max(haut, seuil + 0.3 * (haut - bas))
    if axe == "x":
        ax.set_xlim(bas, haut)
        ax.axvspan(seuil, haut, color=TEINTE_MIEUX, lw=0, zorder=0)
        ax.axvline(seuil, color="k", lw=1.2, zorder=1)
    else:
        ax.set_ylim(bas, haut)
        ax.axhspan(seuil, haut, color=TEINTE_MIEUX, lw=0, zorder=0)
        ax.axhline(seuil, color="k", lw=1.2, zorder=1)


def graphique_gains(par_pli, cibles, fichier):
    """Gain de MAE sur la moyenne et R², par modèle et par cible (moyenne ± écart-type sur les plis)."""
    modeles = [m for m in par_pli.model.unique() if m != REFERENCE]
    t = par_pli[par_pli.model != REFERENCE].groupby(["model", "target"])[["gain", "R2"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(modeles) * max(1, len(cibles) / 2) + 2.3), sharey=True)
    hauteur = 0.8 / len(cibles)
    for j, cible in enumerate(cibles):
        g = t.xs(cible, level="target").loc[modeles]
        pos = np.arange(len(modeles)) + (j - (len(cibles) - 1) / 2) * hauteur
        for ax, mesure in zip(axes, ["gain", "R2"]):
            ax.barh(pos, g[(mesure, "mean")], height=hauteur, xerr=g[(mesure, "std")], label=cible, capsize=2, zorder=2)
    for ax in axes:
        _cote_mieux(ax)
    axes[0].set(yticks=range(len(modeles)), yticklabels=modeles,
                xlabel=_t("gain de MAE par rapport au score moyen (points)", "MAE gain over the mean score (points)"))
    axes[0].set_title(_t("Gain : bon résultat à droite de 0 (zone bleutée)", "Gain: good result right of 0 (shaded area)"),
                      loc="left", fontsize=10)
    axes[0].invert_yaxis()
    axes[1].set(xlabel="R²")
    axes[1].set_title(_t("R² : bon résultat à droite de 0 (mieux que la moyenne)",
                         "R²: good result right of 0 (better than the mean)"), loc="left", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(cibles), frameon=False)
    fig.text(0.01, 0.01, _t("Un gain n'est démontré que si son intervalle de confiance à 95 % (tableau) est entièrement "
                            "au-dessus de 0. Barres : moyenne ± écart-type sur les plis.",
                            "A gain is demonstrated only if its 95% confidence interval (table) is entirely above 0. "
                            "Bars: mean ± standard deviation over the folds."), fontsize=9, color="#52514e")
    plt.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(fichier, dpi=150)
    plt.show()


def graphique_predit_observe(vrai, pred, titre):
    """Un point par enfant (graphique individuel : à ne pas partager)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    bornes = [min(vrai.min(), pred.min()) - 2, max(vrai.max(), pred.max()) + 2]
    ax.scatter(vrai, pred, s=12, alpha=0.6)
    ax.plot(bornes, bornes, "k--", lw=1, label=_t("prédiction parfaite", "perfect prediction"))
    ax.axhline(vrai.mean(), color="grey", lw=1, label=_t("QIT moyen", "mean FSIQ"))
    ax.set(xlabel=_t("QIT observé", "observed FSIQ"), ylabel=_t("QIT prédit", "predicted FSIQ"), title=titre)
    ax.legend(loc="upper left")
    ax.text(0.98, 0.03, _t("Bon modèle : points le long de la diagonale\nPas de signal : bande horizontale",
                           "Good model: points along the diagonal\nNo signal: horizontal band"),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="#52514e",
            bbox={"facecolor": "white", "edgecolor": "#c3c2b7", "boxstyle": "round,pad=0.4"})
    plt.tight_layout()
    plt.show()


def _gain_vs_moyenne(y, P, plis):
    B = pd.DataFrame(np.nan, index=P.index, columns=y.index)
    for r, _, tr, te in plis:
        B.loc[r, te] = y.loc[tr].mean()
    Y, Pm, Bm = y.to_numpy(float), P.to_numpy(float), B.to_numpy(float)
    tirages = bootstrap(Y, {"moyenne": Bm, "modele": Pm}, 2000, 0)
    gains = tirages["moyenne"]["MAE"] - tirages["modele"]["MAE"]
    mae_m, mae_b = np.abs(Pm - Y).mean(), np.abs(Bm - Y).mean()
    return {"enfants": len(y), "MAE moyenne": float(mae_b), "MAE modèle": float(mae_m), "gain": float(mae_b - mae_m),
            "IC bas": float(np.percentile(gains, 2.5)), "IC haut": float(np.percentile(gains, 97.5))}


def gain_vs_moyenne(y, P, plis):
    """MAE du modèle, MAE de la moyenne d'entraînement de chaque pli, gain et son IC 95 % (bootstrap sur les enfants)."""
    return _traduire(_gain_vs_moyenne(y, P, plis))


def _coefs_un_pli(Xd, y, tr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefs = faire_pipeline(list(Xd.columns)).fit(Xd.loc[tr], y.loc[tr]).named_steps["modele"].coef_
    return np.count_nonzero(coefs)


def variables_gardees(Xd, y, plis, n_jobs=-1):
    """Nombre moyen de coefficients non nuls d'Elastic Net, ajusté comme dans l'évaluation (un modèle par pli)."""
    return float(np.mean(Parallel(n_jobs=n_jobs)(delayed(_coefs_un_pli)(Xd, y, tr) for _, _, tr, _ in plis)))


def simulation_detection(X, colonnes, cols_hrv, qit, niveaux, n_rep, k_plis, graine):
    """Faux QI qui dépend vraiment de 5 indices HRV, avec une force connue (R² planté), le reste étant du bruit ;
    puis la même méthode (Elastic Net sur `colonnes`) : à partir de quelle force le gain est-il démontré ?"""
    tirage = np.random.default_rng(graine)
    cols_signal = list(tirage.choice(cols_hrv, size=5, replace=False))
    z = X[cols_signal].apply(lambda s: s.fillna(s.median())).rank().sum(axis=1)
    z = (z - z.mean()) / z.std()
    bruit = pd.Series(tirage.standard_normal(len(X)), index=X.index)
    lignes = []
    for r2 in niveaux:
        faux_qit = qit.mean() + qit.std() * (np.sqrt(r2) * z + np.sqrt(1 - r2) * bruit)
        plis = plis_stratifies(faux_qit, n_rep, k_plis, graine)
        P = predictions_croisees(X[colonnes], faux_qit, plis)
        lignes.append({"R² planté": r2, **_gain_vs_moyenne(faux_qit, P, plis)})
    t = pd.DataFrame(lignes).set_index("R² planté")
    t["détecté"] = np.where(t["IC bas"] > 0, "oui", "non")
    return _traduire(t)


def _montrer(fig, fichier):
    """Affiche le graphique et, si `fichier` est donné, l'enregistre (graphiques agrégés seulement)."""
    if fichier is not None:
        fig.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.show()


def graphique_simulation(simulation, reel, fichier=None):
    gain, bas, haut = simulation[_k("gain")], simulation[_k("IC bas")], simulation[_k("IC haut")]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.errorbar(simulation.index * 100, gain, yerr=[gain - bas, haut - gain], fmt="o-", capsize=3,
                label=_t("faux QI : gain de MAE [IC 95 %]", "fake IQ: MAE gain [95% CI]"))
    if reel is not None:
        ax.axhspan(reel.gain_ci_low, reel.gain_ci_high, color="tab:red", alpha=0.15)
        ax.axhline(reel.MAE_gain_vs_baseline, color="tab:red", lw=2,
                   label=_t(f"vrai QIT : {reel.MAE_gain_vs_baseline:+.2f} point",
                            f"real FSIQ: {reel.MAE_gain_vs_baseline:+.2f} point"))
    _cote_mieux(ax, axe="y")
    ax.set(xlabel=_t("part de la variance du QI expliquée par la HRV dans le faux QI (%)",
                     "share of IQ variance explained by HRV in the fake IQ (%)"),
           ylabel=_t("gain de MAE sur la moyenne (points)", "MAE gain over the mean (points)"),
           title=_t("À partir de quand la méthode voit-elle un lien ?\n"
                    "Lien détecté quand tout l'intervalle est au-dessus de 0 (zone bleutée)",
                    "From what strength does the method see a link?\n"
                    "Link detected when the whole interval is above 0 (shaded area)"))
    ax.legend(loc="upper left")
    plt.tight_layout()
    _montrer(fig, fichier)


# ===================================================================================== 4.5 Borne
def premiere_composante(X, colonnes):
    """Résumé global de la HRV : 1re composante principale des indices (rangs centrés réduits, trous à la moyenne).
    Calculée sans regarder le QI. Son signe est arbitraire : seule la force du lien compte."""
    Z = X[colonnes].rank()
    Z = ((Z - Z.mean()) / Z.std()).fillna(0)
    pca = PCA(1).fit(Z)
    return pd.Series(pca.transform(Z)[:, 0], index=X.index), pca.explained_variance_ratio_[0]


def _spearman(x, y, Z=None):
    """Spearman entre x et y ; partielle si Z est donné (l'effet des covariables Z est retiré des deux rangs)."""
    rx, ry = rankdata(x), rankdata(y)
    if Z is not None:
        A = np.column_stack([np.ones(len(rx)), Z])
        rx = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ry = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return np.corrcoef(rx, ry)[0, 1]


def bornes_correlation(indices, y, covariables, niveau=0.95):
    """Pour chaque indice (colonne de `indices`) : corrélation de Spearman avec y, brute et ajustée sur les
    covariables, et son intervalle de confiance (transformation de Fisher, erreur type de Bonett et Wright pour
    Spearman). Les intervalles sont élargis pour le nombre d'indices (Bonferroni) : ils tiennent tous ENSEMBLE avec
    la confiance `niveau`. « variance max » = la plus grande part de variance (en rangs) compatible avec
    l'intervalle, c'est-à-dire max(IC bas², IC haut²)."""
    quantile = norm.ppf(1 - (1 - niveau) / (2 * indices.shape[1]))
    lignes = []
    for nom in indices.columns:
        ok = indices[nom].notna() & y.notna()
        x, yy = indices.loc[ok, nom].to_numpy(float), y[ok].to_numpy(float)
        Z = covariables.loc[ok].apply(lambda s: s.fillna(s.median())).to_numpy(float)
        ligne = {"indice": nom, "enfants": int(ok.sum())}
        for suffixe, z in [("", None), (" ajusté", Z)]:
            r = _spearman(x, yy, z)
            k = 0 if z is None else z.shape[1]                  # une covariable retirée = un degré de liberté en moins
            ecart = quantile * np.sqrt((1 + r ** 2 / 2) / (len(x) - 3 - k))
            bas, haut = np.tanh(np.arctanh(r) - ecart), np.tanh(np.arctanh(r) + ecart)
            ligne.update({f"ρ{suffixe}": r, f"IC bas{suffixe}": bas, f"IC haut{suffixe}": haut,
                          f"variance max{suffixe}": max(bas ** 2, haut ** 2),
                          f"démontré{suffixe}": "oui" if bas > 0 or haut < 0 else "non"})
        lignes.append(ligne)
    return _traduire(pd.DataFrame(lignes).set_index("indice"))


def variance_detectable(n, m, niveau=0.95, puissance=0.80):
    """Plus petite part de variance qu'un test de corrélation sur n enfants (m tests, Bonferroni) détecte avec cette
    puissance (approximation de Fisher)."""
    z = norm.ppf(1 - (1 - niveau) / (2 * m)) + norm.ppf(puissance)
    return np.tanh(z / np.sqrt(n - 3)) ** 2


def tableau_bornes(bornes):
    def rho(suffixe):
        return [f"{r:+.2f} " + _t(f"[{a:+.2f} ; {b:+.2f}]", f"[{a:+.2f}, {b:+.2f}]") for r, a, b in
                zip(bornes[_k(f"ρ{suffixe}")], bornes[_k(f"IC bas{suffixe}")], bornes[_k(f"IC haut{suffixe}")])]
    return pd.DataFrame({
        _k("enfants"): bornes[_k("enfants")],
        _t("ρ brut [IC]", "raw ρ [CI]"): rho(""),
        _t("au plus (brut)", "at most (raw)"): [f"{v:.1%}" for v in bornes[_k("variance max")]],
        _t("démontré (brut)", "demonstrated (raw)"): bornes[_k("démontré")],
        _t("ρ ajusté [IC]", "adjusted ρ [CI]"): rho(" ajusté"),
        _t("au plus (ajusté)", "at most (adjusted)"): [f"{v:.1%}" for v in bornes[_k("variance max ajusté")]],
        _t("démontré (ajusté)", "demonstrated (adjusted)"): bornes[_k("démontré ajusté")],
    }, index=bornes.index)


def graphique_bornes(bornes, r_detectable, fichier=None):
    fig, ax = plt.subplots(figsize=(7.5, 0.7 * len(bornes) + 1.6))
    y = np.arange(len(bornes))[::-1]
    ax.axvspan(-r_detectable, r_detectable, color="grey", alpha=0.12,
               label=_t("liens trop faibles pour être détectés 8 fois sur 10", "links too weak to be detected 8 times in 10"))
    for decalage, suffixe, couleur, nom in [(0.12, "", "tab:blue", _t("corrélation brute", "raw correlation")),
                                            (-0.12, " ajusté", "tab:orange",
                                             _t("corrélation ajustée sur les covariables cliniques",
                                                "correlation adjusted for clinical covariates"))]:
        r = bornes[_k(f"ρ{suffixe}")]
        ax.errorbar(r, y + decalage, xerr=[r - bornes[_k(f"IC bas{suffixe}")], bornes[_k(f"IC haut{suffixe}")] - r],
                    fmt="o", capsize=3, color=couleur, label=nom)
    ax.axvline(0, color="k", lw=1.2)
    ax.set(yticks=y, yticklabels=bornes.index,
           xlabel=_t("corrélation de Spearman avec le QIT [IC simultanés à 95 %]",
                     "Spearman correlation with FSIQ [simultaneous 95% CI]"),
           title=_t("Lien démontré si l'intervalle ne touche pas la ligne 0\n"
                    "Sinon, la HRV explique au plus (borne de l'intervalle)² de la variance",
                    "Link demonstrated if the interval does not touch the 0 line\n"
                    "Otherwise, HRV explains at most (interval bound)² of the variance"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=8, frameon=False)
    plt.tight_layout()
    _montrer(fig, fichier)


# ===================================================================================== 5-6. Classification
def resume_classification(y_bin, P):
    """AUC (moyenne des répétitions, IC 95 % par bootstrap sur les enfants) et performances au seuil 0,5."""
    yb = y_bin.to_numpy(int)
    proba = P.mean(axis=0).to_numpy(float)
    tirage = np.random.default_rng(0)
    aucs = []
    for _ in range(2000):
        i = tirage.integers(0, len(yb), len(yb))
        if 0 < yb[i].sum() < len(i):
            aucs.append(roc_auc_score(yb[i], proba[i]))
    classe = (proba >= 0.5).astype(int)
    return _traduire({
        "AUC": np.mean([roc_auc_score(yb, P.loc[r].to_numpy(float)) for r in P.index]),
        "AUC IC bas": np.percentile(aucs, 2.5), "AUC IC haut": np.percentile(aucs, 97.5),
        "sensibilité": recall_score(yb, classe), "spécificité": recall_score(yb, classe, pos_label=0),
        "exactitude": (classe == yb).mean(), "exactitude équilibrée": balanced_accuracy_score(yb, classe),
        "kappa": cohen_kappa_score(yb, classe)})


# ===================================================================================== 7. Hypothèses
def graphique_hypotheses(hypotheses, fichier=None):
    gain, bas, haut = hypotheses[_k("gain")], hypotheses[_k("IC bas")], hypotheses[_k("IC haut")]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(hypotheses) + 1.3))
    y_pos = np.arange(len(hypotheses))[::-1]
    ax.errorbar(gain, y_pos, fmt="o", capsize=3, xerr=[gain - bas, haut - gain], zorder=2)
    _cote_mieux(ax)
    ax.set(yticks=y_pos, yticklabels=hypotheses.index,
           xlabel=_t("gain de MAE sur le QIT moyen (points) [IC 95 %]", "MAE gain over the mean FSIQ (points) [95% CI]"),
           title=_t("Hypothèse soutenue si tout l'intervalle est à droite de 0 (zone bleutée)",
                    "Hypothesis supported if the whole interval is right of 0 (shaded area)"))
    plt.tight_layout()
    _montrer(fig, fichier)


# ===================================================================================== 8. Prédiction conforme
def quantile_conforme(erreurs, alpha):
    n = len(erreurs)
    return np.quantile(erreurs, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")


def intervalles_conformes(Xd, y, pred, plis, alpha, modele="Elastic Net", selection="toutes", n_iter=12):
    """Plis d'UNE répétition ; pred = prédictions hors pli de cette répétition (section 4), faites par ce modèle.
    Pour chaque pli de test, q est calibrée sur les SEULS enfants d'entraînement : leurs erreurs dans une validation
    croisée interne à 5 plis (le modèle ne les a pas vus). La couverture est ensuite mesurée sur les enfants du pli,
    qui ne servent jamais à calibrer leur propre intervalle. Même calcul pour le score moyen (q_ref)."""
    morceaux = []
    for _, _, tr, te in plis:
        y_tr = y[tr]
        internes = [(1, k, [tr[i] for i in a], [tr[i] for i in b]) for k, (a, b)
                    in enumerate(KFold(K_INTERNE, shuffle=True, random_state=0).split(tr), start=1)]
        p_interne = predictions_croisees(Xd.loc[tr], y_tr, internes, modele, selection, n_iter).loc[1]
        moyenne_interne = pd.Series(np.nan, index=y_tr.index)
        for _, _, a, b in internes:
            moyenne_interne[b] = y[a].mean()
        morceaux.append(pd.DataFrame({
            "vrai": y[te], "pred": pred[te], "moyenne": y_tr.mean(),
            "q": quantile_conforme(np.abs(y_tr - p_interne[tr]).to_numpy(), alpha),
            "q_ref": quantile_conforme(np.abs(y_tr - moyenne_interne).to_numpy(), alpha)}))
    return _traduire(pd.concat(morceaux))


def graphique_intervalles(conforme, seuil, alpha):
    """Graphique individuel : à ne pas partager."""
    tri = conforme.sort_values("pred")
    vrai = tri[_k("vrai")]
    dedans = np.abs(vrai - tri.pred) <= tri.q
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.vlines(range(len(tri)), tri.pred - tri.q, tri.pred + tri.q, color="lightsteelblue", lw=2)
    ax.scatter(range(len(tri)), vrai, s=10, c=np.where(dedans, "black", "tab:red"), zorder=3)
    ax.axhline(seuil, color="grey", ls=":", lw=1)
    q_ref = conforme["q_ref"].mean()
    ax.set(xlabel=_t("enfants triés par QIT prédit", "children sorted by predicted FSIQ"), ylabel=_t("QIT", "FSIQ"),
           title=_t(f"Intervalles à {1 - alpha:.0%} (rouge : vrai QIT en dehors)\n"
                    f"Modèle utile si ses barres sont plus courtes que ± {q_ref:.1f}, la largeur autour du QIT moyen",
                    f"{1 - alpha:.0%} intervals (red: true FSIQ outside)\n"
                    f"Model useful if its bars are shorter than ± {q_ref:.1f}, the width around the mean FSIQ"))
    plt.tight_layout()
    plt.show()


# ===================================================================================== 9. Explication
def _importance_un_pli(Xd, y, tr, te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = faire_pipeline(list(Xd.columns), "Logistique").fit(Xd.loc[tr], y.loc[tr])
        r = permutation_importance(pipe, Xd.loc[te], y.loc[te], scoring="roc_auc", n_repeats=5, random_state=0)
    return pd.Series(r.importances_mean, index=Xd.columns)


def importances_permutation(Xd, y, plis, n_jobs=-1):
    """Baisse moyenne de l'AUC quand on mélange chaque variable chez les enfants de test, sur les plis donnés."""
    morceaux = Parallel(n_jobs=n_jobs)(delayed(_importance_un_pli)(Xd, y, tr, te) for _, _, tr, te in plis)
    return pd.concat(morceaux, axis=1).mean(axis=1).sort_values(ascending=False)


def graphique_importances(importances, titre, fichier=None):
    haut = importances.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.barh(haut.index, haut, color="tab:blue", zorder=2)
    _cote_mieux(ax)
    ax.set(xlabel=_t("baisse de l'AUC quand l'indice est mélangé", "drop in AUC when the index is shuffled"),
           title=f"{titre}\n" + _t("Au-dessus de 0 (zone bleutée) : l'indice aide le modèle ; près de 0 : il ne sert pas",
                                   "Above 0 (shaded area): the index helps the model; near 0: it is not used"))
    plt.tight_layout()
    _montrer(fig, fichier)


# ===================================================================================== Noms anglais
# Les mêmes fonctions, sous des noms anglais, pour 01_prediction_qit_en.ipynb.
set_language = choisir_langue
find_column = trouver_colonne
normalize_text = normaliser_texte
load_files = charger
build_dataset = construire_donnees
make_folds = creer_plis
shared_folds = plis_communs
stratified_folds = plis_stratifies
correlation_vs_chance = correlation_contre_hasard
build_pipeline = faire_pipeline
SEARCH_SPACES = ESPACES
cross_predictions = predictions_croisees
write_predictions = ecrire_predictions
per_fold_metrics = mesures_par_pli
comparison_by_model = comparaison_par_modele
ci = ic
results_table = tableau_resultats
plot_gains = graphique_gains
plot_predicted_observed = graphique_predit_observe
gain_vs_mean = gain_vs_moyenne
kept_variables = variables_gardees
detection_simulation = simulation_detection
plot_simulation = graphique_simulation
first_component = premiere_composante
correlation_bounds = bornes_correlation
detectable_variance = variance_detectable
bounds_table = tableau_bornes
plot_bounds = graphique_bornes
classification_summary = resume_classification
plot_hypotheses = graphique_hypotheses
conformal_quantile = quantile_conforme
conformal_intervals = intervalles_conformes
plot_intervals = graphique_intervalles
permutation_importances = importances_permutation
plot_importances = graphique_importances
