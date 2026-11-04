# %% [markdown]
# Ouvre un fichier de données MATLAB (.mat) en Python et affiche son contenu.
#
# Installation (une seule fois) :  pip install scipy numpy pandas h5py

# %%
import numpy as np
import pandas as pd
import scipy.io

try:
    from IPython.display import display
except ImportError:
    display = print

# ============ PARAMÈTRE À MODIFIER ============
FICHIER_MAT = "._001-GAYlo-M0.mat"
# ==============================================


def charger_mat(chemin):
    """Renvoie un dictionnaire {nom de variable MATLAB: valeur Python}."""
    try:
        donnees = scipy.io.loadmat(chemin, squeeze_me=True, simplify_cells=True)
        return {k: v for k, v in donnees.items() if not k.startswith("__")}
    except NotImplementedError:
        # Fichiers enregistrés avec save(..., '-v7.3') : format HDF5
        import h5py
        donnees = {}
        with h5py.File(chemin, "r") as f:
            def lire(nom, obj):
                if isinstance(obj, h5py.Dataset):
                    donnees[nom] = np.array(obj).T  # MATLAB stocke les colonnes en premier
            f.visititems(lire)
        return donnees


def decrire(nom, valeur, decalage=""):
    """Affiche le type et la taille de chaque variable (y compris dans les struct)."""
    if isinstance(valeur, dict):
        print(f"{decalage}{nom} : struct avec {len(valeur)} champ(s)")
        for k, v in valeur.items():
            decrire(k, v, decalage + "    ")
    elif isinstance(valeur, np.ndarray):
        print(f"{decalage}{nom} : tableau {valeur.shape}, type {valeur.dtype}")
    else:
        print(f"{decalage}{nom} : {type(valeur).__name__} = {str(valeur)[:60]}")


donnees = charger_mat(FICHIER_MAT)
print(f"Variables trouvées dans {FICHIER_MAT} :\n")
for nom, valeur in donnees.items():
    decrire(nom, valeur)

# %% Conversion en tableaux pandas des variables qui s'y prêtent
tableaux = {}
for nom, valeur in donnees.items():
    if isinstance(valeur, np.ndarray) and valeur.ndim in (1, 2) and valeur.dtype.kind in "iuf":
        tableaux[nom] = pd.DataFrame(valeur if valeur.ndim == 2 else {nom: valeur})
    elif isinstance(valeur, dict):
        try:
            tableaux[nom] = pd.DataFrame(valeur)  # struct dont les champs ont la même longueur
        except ValueError:
            pass

for nom, df in tableaux.items():
    print(f"\n=== {nom} : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    display(df.head())

# %% Exemple : récupérer une variable précise (remplacez "RR" par le vrai nom)
# rr = donnees["RR"]
# print(rr[:10])
