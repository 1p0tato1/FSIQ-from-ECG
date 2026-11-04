"""Corrige le décalage de 2 colonnes vers la gauche de quelques lignes de HRV_NonLineaire_Full.csv.

Sur les lignes listées (numéros de ligne du fichier, comme dans Excel ou VS Code), les valeurs à partir de la
colonne CB sont décalées de 2 colonnes vers la gauche : le script les décale de 2 colonnes vers la droite, de sorte
que la valeur de CB passe en CD, celle de CC en CE, etc. Les 2 cases vides (« - ») qui terminent ces lignes prennent
la place libérée en CB et CC. Les autres lignes et les colonnes avant CB ne changent pas.

Le fichier est traité octet par octet : encodage, séparateur et fins de ligne sont conservés tels quels.
Une copie de l'original est gardée à côté (..._avant_correction.csv), et le script refuse de tourner deux fois.
Il n'affiche que des numéros de ligne et des nombres de cases, jamais de valeurs.

Depuis le dossier FSIQ-from-ECG :
    python "GroupA/Jade/corriger_hrv_nonlineaire.py"               aperçu, ne modifie rien
    python "GroupA/Jade/corriger_hrv_nonlineaire.py" --appliquer   corrige le fichier
"""
import argparse
import shutil
import sys
from pathlib import Path

LIGNES = [69, 107, 108, 160, 170, 174, 187, 190, 192]
COLONNE = "CB"
DECALAGE = 2
VIDES = {b"", b"-", b"--", b"na", b"nan", b"n/a"}     # dans ce fichier, une case vide contient souvent « - »


def indice_colonne(lettres):
    """'A' -> 0, 'Z' -> 25, 'AA' -> 26, 'CB' -> 79."""
    n = 0
    for ch in lettres.upper():
        n = n * 26 + ord(ch) - ord("A") + 1
    return n - 1


def lettres_colonne(indice):
    """79 -> 'CB'."""
    lettres = ""
    indice += 1
    while indice:
        indice, reste = divmod(indice - 1, 26)
        lettres = chr(ord("A") + reste) + lettres
    return lettres


def main():
    racine = next(d for d in Path(__file__).resolve().parents if (d / "evaluate_model.py").exists())
    ap = argparse.ArgumentParser()
    ap.add_argument("--fichier", default=str(racine / "data" / "HRV_NonLineaire_Full.csv"))
    ap.add_argument("--appliquer", action="store_true", help="corrige le fichier (sinon simple aperçu)")
    args = ap.parse_args()

    chemin = Path(args.fichier)
    sauvegarde = chemin.with_name(chemin.stem + "_avant_correction" + chemin.suffix)
    if sauvegarde.exists():
        sys.exit(f"{sauvegarde.name} existe déjà : le fichier a déjà été corrigé. Pour recommencer, remettre "
                 "l'original à la place du fichier corrigé et supprimer la sauvegarde.")

    lignes = chemin.read_bytes().splitlines(keepends=True)
    sep = max([b"\t", b";", b","], key=lignes[0].count)
    entete = lignes[0].rstrip(b"\r\n").split(sep)
    i = indice_colonne(COLONNE)
    print(f"{chemin.name} : {len(lignes)} lignes, séparateur {sep!r}, {len(entete)} cases dans l'en-tête")
    print(f"Colonne {COLONNE} = « {entete[i].decode('utf-8', errors='replace')} » ; "
          f"après correction, son contenu actuel passera en « {entete[i + DECALAGE].decode('utf-8', errors='replace')} »\n")

    erreurs = 0
    for numero in sorted(LIGNES):
        ligne = lignes[numero - 1]
        fin = ligne[len(ligne.rstrip(b"\r\n")):]
        cases = ligne.rstrip(b"\r\n").split(sep)
        if b'"' in ligne:
            print(f"ligne {numero} : contient des guillemets, non modifiée (à corriger à la main)")
            erreurs += 1
            continue
        if len(cases) + DECALAGE <= len(entete):
            nouvelles = cases[:i] + [b""] * DECALAGE + cases[i:]           # la ligne a la place de s'allonger
            comment = f"{DECALAGE} cases vides insérées en {COLONNE}"
        elif all(c.strip().lower() in VIDES for c in cases[-DECALAGE:]):
            nouvelles = cases[:i] + cases[-DECALAGE:] + cases[i:-DECALAGE]  # les cases vides de la fin passent en CB, CC...
            marques = " ".join(f"« {c.strip().decode('utf-8', errors='replace')} »" for c in cases[-DECALAGE:])
            comment = (f"les {DECALAGE} dernières cases ({marques}, vides) passent en "
                       + " et ".join(lettres_colonne(i + d) for d in range(DECALAGE)))
        else:
            print(f"ligne {numero} : les {DECALAGE} dernières cases ne sont pas vides, une valeur serait perdue : non modifiée")
            erreurs += 1
            continue
        remplies = sum(1 for c in cases[i:] if c.strip().lower() not in VIDES)
        print(f"ligne {numero} : {len(cases)} cases -> {len(nouvelles)}, {remplies} valeurs décalées de {DECALAGE} "
              f"colonnes vers la droite ; {comment}")
        lignes[numero - 1] = sep.join(nouvelles) + fin

    if not args.appliquer:
        print("\nAperçu seulement : relancer avec --appliquer pour corriger le fichier.")
        return
    if erreurs:
        sys.exit("\nCertaines lignes posent problème : rien n'a été modifié.")
    shutil.copy2(chemin, sauvegarde)
    chemin.write_bytes(b"".join(lignes))
    print(f"\nFichier corrigé. Original gardé dans {sauvegarde.name} (dossier data/, ignoré par git).")


if __name__ == "__main__":
    main()
