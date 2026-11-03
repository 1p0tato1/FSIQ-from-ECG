---
tags:
  - documentation
---
---
# Data Preprocessing and manual shifting

Documents [[data_preprocessing_m.py]] — the cleaning pipeline that does NOT repair the left-shifted HRV columns. Its output lives in `data/clean_data_m_shifted/`; the column shift was repaired manually afterwards (hence "m_shifted").

## Purpose
Clean the WPPSI results file and the two HRV files (linear and non-linear) so that only complete, consistent subjects with an "M0" visit remain. The HRV column shift is left untouched by the script.

## Pipeline
1. **WPPSI results file** — remove the rows with empty cells; remove the same subjects from the two HRV files (linear and non-linear).
2. **HRV files** — keep only the rows whose "Passage" column equals "M0".
3. **HRV files** — if a row has empty cells in EITHER HRV file, the corresponding subject is removed from BOTH HRV files and from the WPPSI file.
4. **WPPSI file** — remove the subjects that do not appear in any HRV file.

No step repairs the rows whose values are shifted to the left (the last cells are empty because the values moved left by SHIFT columns). That repair was done manually on the cleaned CSVs.

## Matching logic
- Rows are matched on the subject **NUMBER** (first column of all files), NOT on the initials alone (second column): initials are repeated across different subjects (e.g. "DIM" appears twice in the WPPSI file).
- The **date of birth** is used as a consistency check: if it does not match, the row is NOT removed.
- Differences in the initials are only reported.

## Configuration
| Constant | Value | Meaning |
|---|---|---|
| `RESULTS_FILE` | `Résultats_WPPSI_III.csv` | WPPSI results file |
| `HRV_FILES` | `HRV_Lineaire_Full.csv`, `HRV_NonLineaire_Full.csv` | linear and non-linear HRV files |
| `SEPARATOR` | `\t` | tab-separated values |
| `ENCODING` | `utf-8` | file encoding |
| `RESULTS_HEADER_ROWS` | `2` | WPPSI file has 2 header rows |
| `PASSAGE_HEADER` | `Passage` | HRV column used to filter visits |
| `PASSAGE_VALUE` | `M0` | only rows with this value are kept |

## Usage
```
python data_preprocessing_m.py                      # looks for the files in the script's folder
python data_preprocessing_m.py --folder /path       # different folder
python data_preprocessing_m.py --mode all_empty     # WPPSI: remove only completely empty rows
```

- `--folder`: folder containing the input files (default: the script's own folder).
- `--mode`: `any_empty` (default) removes a WPPSI row if it has AT LEAST ONE empty cell; `all_empty` removes it only if ALL data cells are empty.

## Output
Files are written (without modifying anything in place) to:
```
data/clean_data_m_shifted/
  HRV_Lineaire_Full.csv
  HRV_NonLineaire_Full.csv
  Résultats_WPPSI_III.csv
```

All files are read as text, without interpreting anything (no automatic NaN).

## Notes
- Each HRV file has 1 header row followed by 1 coding row (e.g. "M/F", "0/1...") which is preserved as-is.
- A warning is printed if the two HRV files do not contain the same subjects; a subject is kept if it appears in at least one of them.

## Related
- [[Important milestones]] — overall cleaning story and why both versions were saved.
- [[Data Preprocessing and automatic shifting]] — the automatic version that repairs the column shift in-script ([[data_preprocessing.py]]).
- [[data_preprocessing_m.py]] — the Python script itself.
