---
tags:
  - documentation
---
---
# Data Preprocessing and automatic shifting

Documents [[data_preprocessing.py]] — the cleaning pipeline that repairs the left-shifted HRV columns automatically. Its output lives in `data/clean_data_shifted/`.

## Purpose
Clean the WPPSI results file and the two HRV files (linear and non-linear) so that only complete, consistent subjects with an "M0" visit remain, AND repair the rows whose values are shifted to the left (the last cells are empty because the values moved left by SHIFT columns).

## Pipeline
1. **WPPSI results file** — remove the rows with empty cells; remove the same subjects from the two HRV files (linear and non-linear).
2. **HRV files** — repair the rows whose values are shifted to the left (the last cells are empty because the values moved left by SHIFT columns).
3. **HRV files** — keep only the rows whose "Passage" column equals "M0".
4. **HRV files** — if a row has empty cells in EITHER HRV file, the corresponding subject is removed from BOTH HRV files and from the WPPSI file.
5. **WPPSI file** — remove the subjects that do not appear in any HRV file.

## Shift repair (`fix_shifted_rows`)
- A row is considered "broken" when its last `SHIFT` cells are empty while everything else is filled.
- For each broken row, the position where `SHIFT` missing cells (`-`) must be re-inserted is chosen so that the row best matches the kind of value (missing / number / text) usually found in each column of the well-formed rows.
- When several positions are equally good (a run of missing values) they all produce the same row; otherwise the first is used and a warning is printed.
- `SHIFT = 2` and the missing marker is `-`.

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
| `SHIFT` | `2` | number of columns by which broken rows are shifted left |
| `MISSING` | `-` | value used for missing data in the HRV files |
| `PASSAGE_HEADER` | `Passage` | HRV column used to filter visits |
| `PASSAGE_VALUE` | `M0` | only rows with this value are kept |

## Usage
```
python data_preprocessing.py                      # looks for the files in the script's folder
python data_preprocessing.py --folder /path       # different folder
python data_preprocessing.py --mode all_empty     # WPPSI: remove only completely empty rows
```

- `--folder`: folder containing the input files (default: the script's own folder).
- `--mode`: `any_empty` (default) removes a WPPSI row if it has AT LEAST ONE empty cell; `all_empty` removes it only if ALL data cells are empty.

## Output
Files are written (without modifying anything in place) to:
```
data/clean_data_shifted/
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
- [[Data Preprocessing and manual shifting]] — the version that does not repair the shift in-script ([[data_preprocessing_m.py]]).
- [[data_preprocessing.py]] — the Python script itself.
