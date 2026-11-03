---
tags:
  - documentation
---
---
# Data cleaning
Data cleaned in two ways
## First cleaned it following these steps
1. WPPSI results file: remove the rows with empty cells; remove the same  
   subjects from the two HRV files (linear and non-linear).
2. HRV files: keep only the rows whose "Passage" column equals "M0". 
3. HRV files: if a row has empty cells in EITHER HRV file, the corresponding  
   subject is removed from BOTH HRV files and from the WPPSI file.
4. WPPSI file: remove the subjects that do not appear in any HRV file.
## Then realized about the shift of the columns
1. WPPSI results file: remove the rows with empty cells; remove the same  
   subjects from the two HRV files (linear and non-linear).
2. ==🟡HRV files: repair the rows whose values are shifted to the left==  
   ==🟡(the last cells are empty because the values moved left by SHIFT columns).==
3. HRV files: keep only the rows whose "Passage" column equals "M0". 
4. HRV files: if a row has empty cells in EITHER HRV file, the corresponding  
   subject is removed from BOTH HRV files and from the WPPSI file.
5. WPPSI file: remove the subjects that do not appear in any HRV file.
## But saved both ways in case the data shifting altered the data itself
[[WS26-27/BIP Proyect/FSIQ-from-ECG/data/clean_data_m_shifted/HRV_Lineaire_Full.csv|HRV_Lineaire_Full]] was shifted manually after cleaning the data with  [[data_preprocessing_m.py]]. While [[WS26-27/BIP Proyect/FSIQ-from-ECG/data/clean_data_shifted/HRV_Lineaire_Full.csv|HRV_Lineaire_Full]] was shifted through the python file [[data_preprocessing.py]].

Documentation about [[data_preprocessing_m.py]] in [[Data Preprocessing and manual shifting]].

Documentation about [[data_preprocessing.py]] in [[Data Preprocessing and automatic shifting]].

## Key differences between the two files

| Aspect | [[data_preprocessing_m.py]] (manual shifting) | [[data_preprocessing.py]] (automatic shifting) |
|---|---|---|
| Repairs shifted HRV columns | **No** — the left shift is left untouched by the script | **Yes** — via `fix_shifted_rows()`, re-inserts `SHIFT=2` missing cells (`-`) at the best-fitting column |
| Order of shift repair | N/A (none) | Done **before** filtering by `Passage == "M0"` |
| Output directory | `data/clean_data_m_shifted/` | `data/clean_data_shifted/` |
| Shift-related config | None | `SHIFT = 2`, `MISSING = "-"` |
| Extra functions | None | `fix_shifted_rows`, `cell_kind`, `score` |
| Column shift repair | Performed **manually** on the cleaned CSVs afterwards | Performed **automatically** in-script |

**When to use which:** use [[data_preprocessing_m.py]] + [[Data Preprocessing and manual shifting]] when you want the cleaning without any automatic column repair (the shift was fixed by hand on the saved CSVs). Use [[data_preprocessing.py]] + [[Data Preprocessing and automatic shifting]] when you want the shift repaired automatically by the script.

Both files share the same cleaning steps (WPPSI empty-row removal, subject matching by NUMBER with DOB consistency check, `Passage == "M0"` filtering, empty-cell subject removal across both HRV files, and removal of WPPSI-only subjects) and neither modifies the input files in place.

---
# Data merge
On [[data_merge.py]] data is going to be merged in one table easy to use when creating the machine learning methods.

Detailed info in [[Data Merge]].

---
# Clean Features

On [[data_feature.py]] the features get a basic clean.

Detailed info in [[Feature Cleaning]].