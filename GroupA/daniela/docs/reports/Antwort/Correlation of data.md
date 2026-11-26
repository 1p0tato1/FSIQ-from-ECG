# Features vs IQ
All correlations are weak (|r| < 0.17). No feature survives FDR correction (all `significative_fdr = False`).

![[Pasted image 20261106154905.png]]

## Top 15 features by correlation with IQ

| feature                     | group    | n   | pearson_r | pearson_p | spearman_rho | spearman_p | pearson_p_fdr | pearson_significative_fdr | spearman_p_fdr | spearman_significative_fdr | abs_pearson |
| --------------------------- | -------- | --- | --------- | --------- | ------------ | ---------- | ------------- | ------------------------- | -------------- | -------------------------- | ----------- |
| Rég à la naissance_Linear   | HRV      | 188 | -0.1651   | 0.0236    | -0.1867      | 0.0103     | 0.8073        | False                     | 0.6277         | False                      | 0.1651      |
| Numéro_Clinical             | Clinical | 190 | 0.1610    | 0.0265    | 0.1465       | 0.0437     | 0.8073        | False                     | 0.9369         | False                      | 0.1610      |
| SFA_Linear                  | HRV      | 189 | -0.1229   | 0.0920    | -0.1194      | 0.1017     | 0.9505        | False                     | 0.9369         | False                      | 0.1229      |
| Mean RR (TD24h)_Linear      | HRV      | 190 | 0.1136    | 0.1187    | 0.0306       | 0.6754     | 0.9505        | False                     | 0.9369         | False                      | 0.1136      |
| Maturation Pulm_Linear      | HRV      | 189 | -0.1017   | 0.1636    | -0.0986      | 0.1771     | 0.9505        | False                     | 0.9369         | False                      | 0.1017      |
| nb RR analyzed total_Linear | HRV      | 190 | -0.0980   | 0.1787    | -0.0629      | 0.3884     | 0.9505        | False                     | 0.9369         | False                      | 0.0980      |
| H (DFA24h)_NonLinear        | HRV      | 188 | 0.0968    | 0.1865    | 0.0978       | 0.1819     | 0.9505        | False                     | 0.9369         | False                      | 0.0968      |
| Katz (24h)_NonLinear        | HRV      | 182 | 0.0879    | 0.2382    | 0.0465       | 0.5327     | 0.9505        | False                     | 0.9369         | False                      | 0.0879      |
| alpha2 (DFADay)_NonLinear   | HRV      | 188 | 0.0847    | 0.2479    | 0.0933       | 0.2030     | 0.9505        | False                     | 0.9369         | False                      | 0.0847      |
| Tri ind (GIDay)_Linear      | HRV      | 188 | -0.0830   | 0.2573    | -0.0333      | 0.6503     | 0.9505        | False                     | 0.9369         | False                      | 0.0830      |
| Ptot (FTDay)_Linear         | HRV      | 188 | 0.0774    | 0.2908    | 0.0955       | 0.1924     | 0.9505        | False                     | 0.9369         | False                      | 0.0774      |
| nb RR excluded total_Linear | HRV      | 190 | -0.0766   | 0.2938    | -0.0278      | 0.7035     | 0.9505        | False                     | 0.9369         | False                      | 0.0766      |
| TINN (GIDay)_Linear         | HRV      | 188 | -0.0710   | 0.3333    | -0.0768      | 0.2949     | 0.9505        | False                     | 0.9369         | False                      | 0.0710      |
| NN50 (TD24h)_Linear         | HRV      | 188 | 0.0702    | 0.3382    | 0.0591       | 0.4201     | 0.9505        | False                     | 0.9369         | False                      | 0.0702      |
| AC (ACDC24h)_NonLinear      | HRV      | 188 | 0.0680    | 0.3542    | 0.0379       | 0.6054     | 0.9505        | False                     | 0.9369         | False                      | 0.0680      |

[[WS26-27/BIP Proyect/FSIQ-from-ECG/GroupA/daniela/docs/Antwort/correlation_features_vs_IQ.csv|correlation_features_vs_IQ]]
---

# Linear Operations vs IQ
Top correlations driven by `Maturation Pulm / Prmatur` ratio (n=15, very small sample). No combination survives FDR correction. Results with n=15 are unreliable; n=51 rows (last 4) have significant p-values but low Spearman correlation.

![[Pasted image 20261108161834.png]]

## Top 20 lineal operations (n ≥ 10) from 67221 operations

| feature_a | feature_b | operation | derived_feature | iq_target | n | pearson_r | pearson_p | spearman_rho | spearman_p | pearson_p_fdr | pearson_significative | spearman_p_fdr | spearman_significative | abs_pearson |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Col6_WPPSI | 15 | -0.8278 | 0.0001 | -0.7810 | 0.0006 | 0.3014 | False | 0.5701 | False | 0.8278 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Rang percentile_WPPSI | 15 | -0.8043 | 0.0003 | -0.7711 | 0.0008 | 0.3632 | False | 0.5701 | False | 0.8043 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Col11_WPPSI | 15 | -0.8031 | 0.0003 | -0.7725 | 0.0007 | 0.3632 | False | 0.5701 | False | 0.8031 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Col12_WPPSI | 15 | -0.7876 | 0.0005 | -0.7711 | 0.0008 | 0.4377 | False | 0.5701 | False | 0.7876 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Note composite_WPPSI | 15 | -0.7352 | 0.0018 | -0.7711 | 0.0008 | 0.5550 | False | 0.5701 | False | 0.7352 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Col5_WPPSI | 15 | -0.7258 | 0.0022 | -0.7224 | 0.0023 | 0.5911 | False | 0.5701 | False | 0.7258 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | Col11_WPPSI | 15 | -0.6389 | 0.0103 | -0.5823 | 0.0228 | 0.8329 | False | 0.8394 | False | 0.6389 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | NS Perf._WPPSI | 15 | -0.6337 | 0.0112 | -0.6556 | 0.0080 | 0.8346 | False | 0.6875 | False | 0.6337 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | Col10_WPPSI | 15 | -0.6155 | 0.0146 | -0.6484 | 0.0089 | 0.8648 | False | 0.6875 | False | 0.6155 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | Rang percentile_WPPSI | 15 | -0.6106 | 0.0156 | -0.5812 | 0.0231 | 0.8888 | False | 0.8428 | False | 0.6106 |
| Maturation Pulm_Linear | Prmatur_Linear | ratio | Maturation Pulm_Linear / Prmatur_Linear | NS Verb._WPPSI | 15 | -0.5904 | 0.0205 | -0.6384 | 0.0104 | 0.9494 | False | 0.6875 | False | 0.5904 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | Note composite_WPPSI | 15 | -0.5870 | 0.0214 | -0.5812 | 0.0231 | 0.9494 | False | 0.8428 | False | 0.5870 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | NS Perf._WPPSI | 15 | -0.5798 | 0.0235 | -0.5485 | 0.0343 | 0.9616 | False | 1.0000 | False | 0.5798 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | Col5_WPPSI | 15 | -0.5545 | 0.0319 | -0.4870 | 0.0656 | 0.9967 | False | 1.0000 | False | 0.5545 |
| HTA_Linear | Prmatur_Linear | ratio | HTA_Linear / Prmatur_Linear | Col8_WPPSI | 15 | 0.5537 | 0.0322 | 0.4430 | 0.0982 | 0.9967 | False | 1.0000 | False | 0.5537 |
| Tabac_Linear | Prmatur_Linear | ratio | Tabac_Linear / Prmatur_Linear | NS Verb._WPPSI | 15 | -0.5177 | 0.0481 | -0.4705 | 0.0767 | 0.9967 | False | 1.0000 | False | 0.5177 |
| nb RR corrected total_Linear | H (DFA24h)_NonLinear | ratio | nb RR corrected total_Linear / H (DFA24h)_NonLinear | Col5_WPPSI | 51 | -0.5115 | 0.0001 | -0.2052 | 0.1485 | 0.2804 | False | 1.0000 | False | 0.5115 |
| alpha1 (DFA24h)_NonLinear | H (DFA24h)_NonLinear | ratio | alpha1 (DFA24h)_NonLinear / H (DFA24h)_NonLinear | Col5_WPPSI | 51 | -0.5115 | 0.0001 | -0.2016 | 0.1559 | 0.2804 | False | 1.0000 | False | 0.5115 |
| LF (FTNight)_Linear | H (DFA24h)_NonLinear | ratio | LF (FTNight)_Linear / H (DFA24h)_NonLinear | Col5_WPPSI | 51 | -0.5115 | 0.0001 | -0.1443 | 0.3124 | 0.2804 | False | 1.0000 | False | 0.5115 |
| M (GI24h)_Linear | H (DFA24h)_NonLinear | ratio | M (GI24h)_Linear / H (DFA24h)_NonLinear | Col5_WPPSI | 51 | -0.5115 | 0.0001 | -0.1486 | 0.2979 | 0.2804 | False | 1.0000 | False | 0.5115 |

[[WS26-27/BIP Proyect/FSIQ-from-ECG/GroupA/daniela/docs/Antwort/correlation_linear_operations_vs_IQ.csv|correlation_linear_operations_vs_IQ]]
---

# Interpretation

## 1. Correlation of each individual parameter with the output (IQ / FSIQ)

**Method:** Pearson correlation coefficient (linear relationship) and Spearman rank correlation (monotonic relationship) between every feature and the IQ target. P-values were adjusted for multiple testing using the False Discovery Rate (FDR) procedure.

**Results:**
*   Sample size per feature: **n = 182 to 190**, i.e., close to the full available cohort.
*   All absolute Pearson correlations are **weak**: |r| < 0.17.
*   The strongest individual features are:
    *   `Rég à la naissance_Linear`: r = −0.1651 (p = 0.0236 uncorrected)
    *   `Numéro_Clinical`: r = 0.1610 (p = 0.0265 uncorrected)
*   Before correction, a few p-values look nominally significant (p < 0.05). **After FDR correction, no feature remains significant** (all `pearson_significative_fdr = False`; adjusted p-values between 0.8073 and 0.9505). Spearman results show the same pattern: weak monotonic relationships, none significant after correction.

**Interpretation:** No single ECG / HRV / clinical parameter has a robust linear (or monotonic) relationship with FSIQ. With ~190 subjects the analysis is sufficiently powered to detect moderate effects, but the signal is too weak to survive correction for testing many features simultaneously. Individual parameters therefore cannot by themselves explain or predict IQ in this cohort.

---

## 2. Correlation of linear combinations with the output (IQ / FSIQ)

**Method:** Pairwise linear operations (primarily ratios such as feature_A / feature_B) between parameters, correlated with the IQ targets. Sample sizes vary because each operation requires non-missing values in both features.

**Results:**
*   The strongest raw correlations come from the ratio **`Maturation Pulm_Linear / Prmatur_Linear`**, but only on **n = 15 subjects** (e.g., r = −0.8278 vs. `Col6_WPPSI`). These do **not** survive FDR correction (adjusted p = 0.3014 – 0.9967). A sample of n = 15 is too small to be reliable; such large coefficients are likely spurious.
*   A second cluster uses **n = 51 subjects** (e.g., `nb RR corrected total_Linear / H (DFA24h)_NonLinear` vs. `Col5_WPPSI`, r = −0.5115, uncorrected p = 0.0001). However:
    *   The FDR-adjusted p-value is **0.2804** (not significant).
    *   The Spearman correlation is weak and non-significant (rho = −0.2052, p = 0.1485).
    *   The large gap between Pearson (strong) and Spearman (weak/non-significant) indicates the Pearson result is likely driven by outliers or a non-linear pattern rather than a true linear association.
*   **No linear combination survives FDR correction.** All `pearson_significative` and `spearman_significative` flags are False.

**Interpretation:** Apparent strong correlations in linear combinations are either based on very small, unreliable samples (n = 15) or vanish after multiple-comparison correction (n = 51). There is **no evidence** that a simple linear combination (sum, ratio, or similar pairwise operation) of the used parameters robustly correlates with FSIQ.

---

## 3. Conclusion

1. **Each individual parameter vs. the output:**  
	All |r| < 0.17; none are statistically significant after FDR correction.

2. **A linear combination (e.g., sum) of the parameters vs. the output:**  
	Pairwise ratios were examined; the strongest effects are not robust (n = 15 unreliable; n = 51 not FDR-significant; Spearman contradicts Pearson). No linear combination yields a significant correlation with IQ.

**Conclusion:** Neither the dataset size (n = 190), nor individual feature correlations, nor simple linear combinations provide a statistically valid linear predictor of FSIQ. The weak correlations suggest that if a relationship exists at all, it is likely non-linear, involves higher-order interactions, or is simply absent in this data. This explains why the machine learning models do not produce better results: there is no strong linear signal to learn, and with ~190 subjects and many features, complex models risk fitting noise rather than a real pattern.

---

## Zusammenfassung

Die Korrelation jedes einzelnen Parameters mit der Ausgangsgröße (IQ) ist schwach (|r| < 0.17); nach FDR-Korrektur ist kein Merkmal signifikant. Auch lineare Kombinationen (z. B. Quotienten) zeigen keine robuste Signifikanz: starke Roh-Korrelationen beruhen auf sehr kleinen Stichproben (n = 15, nicht aussagekräftig) oder verschwinden nach Mehrfachtest-Korrektur (n = 51, Pearson signifikant, aber Spearman nicht und FDR nicht signifikant). Es gibt also keine evidenzbasierte lineare Beziehung zwischen den genutzten Parametern und dem IQ. Dies erklärt, warum die ML-Modelle keine überlegenen Ergebnisse liefern: Es liegt kein starkes lineares Signal in den Daten vor, das gelernt werden könnte.