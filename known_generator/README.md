# Known-generator study (4,000 paths)

Eight two-state Student-t HMM designs, 500 paths each, T = 6,000 (4,800 estimation, 1,200 holdout). Used in Section 4 and Supplements S1–S2.

| File | Contents |
|---|---|
| `01_generate_and_recover_hmm6.py`, `giorno_models.py` | Generator and HMM-6 estimation |
| `02_blind_econometric_competitors.py` | The 19 competing models |
| `GT_GARCH_HMM_Joint_Experiment.py` | GARCH, GARCH-mixture and GARCH–HMM models used in Section 5 |
| `DGP_MANIFEST.csv`, `study_design_PUBLIC.json` | Design, path ids and seeds (every path can be regenerated exactly) |
| `HMM6_PARAMETER_RECOVERY_ALL.csv`, `HMM6_RECOVERY_SUMMARY.csv` | Per-path HMM fits and summaries |
| `PATH_MATCHED_HMM6_VS_19_MODELS.csv`, `ALL_COMPETITOR_SUMMARIES.csv`, `APARCH_COEFFICIENT_ATTRIBUTION.csv` | Competitor comparisons and APARCH estimates |
| `GT_GARCH_HMM_All_8_Verified_*.csv`, `paired_tests_available_only.csv` | Intraday holdout summaries (no price data) |

These files come from the earlier HMM-6 release (github.com/profdrgiorno/Giorno_garch_hmm6) and are unchanged.
