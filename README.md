# When Can Market Regimes Improve Value at Risk?
### A Proposed Information Bound for Student-t Hidden Markov Models Identification

Joilson Giorno — Giorno–Tsallis research programme. Paper, online supplement, code and results.

**Main result.** For any two-state hidden Markov model, the information that returns carry about regime persistence is exactly (T−1)κ², where κ measures how well the regime can be recognised from a single return (an order-2 Tsallis mutual information). The effective sample size Tκ² tells, before fitting, whether a regime model can improve Value at Risk.

## Repository map

| Folder | Contents |
|---|---|
| `paper/` | The manuscript: one self-contained `.tex` file (references included), the PDF, and Figure 1 (`collapse.pdf`). |
| `supplement/` | Online supplement S1–S6 (`.tex`, PDF, figure). |
| `known_generator/` | The 4,000-path known-generator study: generator and competitor code, manifest with seeds, per-path fits and summaries. |
| `code/` | Scripts, one folder per part of the paper (see below and `code/README.md`). |
| `results/` | Outputs of every script, same folder names as `code/`. |

## Where each result comes from

| In the paper | Code | Results |
|---|---|---|
| Theorem 1, Table 1, Supplement S3 (checks of the theory) | `code/1_theory/` | `results/1_theory/` |
| Table 1 (fitted ρ̂), Supplement S1 | `known_generator/01_generate_and_recover_hmm6.py` | `known_generator/HMM6_PARAMETER_RECOVERY_ALL.csv` |
| Section 4.1, Supplement S2 (19 competitors, repaired EGARCH) | `known_generator/02_blind_econometric_competitors.py`, `code/2_competitors_egarch/` | `known_generator/PATH_MATCHED_HMM6_VS_19_MODELS.csv`, `results/2_competitors_egarch/` |
| Figure 1, Section 4.3–4.4 (identification collapse, size and power of the test) | `code/3_collapse_and_test/` | `results/3_collapse_and_test/` |
| Table 2, Panels A and B (VaR gains against Tκ²) | `code/4_var_gains/` | `results/4_var_gains/` |
| Section 4.4, "After GARCH filtering" | `code/5_garch_filtered_test/` | `results/5_garch_filtered_test/` |
| Section 5, Tables 3–4 (intraday and daily evidence) | `code/6_empirical/` | `results/6_empirical/`, `known_generator/GT_GARCH_HMM_All_8_Verified_*.csv` |
| Section 6 (when to use regime models) on your own data | `code/6_empirical/persistence_check.py` | — |

## Quick start

```
pip install -r requirements.txt
# compile the small C routines once (Linux/macOS; on Windows use MinGW gcc)
gcc -O2 -shared -fPIC -o code/3_collapse_and_test/hmmfast.so code/3_collapse_and_test/hmmfast.c -lm
gcc -O2 -shared -fPIC -o code/4_var_gains/mhmm.so            code/4_var_gains/mhmm.c -lm
gcc -O2 -shared -fPIC -o code/4_var_gains/aparch2.so         code/4_var_gains/aparch2.c -lm
gcc -O2 -shared -fPIC -o code/5_garch_filtered_test/aparch2.so code/5_garch_filtered_test/aparch2.c -lm
gcc -O2 -shared -fPIC -o code/6_empirical/mhmm.so            code/6_empirical/mhmm.c -lm

# regime diagnostic for any price series (kappa-hat, T*kappa-hat^2, regime test)
python code/6_empirical/persistence_check.py --csv prices.csv --date-col Date --price-col Close --garch
```

All scripts find the files they need from their own location, so they can be run from any folder.

## Data

Market price series are not redistributed. The daily S&P 500 data are the `sp500data` set of the R package `gets` (Pretis, Reade and Sucarrat, 2018, *Journal of Statistical Software* 86(3)); `code/6_empirical/read_rdx.py` reads them without R. All simulated data can be regenerated exactly from the seeds in `known_generator/DGP_MANIFEST.csv`.

## Licence

No licence has been granted yet; see `LICENSE_NOTICE.md`.
