# Code

Python 3 with numpy, scipy, pandas (and matplotlib for figures). Compile the C routines first (see the main README).
Each folder writes its outputs to the current directory; the archived outputs are in the matching folder of `results/`.
Long simulations take PART NPARTS arguments so they can run in parallel (e.g. `0 2` and `1 2`).

## 1_theory — checks of Theorem 1 and Proposition 1
```
python theorem_check.py          # kappa, T*kappa^2, contrast autocorrelations, exact score variance  -> theorem_check.json, kappa_by_design.csv
python theorem_check2.py         # small-contrast information and forecast-value checks             -> theorem_check2.json
```

## 2_competitors_egarch — 19 competitors and the repaired EGARCH (Supplement S2)
```
python rerun_alpha_bound.py ALL 0 2 alpha_bound_all_p0.csv     # and: ALL 1 2 alpha_bound_all_p1.csv
python analyze_egarch.py                                        # -> egarch_repaired_summary.csv
```
`harness.py` runs the competitor code of `known_generator/` and compiles its EGARCH filter in C automatically.

## 3_collapse_and_test — Figure 1 and the size/power of the regime test
```
python collapse.py 0 2 40        # identification collapse (SD of rho-hat vs T*kappa^2)  -> collapse_p0.json
python scoretest.py 0 2 100      # size and power of the lag-one regime test            -> score_p0.json
```

## 4_var_gains — Table 2 (VaR gain of the HMM against T*kappa^2)
```
python jumparm.py 0.02 0.9 30 0 2 pilot_p0.json      # Panel A: refits on the archived paths (jump-free arm is used here)
python var_by_neff.py pilot_var_eval.csv ../../results/1_theory/kappa_by_design.csv   # -> var_by_neff.csv, t5.tex
python intermediate.py 200 0 2 inter_p0.json         # Panel B: designs with T*kappa^2 = 2, 8, 25, 60
python analyze_intermediate.py                       # run where inter_p*.json are -> intermediate_summary.csv
```

## 5_garch_filtered_test — regime test after GARCH filtering (Section 4.4)
```
python boot_test.py garch_t 300 3000 11 bt_garch_t.csv     # also: garch_mix_iid, garch_hmm_tail, garch_hmm_scale
python analyze_boot.py                                       # run where bt_*.csv are -> boot_summary.csv
```
`monitor.py` contains the mixture, regime-index and GARCH routines used by the test.

## 6_empirical — Section 5 and the practical rule of Section 6
```
python read_rdx.py sp500data.rdx sp500.csv           # S&P 500 data from the R package gets (xz-decompress data/sp500data.rda first)
python daily_pipeline.py --csv sp500.csv --start 1990-01-01 --out SP500_D1 --B 399
python daily_pipeline.py --csv WIN_D1.csv --mt5 --start 2005-01-01 --out WIN_D1 --B 399 --ohlc-split   # MetaTrader export
python intraday_rerun_nested.py --csv WIN.csv --format win --market WIN --timeframe 1 --out WIN_M1
python forecast_tests.py --file SP500:D1=SP500_D1/holdout_forecasts.csv --out tests   # DM, Giacomini-White, MCS
python persistence_check.py --csv prices.csv --date-col Date --price-col Close --garch  # regime diagnostic for any series
```
`nested_start.py` starts the GARCH–HMM from the fitted i.i.d.-mixture solution; `marked_core.py` + `mhmm.c` fit the volatility-regime HMM.
