# Lab 10 report tables (analyze.py; generated from results/*.csv)


## W4-01 replication coverage (toys passing ALL runs of a cell; search budget S0)

| batch           | cid              | Qf    | variant   |   toys_all_pass |   toys |   runs_pass |   runs | failing (pass/runs)                                               |
|:----------------|:-----------------|:------|:----------|----------------:|-------:|------------:|-------:|:------------------------------------------------------------------|
| REP-1           | MX-B08c03@AD     | Q05@C | s9        |               5 |     12 |          57 |    108 | C3L(6/9) CR3(0/9) CR3L(6/9) CR4(0/9) CR4L(0/9) CR5(0/9) CR5L(0/9) |
| REP-1           | MX-B08c03@AD     | Q05@C | s9_2      |              10 |     12 |          90 |    108 | CR3(0/9) CR5(0/9)                                                 |
| REP-1           | RW-GL-l0.75-a1.0 | Q05@C | s9        |              11 |     12 |         105 |    108 | CR4(6/9)                                                          |
| REP-1           | RW-GL-l0.75-a1.0 | Q05@C | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)                                                |
| REP-2           | RW-GL-l0.75-a1.0 | Q05@C | s9        |              11 |     12 |          99 |    108 | CR4(0/9)                                                          |
| REP-2           | RW-GL-l0.75-a1.0 | Q05@C | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)                                                |
| REP-2C          | MX-B08c03@AD     | Q01@C | s9        |               5 |     12 |          57 |    108 | C3L(6/9) CR3(0/9) CR3L(6/9) CR4(0/9) CR4L(0/9) CR5(0/9) CR5L(0/9) |
| REP-2C          | MX-B08c03@AD     | Q01@C | s9_2      |              10 |     12 |          90 |    108 | CR3(0/9) CR5(0/9)                                                 |
| REP-2C          | RW-GL-l0.75-a1.0 | Q01@C | s9        |              11 |     12 |         102 |    108 | CR4(3/9)                                                          |
| REP-2C          | RW-GL-l0.75-a1.0 | Q01@C | s9_2      |              10 |     12 |          93 |    108 | CR4(0/9) CR4L(3/9)                                                |
| REP-3           | RW-GL-l0.75-a1.0 | Q05@D | s9        |              11 |     12 |          66 |     72 | CR4(0/6)                                                          |
| REP-3           | RW-GL-l0.75-a1.0 | Q05@D | s9_2      |              10 |     12 |          60 |     72 | CR4(0/6) CR4L(0/6)                                                |
| W3-FIN (Lab 09) | MX-B08c03@AD     | Q01@C | s9        |               7 |     12 |          63 |    108 | CR3(0/9) CR4(0/9) CR4L(0/9) CR5(0/9) CR5L(0/9)                    |
| W3-FIN (Lab 09) | MX-B08c03@AD     | Q01@C | s9_2      |              10 |     12 |          90 |    108 | CR3(0/9) CR5(0/9)                                                 |
| W3-FIN (Lab 09) | RW-GL-l0.75-a1.0 | Q01@C | s9        |              11 |     12 |          99 |    108 | CR4(0/9)                                                          |
| W3-FIN (Lab 09) | RW-GL-l0.75-a1.0 | Q01@C | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)                                                |


## Q05 RW-GL: intersection over REP-1 + REP-2 + REP-3 (every C and D search)

| variant   |   toys_all |   runs |
|:----------|-----------:|-------:|
| s9        |         11 |    288 |
| s9_2      |         10 |    288 |

per toy:

| toy   | s9    | s9_2   |
|:------|:------|:-------|
| C1    | True  | True   |
| C1L   | True  | True   |
| C2    | True  | True   |
| C2L   | True  | True   |
| C3    | True  | True   |
| C3L   | True  | True   |
| CR3   | True  | True   |
| CR3L  | True  | True   |
| CR4   | False | False  |
| CR4L  | True  | False  |
| CR5   | True  | True   |
| CR5L  | True  | True   |


## Q05 RW-GL failures: post-lock classification on the identical objective

| batch   | toy   | variant   | key                          |   MCC |   MCC_gap |   leak_max |   L_hat |   L_Rstar |   dL_vs_Rstar |   tau | category                   |
|:--------|:------|:----------|:-----------------------------|------:|----------:|-----------:|--------:|----------:|--------------:|------:|:---------------------------|
| REP-1   | CR4   | s9        | s9|Q05@C|FULL|i101|s31|S0    | 0.789 |     0.061 |      0.399 |   0.828 |     0.838 |        -0.010 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9        | s9|Q05@C|FULL|i101|s73|S0    | 0.787 |     0.063 |      0.407 |   0.828 |     0.838 |        -0.010 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9        | s9|Q05@C|FULL|i101|s101|S0   | 0.789 |     0.061 |      0.399 |   0.828 |     0.838 |        -0.010 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i307|s31|S0    | 0.790 |     0.060 |      0.392 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i307|s73|S0    | 0.788 |     0.063 |      0.402 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i307|s101|S0   | 0.790 |     0.060 |      0.391 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i401|s31|S0    | 0.789 |     0.061 |      0.392 |   0.827 |     0.838 |        -0.012 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i401|s73|S0    | 0.786 |     0.064 |      0.403 |   0.827 |     0.838 |        -0.012 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i401|s101|S0   | 0.789 |     0.061 |      0.391 |   0.827 |     0.838 |        -0.012 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i503|s31|S0    | 0.789 |     0.061 |      0.398 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i503|s73|S0    | 0.787 |     0.063 |      0.407 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9        | s9|Q05@C|FULL|i503|s101|S0   | 0.786 |     0.064 |      0.410 |   0.828 |     0.839 |        -0.011 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i211|s31|S0    | 0.787 |     0.063 |      0.404 |   0.827 |     0.838 |        -0.011 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i211|s73|S0    | 0.785 |     0.065 |      0.411 |   0.827 |     0.838 |        -0.011 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i211|s101|S0   | 0.784 |     0.066 |      0.413 |   0.827 |     0.838 |        -0.011 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i613|s31|S0    | 0.789 |     0.061 |      0.399 |   0.830 |     0.840 |        -0.010 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i613|s73|S0    | 0.787 |     0.063 |      0.407 |   0.830 |     0.840 |        -0.010 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9        | s9|Q05@D|FULL|i613|s101|S0   | 0.789 |     0.061 |      0.399 |   0.830 |     0.840 |        -0.010 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i31|s31|S0   | 0.799 |     0.047 |      0.359 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i31|s73|S0   | 0.801 |     0.046 |      0.354 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i31|s101|S0  | 0.801 |     0.046 |      0.354 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i73|s31|S0   | 0.797 |     0.049 |      0.366 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i73|s73|S0   | 0.800 |     0.047 |      0.356 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i73|s101|S0  | 0.799 |     0.047 |      0.358 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i101|s31|S0  | 0.798 |     0.048 |      0.365 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i101|s73|S0  | 0.802 |     0.045 |      0.351 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4   | s9_2      | s9_2|Q05@C|FULL|i101|s101|S0 | 0.800 |     0.046 |      0.357 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i307|s31|S0  | 0.800 |     0.046 |      0.356 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i307|s73|S0  | 0.801 |     0.045 |      0.351 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i307|s101|S0 | 0.801 |     0.045 |      0.352 |   0.829 |     0.835 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i401|s31|S0  | 0.804 |     0.043 |      0.336 |   0.827 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i401|s73|S0  | 0.801 |     0.046 |      0.351 |   0.827 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i401|s101|S0 | 0.802 |     0.044 |      0.343 |   0.827 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i503|s31|S0  | 0.798 |     0.049 |      0.366 |   0.829 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i503|s73|S0  | 0.801 |     0.046 |      0.354 |   0.829 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-2   | CR4   | s9_2      | s9_2|Q05@C|FULL|i503|s101|S0 | 0.800 |     0.046 |      0.357 |   0.829 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i211|s31|S0  | 0.797 |     0.050 |      0.369 |   0.828 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i211|s73|S0  | 0.799 |     0.048 |      0.361 |   0.828 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i211|s101|S0 | 0.799 |     0.048 |      0.362 |   0.828 |     0.834 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i613|s31|S0  | 0.799 |     0.048 |      0.364 |   0.831 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i613|s73|S0  | 0.801 |     0.046 |      0.354 |   0.831 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-3   | CR4   | s9_2      | s9_2|Q05@D|FULL|i613|s101|S0 | 0.800 |     0.046 |      0.358 |   0.831 |     0.836 |        -0.006 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i31|s31|S0   | 0.859 |     0.059 |      0.382 |   0.828 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i31|s73|S0   | 0.858 |     0.061 |      0.386 |   0.828 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i31|s101|S0  | 0.770 |     0.148 |      0.536 |   0.828 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i73|s31|S0   | 0.863 |     0.056 |      0.373 |   0.827 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i73|s73|S0   | 0.864 |     0.055 |      0.370 |   0.827 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i73|s101|S0  | 0.864 |     0.054 |      0.370 |   0.827 |     0.836 |        -0.008 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i101|s31|S0  | 0.883 |     0.036 |      0.315 |   0.825 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i101|s73|S0  | 0.883 |     0.036 |      0.316 |   0.825 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |
| REP-1   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i101|s101|S0 | 0.883 |     0.036 |      0.316 |   0.825 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i307|s31|S0  | 0.884 |     0.034 |      0.311 |   0.828 |     0.835 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i307|s73|S0  | 0.884 |     0.034 |      0.310 |   0.828 |     0.835 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i307|s101|S0 | 0.884 |     0.035 |      0.313 |   0.828 |     0.835 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i401|s31|S0  | 0.882 |     0.036 |      0.315 |   0.833 |     0.839 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i401|s73|S0  | 0.882 |     0.037 |      0.318 |   0.833 |     0.839 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i401|s101|S0 | 0.882 |     0.037 |      0.317 |   0.833 |     0.839 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i503|s31|S0  | 0.878 |     0.041 |      0.331 |   0.826 |     0.833 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i503|s73|S0  | 0.878 |     0.041 |      0.331 |   0.826 |     0.833 |        -0.007 | 0.001 | objective / estimand issue |
| REP-2   | CR4L  | s9_2      | s9_2|Q05@C|FULL|i503|s101|S0 | 0.878 |     0.041 |      0.331 |   0.826 |     0.833 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i211|s31|S0  | 0.882 |     0.037 |      0.319 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i211|s73|S0  | 0.882 |     0.037 |      0.319 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i211|s101|S0 | 0.881 |     0.037 |      0.321 |   0.827 |     0.834 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i613|s31|S0  | 0.875 |     0.043 |      0.340 |   0.824 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i613|s73|S0  | 0.674 |     0.244 |      0.616 |   0.824 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |
| REP-3   | CR4L  | s9_2      | s9_2|Q05@D|FULL|i613|s101|S0 | 0.674 |     0.244 |      0.616 |   0.824 |     0.832 |        -0.007 | 0.001 | objective / estimand issue |


## Q05 RW-GL distributions over all REP runs

| variant   | toy   |   MCC_med |   MCC_min |   gap_max |   leak_med |   leak_max |   sup_MCC |   abs_good |
|:----------|:------|----------:|----------:|----------:|-----------:|-----------:|----------:|-----------:|
| s9        | C1    |     0.973 |     0.973 |     0.000 |      0.045 |      0.047 |     0.973 |      1.000 |
| s9        | C1L   |     0.998 |     0.998 |     0.000 |      0.037 |      0.038 |     0.998 |      1.000 |
| s9        | C2    |     0.977 |     0.977 |     0.000 |      0.042 |      0.046 |     0.977 |      1.000 |
| s9        | C2L   |     0.996 |     0.996 |     0.000 |      0.040 |      0.041 |     0.997 |      1.000 |
| s9        | C3    |     0.970 |     0.970 |     0.001 |      0.073 |      0.074 |     0.971 |      1.000 |
| s9        | C3L   |     0.994 |     0.994 |     0.001 |      0.055 |      0.059 |     0.995 |      1.000 |
| s9        | CR3   |     0.864 |     0.862 |     0.008 |      0.184 |      0.191 |     0.871 |      0.000 |
| s9        | CR3L  |     0.956 |     0.949 |     0.016 |      0.205 |      0.254 |     0.964 |      1.000 |
| s9        | CR4   |     0.789 |     0.784 |     0.066 |      0.399 |      0.413 |     0.850 |      0.000 |
| s9        | CR4L  |     0.911 |     0.908 |     0.009 |      0.216 |      0.246 |     0.917 |      1.000 |
| s9        | CR5   |     0.878 |     0.878 |     0.007 |      0.180 |      0.192 |     0.885 |      0.000 |
| s9        | CR5L  |     0.952 |     0.947 |     0.008 |      0.167 |      0.209 |     0.955 |      1.000 |
| s9_2      | C1    |     0.973 |     0.973 |     0.000 |      0.044 |      0.046 |     0.973 |      1.000 |
| s9_2      | C1L   |     0.998 |     0.998 |     0.000 |      0.037 |      0.038 |     0.998 |      1.000 |
| s9_2      | C2    |     0.976 |     0.976 |     0.001 |      0.061 |      0.064 |     0.977 |      1.000 |
| s9_2      | C2L   |     0.996 |     0.996 |     0.000 |      0.039 |      0.040 |     0.997 |      1.000 |
| s9_2      | C3    |     0.970 |     0.970 |     0.001 |      0.073 |      0.075 |     0.971 |      1.000 |
| s9_2      | C3L   |     0.994 |     0.994 |     0.001 |      0.057 |      0.064 |     0.995 |      1.000 |
| s9_2      | CR3   |     0.859 |     0.855 |     0.013 |      0.160 |      0.188 |     0.869 |      0.000 |
| s9_2      | CR3L  |     0.945 |     0.936 |     0.029 |      0.251 |      0.295 |     0.965 |      1.000 |
| s9_2      | CR4   |     0.800 |     0.797 |     0.050 |      0.357 |      0.369 |     0.847 |      0.000 |
| s9_2      | CR4L  |     0.880 |     0.674 |     0.244 |      0.326 |      0.616 |     0.919 |      0.000 |
| s9_2      | CR5   |     0.880 |     0.878 |     0.006 |      0.132 |      0.148 |     0.884 |      0.000 |
| s9_2      | CR5L  |     0.929 |     0.918 |     0.040 |      0.253 |      0.293 |     0.958 |      1.000 |


## REP-4 blind S1 escalations (reported separately; selections not replaced)

| toy   | variant   | key                             |   L_hat |   MCC |   leak_max | pass_lab05   | category                   |
|:------|:----------|:--------------------------------|--------:|------:|-----------:|:-------------|:---------------------------|
| CR3   | s9        | s9|Q05@C|MASK-ALL|i31|s31|S1    |   0.839 | 0.863 |      0.184 | True         | recovered                  |
| CR3   | s9        | s9|Q05@C|MASK-ALL|i73|s31|S1    |   0.843 | 0.865 |      0.181 | True         | recovered                  |
| CR3   | s9        | s9|Q05@C|MASK-ALL|i101|s31|S1   |   0.838 | 0.862 |      0.190 | True         | recovered                  |
| CR3   | s9_2      | s9_2|Q05@C|MASK-ALL|i101|s31|S1 |   0.835 | 0.857 |      0.170 | True         | recovered                  |
| CR3   | s9        | s9|Q05@C|MASK-ALL|i307|s31|S1   |   0.841 | 0.864 |      0.178 | True         | recovered                  |
| CR3   | s9_2      | s9_2|Q05@C|MASK-ALL|i401|s31|S1 |   0.839 | 0.857 |      0.174 | True         | recovered                  |
| CR4   | s9        | s9|Q05@C|FULL|i31|s31|S1        |   0.828 | 0.844 |      0.246 | True         | recovered                  |
| CR4   | s9        | s9|Q05@C|FULL|i73|s31|S1        |   0.826 | 0.844 |      0.253 | True         | recovered                  |
| CR4   | s9        | s9|Q05@C|FULL|i101|s31|S1       |   0.828 | 0.788 |      0.404 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i31|s31|S1      |   0.829 | 0.801 |      0.352 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i73|s31|S1      |   0.827 | 0.800 |      0.353 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i101|s31|S1     |   0.829 | 0.802 |      0.351 | False        | objective / estimand issue |
| CR4   | s9        | s9|Q05@C|FULL|i307|s31|S1       |   0.828 | 0.789 |      0.396 | False        | objective / estimand issue |
| CR4   | s9        | s9|Q05@C|FULL|i401|s31|S1       |   0.827 | 0.788 |      0.396 | False        | objective / estimand issue |
| CR4   | s9        | s9|Q05@C|FULL|i503|s31|S1       |   0.828 | 0.787 |      0.404 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i307|s31|S1     |   0.829 | 0.802 |      0.348 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i401|s31|S1     |   0.827 | 0.801 |      0.351 | False        | objective / estimand issue |
| CR4   | s9_2      | s9_2|Q05@C|FULL|i503|s31|S1     |   0.829 | 0.801 |      0.353 | False        | objective / estimand issue |
| CR5   | s9_2      | s9_2|Q05@C|MASK-ALL|i31|s31|S1  |   0.820 | 0.878 |      0.148 | True         | recovered                  |
| CR5   | s9_2      | s9_2|Q05@C|MASK-ALL|i101|s31|S1 |   0.820 | 0.879 |      0.142 | True         | recovered                  |
| CR5   | s9        | s9|Q05@C|MASK-ALL|i307|s31|S1   |   0.823 | 0.878 |      0.175 | True         | recovered                  |
| CR5   | s9_2      | s9_2|Q05@C|MASK-ALL|i307|s31|S1 |   0.824 | 0.880 |      0.126 | True         | recovered                  |
| CR5   | s9_2      | s9_2|Q05@C|MASK-ALL|i503|s31|S1 |   0.824 | 0.881 |      0.124 | True         | recovered                  |
| CR3L  | s9_2      | s9_2|Q05@C|MASK-ALL|i31|s31|S1  |   0.820 | 0.950 |      0.221 | True         | recovered                  |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i31|s31|S1      |   0.828 | 0.779 |      0.526 | False        | objective / estimand issue |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i73|s31|S1      |   0.827 | 0.863 |      0.373 | False        | objective / estimand issue |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i101|s31|S1     |   0.825 | 0.883 |      0.316 | False        | objective / estimand issue |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i307|s31|S1     |   0.828 | 0.884 |      0.311 | False        | objective / estimand issue |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i401|s31|S1     |   0.833 | 0.882 |      0.316 | False        | objective / estimand issue |
| CR4L  | s9_2      | s9_2|Q05@C|FULL|i503|s31|S1     |   0.826 | 0.878 |      0.330 | False        | objective / estimand issue |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i31|s31|S1    |   0.831 | 0.952 |      0.164 | True         | recovered                  |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i73|s31|S1    |   0.826 | 0.953 |      0.144 | True         | recovered                  |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i101|s31|S1   |   0.824 | 0.947 |      0.209 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i31|s31|S1  |   0.831 | 0.925 |      0.266 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i73|s31|S1  |   0.826 | 0.936 |      0.227 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i101|s31|S1 |   0.822 | 0.918 |      0.292 | True         | recovered                  |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i307|s31|S1   |   0.827 | 0.954 |      0.123 | True         | recovered                  |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i401|s31|S1   |   0.826 | 0.950 |      0.186 | True         | recovered                  |
| CR5L  | s9        | s9|Q05@C|MASK-ALL|i503|s31|S1   |   0.819 | 0.949 |      0.186 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i307|s31|S1 |   0.827 | 0.936 |      0.226 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i401|s31|S1 |   0.825 | 0.926 |      0.266 | True         | recovered                  |
| CR5L  | s9_2      | s9_2|Q05@C|MASK-ALL|i503|s31|S1 |   0.817 | 0.926 |      0.264 | True         | recovered                  |


## REP-4 independent D re-evaluation of every REP-1/2 cell (rank stability; dL vs R* on identical / independent caches)

| batch   | variant   |   cells |   escalated |   D211_best_sel |   D613_best_sel |   not_conv |   sep_minima |   seed_incons |   C_below_star |   D211_below_star |   D613_below_star |
|:--------|:----------|--------:|------------:|----------------:|----------------:|-----------:|-------------:|--------------:|---------------:|------------------:|------------------:|
| REP-1   | s9        |     108 |          27 |           0.944 |           0.944 |         10 |            9 |            27 |            108 |               108 |               108 |
| REP-1   | s9_2      |     108 |          34 |           0.991 |           0.944 |         25 |           18 |            18 |            108 |               108 |               108 |
| REP-2   | s9        |     108 |          22 |           1.000 |           1.000 |         16 |            9 |             9 |            108 |               108 |               108 |
| REP-2   | s9_2      |     108 |          33 |           1.000 |           0.944 |         25 |           21 |             0 |            108 |               108 |               108 |


## Paired L(R_hat) - L(R*) by toy (median over the 18 REP-1/2 cells; negative = found rotation preferred to R*)

| toy   | variant   |        dC |     dD211 |     dD613 |    dC_min |    dC_max |
|:------|:----------|----------:|----------:|----------:|----------:|----------:|
| C1    | s9        | -7.70e-04 | -7.30e-04 | -7.16e-04 | -1.17e-03 | -6.25e-04 |
| C1    | s9_2      | -6.28e-04 | -6.06e-04 | -5.96e-04 | -9.96e-04 | -4.81e-04 |
| C1L   | s9        | -2.66e-03 | -2.79e-03 | -2.87e-03 | -2.98e-03 | -2.48e-03 |
| C1L   | s9_2      | -2.55e-03 | -2.69e-03 | -2.78e-03 | -2.89e-03 | -2.39e-03 |
| C2    | s9        | -2.09e-03 | -1.81e-03 | -1.73e-03 | -2.47e-03 | -1.86e-03 |
| C2    | s9_2      | -2.90e-03 | -2.58e-03 | -2.53e-03 | -3.15e-03 | -2.48e-03 |
| C2L   | s9        | -6.33e-03 | -6.48e-03 | -6.47e-03 | -7.00e-03 | -5.95e-03 |
| C2L   | s9_2      | -5.60e-03 | -5.72e-03 | -5.72e-03 | -6.28e-03 | -5.24e-03 |
| C3    | s9        | -1.97e-03 | -2.09e-03 | -2.01e-03 | -2.18e-03 | -1.86e-03 |
| C3    | s9_2      | -2.59e-03 | -2.73e-03 | -2.60e-03 | -2.80e-03 | -2.48e-03 |
| C3L   | s9        | -5.40e-03 | -5.48e-03 | -5.26e-03 | -5.93e-03 | -4.99e-03 |
| C3L   | s9_2      | -5.71e-03 | -5.82e-03 | -5.57e-03 | -6.22e-03 | -5.23e-03 |
| CR3   | s9        | -1.08e-02 | -1.07e-02 | -1.10e-02 | -1.12e-02 | -9.19e-03 |
| CR3   | s9_2      | -8.11e-03 | -7.84e-03 | -8.09e-03 | -8.42e-03 | -6.53e-03 |
| CR3L  | s9        | -9.30e-03 | -9.61e-03 | -9.00e-03 | -1.39e-02 | -8.72e-03 |
| CR3L  | s9_2      | -9.11e-03 | -9.68e-03 | -9.29e-03 | -1.33e-02 | -8.52e-03 |
| CR4   | s9        | -1.14e-02 | -1.13e-02 | -1.02e-02 | -1.28e-02 | -1.02e-02 |
| CR4   | s9_2      | -6.21e-03 | -6.13e-03 | -5.82e-03 | -6.85e-03 | -5.60e-03 |
| CR4L  | s9        | -6.05e-03 | -5.85e-03 | -6.69e-03 | -6.48e-03 | -4.28e-03 |
| CR4L  | s9_2      | -7.21e-03 | -7.06e-03 | -7.25e-03 | -8.41e-03 | -6.63e-03 |
| CR5   | s9        | -1.43e-02 | -1.42e-02 | -1.38e-02 | -1.78e-02 | -1.32e-02 |
| CR5   | s9_2      | -7.96e-03 | -7.86e-03 | -7.50e-03 | -1.08e-02 | -7.00e-03 |
| CR5L  | s9        | -2.90e-03 | -2.42e-03 | -3.08e-03 | -4.71e-03 | -1.81e-03 |
| CR5L  | s9_2      | -5.21e-03 | -4.80e-03 | -4.63e-03 | -6.36e-03 | -3.84e-03 |


## Single-deployment rules on the development REP-1 cells (fixed pair i31/s31 is the post-hoc FIN-SENS cell itself, hence biased)

| toy   | variant   | D_ranked_key   | D_ranked_pass   | fixed_pair_pass   |   all_run_pass |
|:------|:----------|:---------------|:----------------|:------------------|---------------:|
| C1    | s9        | i101|s73       | True            | True              |              9 |
| C1    | s9_2      | i101|s73       | True            | True              |              9 |
| C1L   | s9        | i31|s73        | True            | True              |              9 |
| C1L   | s9_2      | i31|s73        | True            | True              |              9 |
| C2    | s9        | i73|s73        | True            | True              |              9 |
| C2    | s9_2      | i73|s101       | True            | True              |              9 |
| C2L   | s9        | i101|s73       | True            | True              |              9 |
| C2L   | s9_2      | i101|s101      | True            | True              |              9 |
| C3    | s9        | i101|s101      | True            | True              |              9 |
| C3    | s9_2      | i101|s73       | True            | True              |              9 |
| C3L   | s9        | i101|s31       | True            | True              |              9 |
| C3L   | s9_2      | i101|s73       | True            | True              |              9 |
| CR3   | s9        | i31|s31        | True            | True              |              9 |
| CR3   | s9_2      | i31|s101       | True            | True              |              9 |
| CR3L  | s9        | i101|s73       | True            | True              |              9 |
| CR3L  | s9_2      | i101|s31       | True            | True              |              9 |
| CR4   | s9        | i101|s73       | False           | True              |              6 |
| CR4   | s9_2      | i31|s31        | False           | False             |              0 |
| CR4L  | s9        | i101|s101      | True            | True              |              9 |
| CR4L  | s9_2      | i101|s101      | False           | False             |              0 |
| CR5   | s9        | i73|s101       | True            | True              |              9 |
| CR5   | s9_2      | i73|s31        | True            | True              |              9 |
| CR5L  | s9        | i73|s101       | True            | True              |              9 |
| CR5L  | s9_2      | i73|s31        | True            | True              |              9 |

| variant   |   D_ranked_pass |   fixed_pair_pass |
|:----------|----------------:|------------------:|
| s9        |              11 |                12 |
| s9_2      |              10 |                10 |


## W4-02 controlled measure grid (3 integration x 3 search seeds, all 12 toys)

| Q          | variant   |   toys_all_pass |   toys |   runs_pass |   runs | failing (pass/runs)   |
|:-----------|:----------|----------------:|-------:|------------:|-------:|:----------------------|
| M-D (=Q05) | s9        |              11 |     12 |         105 |    108 | CR4(6/9)              |
| M-D (=Q05) | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)    |
| M-E        | s9        |              11 |     12 |          99 |    108 | CR4(0/9)              |
| M-E        | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)    |
| M-I        | s9        |              11 |     12 |          99 |    108 | CR4(0/9)              |
| M-I        | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)    |
| M-J        | s9        |              11 |     12 |         105 |    108 | CR4(6/9)              |
| M-J        | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)    |
| M-P (=Q01) | s9        |              11 |     12 |          99 |    108 | CR4(0/9)              |
| M-P (=Q01) | s9_2      |              10 |     12 |          90 |    108 | CR4(0/9) CR4L(0/9)    |


## W4-02 covariance / scale / truncation / subpool / mixture sensitivity (s9)

| batch   | Q             |   toys_all_pass |   toys |   runs_pass |   runs | failing (pass/runs)   |
|:--------|:--------------|----------------:|-------:|------------:|-------:|:----------------------|
| MIX     | MIX-0.25      |              11 |     12 |         102 |    108 | CR4(3/9)              |
| MIX     | MIX-0.5       |              11 |     12 |         105 |    108 | CR4(6/9)              |
| MIX     | MIX-0.75      |              11 |     12 |         105 |    108 | CR4(6/9)              |
| MS      | KAP-0.75      |              12 |     12 |          36 |     36 | -                     |
| MS      | KAP-1.25      |              12 |     12 |          36 |     36 | -                     |
| MS      | RHO-0.25      |              12 |     12 |          36 |     36 | -                     |
| MS      | RHO-0.5       |              12 |     12 |          36 |     36 | -                     |
| MS      | RHO-0.75      |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.25-1-XC |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.25-1-XF |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.25-2-XC |              11 |     12 |          35 |     36 | CR4(2/3)              |
| MS      | SUB-0.25-2-XF |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.25-3-XC |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.25-3-XF |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.5-1-XC  |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.5-1-XF  |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.5-2-XC  |              11 |     12 |          35 |     36 | CR4(2/3)              |
| MS      | SUB-0.5-2-XF  |              12 |     12 |          36 |     36 | -                     |
| MS      | SUB-0.5-3-XC  |              11 |     12 |          33 |     36 | CR4(0/3)              |
| MS      | SUB-0.5-3-XF  |              12 |     12 |          36 |     36 | -                     |
| MS      | TRN-2         |              11 |     12 |          33 |     36 | CR4(0/3)              |
| MS      | TRN-3         |              11 |     12 |          33 |     36 | CR4(0/3)              |


## W4-04 robustness: other measures, masks (primary panel)

| batch   | Q   | mask     | variant   |   toys_all_pass |   toys |   runs_pass |   runs | failing (pass/runs)   |
|:--------|:----|:---------|:----------|----------------:|-------:|------------:|-------:|:----------------------|
| RB-MASK | Q05 | FULL     | s9        |               4 |      4 |          36 |     36 | -                     |
| RB-MASK | Q05 | FULL     | s9_2      |               3 |      4 |          30 |     36 | CR5L(3/9)             |
| RB-MASK | Q05 | MASK-COL | s9        |               4 |      4 |          36 |     36 | -                     |
| RB-MASK | Q05 | MASK-COL | s9_2      |               3 |      4 |          30 |     36 | CR5L(3/9)             |
| RB-Q    | Q02 | FULL     | s9        |               7 |      8 |          63 |     72 | CR4(0/9)              |
| RB-Q    | Q02 | FULL     | s9_2      |               6 |      8 |          54 |     72 | CR4(0/9) CR4L(0/9)    |
| RB-Q    | Q02 | MASK-ALL | s9        |               4 |      4 |          36 |     36 | -                     |
| RB-Q    | Q02 | MASK-ALL | s9_2      |               4 |      4 |          36 |     36 | -                     |
| RB-Q    | Q03 | FULL     | s9        |               7 |      8 |          66 |     72 | CR4(3/9)              |
| RB-Q    | Q03 | FULL     | s9_2      |               6 |      8 |          54 |     72 | CR4(0/9) CR4L(0/9)    |
| RB-Q    | Q03 | MASK-ALL | s9        |               4 |      4 |          36 |     36 | -                     |
| RB-Q    | Q03 | MASK-ALL | s9_2      |               4 |      4 |          36 |     36 | -                     |
| RB-Q    | Q04 | MASK-ALL | s9        |               4 |      4 |          36 |     36 | -                     |
| RB-Q    | Q04 | MASK-ALL | s9_2      |               4 |      4 |          36 |     36 | -                     |


## W4-04 secondary saved NPs (5 toys; alternative beta is not a training-seed replicate)

| cid              | Q   | variant   |   toys_all_pass |   toys |   runs_pass |   runs | failing (pass/runs)        |
|:-----------------|:----|:----------|----------------:|-------:|------------:|-------:|:---------------------------|
| MX-B08c03@AD     | Q01 | s9        |               3 |      5 |          27 |     45 | CR3#b5e-5(0/9) CR4#r2(0/9) |
| MX-B08c03@AD     | Q01 | s9_2      |               0 |      1 |           0 |      9 | CR3#b5e-5(0/9)             |
| RW-GL-l0.75-a1.0 | Q01 | s9        |               5 |      5 |          45 |     45 | -                          |
| RW-GL-l0.75-a1.0 | Q01 | s9_2      |               1 |      1 |           9 |      9 | -                          |
| RW-GL-l0.75-a1.0 | Q05 | s9        |               5 |      5 |          45 |     45 | -                          |
| RW-GL-l0.75-a1.0 | Q05 | s9_2      |               1 |      1 |           9 |      9 | -                          |


## Support diagnostics (seed 31 C TRAIN laws vs empirical Z)

| toy   | law                    |    n |   mean_abs_std_mean |   sd_ratio_mean |   skew_max |   exkurt_max |   exkurt_mean |   mean_abs_corr |   max_abs_corr |   eig_ratio |   frac_out_1_99 |   frac_out_05_995 |   frac_maha_gt_val99 |   frac_nn_gt_val99 |   frac_tail2 |   frac_tail3 |
|:------|:-----------------------|-----:|--------------------:|----------------:|-----------:|-------------:|--------------:|----------------:|---------------:|------------:|----------------:|------------------:|---------------------:|-------------------:|-------------:|-------------:|
| C1    | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.059 |        0.345 |        -0.291 |           0.017 |          0.036 |       1.104 |           0.080 |             0.040 |                0.009 |              0.006 |        0.163 |        0.004 |
| C1    | empirical VALIDATION Z | 1200 |               0.017 |           1.002 |      0.114 |        0.367 |        -0.298 |           0.018 |          0.045 |       1.145 |           0.090 |             0.039 |                0.010 |              0.010 |        0.165 |        0.003 |
| C1    | Q01                    |  512 |               0.030 |           1.025 |      0.059 |        0.429 |        -0.284 |           0.036 |          0.075 |       1.235 |           0.104 |             0.055 |                0.012 |              0.008 |        0.180 |        0.006 |
| C1    | Q02                    |  512 |               0.030 |           1.025 |      0.059 |        0.429 |        -0.284 |           0.036 |          0.075 |       1.235 |           0.104 |             0.055 |                0.012 |              0.008 |        0.180 |        0.006 |
| C1    | Q03                    |  512 |               0.029 |           1.000 |      0.249 |        0.343 |        -0.043 |           0.038 |          0.071 |       1.285 |           0.102 |             0.064 |                0.027 |              0.033 |        0.156 |        0.014 |
| C1    | M-E                    |  512 |               0.030 |           1.025 |      0.059 |        0.429 |        -0.284 |           0.036 |          0.075 |       1.235 |           0.104 |             0.055 |                0.012 |              0.008 |        0.180 |        0.006 |
| C1    | M-J                    |  512 |               0.031 |           1.020 |      0.111 |        0.358 |        -0.054 |           0.032 |          0.056 |       1.188 |           0.117 |             0.072 |                0.033 |              0.023 |        0.199 |        0.010 |
| C1    | M-D                    |  512 |               0.031 |           1.020 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.184 |           0.115 |             0.068 |                0.029 |              0.021 |        0.199 |        0.008 |
| C1    | M-I                    |  512 |               0.045 |           0.997 |      0.161 |        0.475 |        -0.318 |           0.038 |          0.079 |       1.305 |           0.070 |             0.033 |                0.016 |              0.014 |        0.150 |        0.004 |
| C1    | RHO-0.25               |  512 |               0.031 |           1.020 |      0.112 |        0.359 |        -0.057 |           0.028 |          0.063 |       1.179 |           0.115 |             0.068 |                0.029 |              0.021 |        0.199 |        0.008 |
| C1    | RHO-0.5                |  512 |               0.031 |           1.020 |      0.111 |        0.359 |        -0.056 |           0.029 |          0.060 |       1.178 |           0.117 |             0.068 |                0.029 |              0.021 |        0.199 |        0.010 |
| C1    | RHO-0.75               |  512 |               0.031 |           1.020 |      0.111 |        0.359 |        -0.055 |           0.030 |          0.058 |       1.181 |           0.117 |             0.070 |                0.031 |              0.023 |        0.201 |        0.010 |
| C1    | KAP-0.75               |  512 |               0.023 |           0.765 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.184 |           0.010 |             0.004 |                0.000 |              0.000 |        0.035 |        0.000 |
| C1    | KAP-1.25               |  512 |               0.039 |           1.275 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.184 |           0.303 |             0.217 |                0.172 |              0.127 |        0.395 |        0.080 |
| C1    | TRN-2                  |  512 |               0.025 |           0.888 |      0.170 |        0.776 |        -0.615 |           0.038 |          0.080 |       1.271 |           0.000 |             0.000 |                0.002 |              0.000 |        0.000 |        0.000 |
| C1    | TRN-3                  |  512 |               0.024 |           0.975 |      0.238 |        0.380 |        -0.218 |           0.025 |          0.042 |       1.180 |           0.078 |             0.043 |                0.021 |              0.016 |        0.143 |        0.000 |
| C1L   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.026 |        0.051 |        -0.007 |           0.010 |          0.029 |       1.073 |           0.078 |             0.039 |                0.011 |              0.008 |        0.172 |        0.011 |
| C1L   | empirical VALIDATION Z | 1200 |               0.020 |           1.003 |      0.117 |        0.298 |         0.054 |           0.040 |          0.092 |       1.275 |           0.096 |             0.054 |                0.010 |              0.010 |        0.179 |        0.011 |
| C1L   | Q01                    |  512 |               0.025 |           1.021 |      0.152 |        0.367 |        -0.069 |           0.033 |          0.056 |       1.190 |           0.092 |             0.041 |                0.014 |              0.010 |        0.166 |        0.008 |
| C1L   | Q02                    |  512 |               0.025 |           1.021 |      0.152 |        0.367 |        -0.069 |           0.033 |          0.056 |       1.190 |           0.092 |             0.041 |                0.014 |              0.010 |        0.166 |        0.008 |
| C1L   | Q03                    |  512 |               0.029 |           1.000 |      0.251 |        0.343 |        -0.033 |           0.045 |          0.061 |       1.293 |           0.082 |             0.053 |                0.008 |              0.014 |        0.158 |        0.016 |
| C1L   | M-E                    |  512 |               0.025 |           1.021 |      0.152 |        0.367 |        -0.069 |           0.033 |          0.056 |       1.190 |           0.092 |             0.041 |                0.014 |              0.010 |        0.166 |        0.008 |
| C1L   | M-J                    |  512 |               0.031 |           1.020 |      0.110 |        0.360 |        -0.058 |           0.024 |          0.071 |       1.208 |           0.096 |             0.047 |                0.008 |              0.006 |        0.197 |        0.008 |
| C1L   | M-D                    |  512 |               0.031 |           1.020 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.217 |           0.096 |             0.047 |                0.004 |              0.006 |        0.199 |        0.008 |
| C1L   | M-I                    |  512 |               0.010 |           0.987 |      0.181 |        0.396 |         0.059 |           0.047 |          0.081 |       1.265 |           0.072 |             0.043 |                0.012 |              0.008 |        0.166 |        0.012 |
| C1L   | RHO-0.25               |  512 |               0.031 |           1.020 |      0.112 |        0.360 |        -0.058 |           0.027 |          0.066 |       1.213 |           0.096 |             0.047 |                0.004 |              0.006 |        0.199 |        0.008 |
| C1L   | RHO-0.5                |  512 |               0.031 |           1.020 |      0.111 |        0.360 |        -0.058 |           0.026 |          0.068 |       1.210 |           0.096 |             0.047 |                0.006 |              0.006 |        0.197 |        0.008 |
| C1L   | RHO-0.75               |  512 |               0.031 |           1.020 |      0.111 |        0.360 |        -0.058 |           0.024 |          0.069 |       1.208 |           0.096 |             0.047 |                0.008 |              0.006 |        0.197 |        0.008 |
| C1L   | KAP-0.75               |  512 |               0.023 |           0.765 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.217 |           0.004 |             0.004 |                0.000 |              0.000 |        0.035 |        0.000 |
| C1L   | KAP-1.25               |  512 |               0.039 |           1.275 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.217 |           0.268 |             0.188 |                0.086 |              0.115 |        0.395 |        0.080 |
| C1L   | TRN-2                  |  512 |               0.025 |           0.888 |      0.170 |        0.776 |        -0.615 |           0.038 |          0.080 |       1.273 |           0.000 |             0.000 |                0.000 |              0.000 |        0.000 |        0.000 |
| C1L   | TRN-3                  |  512 |               0.024 |           0.975 |      0.238 |        0.380 |        -0.218 |           0.025 |          0.042 |       1.200 |           0.057 |             0.025 |                0.006 |              0.010 |        0.143 |        0.000 |
| C2    | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.018 |        0.552 |        -0.407 |           0.038 |          0.081 |       1.259 |           0.080 |             0.040 |                0.012 |              0.006 |        0.151 |        0.001 |
| C2    | empirical VALIDATION Z | 1200 |               0.021 |           1.000 |      0.135 |        0.567 |        -0.358 |           0.049 |          0.078 |       1.285 |           0.100 |             0.045 |                0.010 |              0.010 |        0.171 |        0.000 |
| C2    | Q01                    |  512 |               0.027 |           1.020 |      0.118 |        0.713 |        -0.433 |           0.041 |          0.073 |       1.298 |           0.098 |             0.051 |                0.029 |              0.008 |        0.160 |        0.000 |
| C2    | Q02                    |  512 |               0.027 |           1.020 |      0.118 |        0.713 |        -0.433 |           0.041 |          0.073 |       1.298 |           0.098 |             0.051 |                0.029 |              0.008 |        0.160 |        0.000 |
| C2    | Q03                    |  512 |               0.029 |           1.002 |      0.245 |        0.343 |        -0.048 |           0.060 |          0.128 |       1.507 |           0.102 |             0.072 |                0.041 |              0.037 |        0.164 |        0.010 |
| C2    | M-E                    |  512 |               0.027 |           1.020 |      0.118 |        0.713 |        -0.433 |           0.041 |          0.073 |       1.298 |           0.098 |             0.051 |                0.029 |              0.008 |        0.160 |        0.000 |
| C2    | M-J                    |  512 |               0.032 |           1.019 |      0.117 |        0.358 |        -0.057 |           0.044 |          0.076 |       1.275 |           0.135 |             0.080 |                0.055 |              0.039 |        0.189 |        0.008 |
| C2    | M-D                    |  512 |               0.031 |           1.020 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.190 |           0.141 |             0.078 |                0.057 |              0.035 |        0.199 |        0.008 |
| C2    | M-I                    |  512 |               0.020 |           0.999 |      0.153 |        0.391 |        -0.265 |           0.021 |          0.059 |       1.164 |           0.094 |             0.061 |                0.035 |              0.031 |        0.166 |        0.006 |
| C2    | RHO-0.25               |  512 |               0.031 |           1.020 |      0.113 |        0.359 |        -0.058 |           0.026 |          0.061 |       1.178 |           0.141 |             0.080 |                0.057 |              0.035 |        0.193 |        0.008 |
| C2    | RHO-0.5                |  512 |               0.031 |           1.020 |      0.115 |        0.359 |        -0.058 |           0.029 |          0.057 |       1.187 |           0.139 |             0.080 |                0.053 |              0.037 |        0.191 |        0.008 |
| C2    | RHO-0.75               |  512 |               0.032 |           1.019 |      0.116 |        0.359 |        -0.057 |           0.036 |          0.064 |       1.225 |           0.141 |             0.080 |                0.055 |              0.035 |        0.188 |        0.008 |
| C2    | KAP-0.75               |  512 |               0.023 |           0.765 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.190 |           0.012 |             0.006 |                0.000 |              0.000 |        0.035 |        0.000 |
| C2    | KAP-1.25               |  512 |               0.039 |           1.275 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.190 |           0.314 |             0.242 |                0.213 |              0.182 |        0.395 |        0.080 |
| C2    | TRN-2                  |  512 |               0.025 |           0.888 |      0.170 |        0.776 |        -0.615 |           0.038 |          0.080 |       1.307 |           0.000 |             0.000 |                0.004 |              0.004 |        0.000 |        0.000 |
| C2    | TRN-3                  |  512 |               0.024 |           0.975 |      0.238 |        0.380 |        -0.218 |           0.025 |          0.042 |       1.151 |           0.090 |             0.051 |                0.027 |              0.029 |        0.143 |        0.000 |
| C2L   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.013 |        0.062 |         0.017 |           0.016 |          0.033 |       1.143 |           0.077 |             0.039 |                0.011 |              0.007 |        0.172 |        0.012 |
| C2L   | empirical VALIDATION Z | 1200 |               0.021 |           1.002 |      0.129 |        0.198 |         0.052 |           0.034 |          0.059 |       1.215 |           0.092 |             0.037 |                0.010 |              0.010 |        0.179 |        0.010 |
| C2L   | Q01                    |  512 |               0.024 |           1.022 |      0.144 |        0.556 |         0.210 |           0.030 |          0.055 |       1.278 |           0.096 |             0.053 |                0.012 |              0.010 |        0.191 |        0.023 |
| C2L   | Q02                    |  512 |               0.024 |           1.022 |      0.144 |        0.556 |         0.210 |           0.030 |          0.055 |       1.278 |           0.096 |             0.053 |                0.012 |              0.010 |        0.191 |        0.023 |
| C2L   | Q03                    |  512 |               0.029 |           1.000 |      0.250 |        0.343 |        -0.039 |           0.033 |          0.064 |       1.267 |           0.078 |             0.053 |                0.008 |              0.010 |        0.156 |        0.014 |
| C2L   | M-E                    |  512 |               0.024 |           1.022 |      0.144 |        0.556 |         0.210 |           0.030 |          0.055 |       1.278 |           0.096 |             0.053 |                0.012 |              0.010 |        0.191 |        0.023 |
| C2L   | M-J                    |  512 |               0.031 |           1.020 |      0.111 |        0.363 |        -0.056 |           0.035 |          0.092 |       1.333 |           0.098 |             0.047 |                0.008 |              0.012 |        0.201 |        0.010 |
| C2L   | M-D                    |  512 |               0.031 |           1.020 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.305 |           0.098 |             0.049 |                0.008 |              0.010 |        0.199 |        0.008 |
| C2L   | M-I                    |  512 |               0.020 |           0.998 |      0.118 |        0.436 |        -0.164 |           0.037 |          0.069 |       1.270 |           0.068 |             0.031 |                0.008 |              0.012 |        0.168 |        0.006 |
| C2L   | RHO-0.25               |  512 |               0.031 |           1.020 |      0.112 |        0.361 |        -0.057 |           0.030 |          0.072 |       1.308 |           0.098 |             0.047 |                0.008 |              0.010 |        0.201 |        0.008 |
| C2L   | RHO-0.5                |  512 |               0.031 |           1.020 |      0.112 |        0.361 |        -0.057 |           0.031 |          0.079 |       1.313 |           0.098 |             0.047 |                0.008 |              0.010 |        0.201 |        0.010 |
| C2L   | RHO-0.75               |  512 |               0.031 |           1.020 |      0.111 |        0.362 |        -0.056 |           0.032 |          0.085 |       1.322 |           0.098 |             0.047 |                0.008 |              0.012 |        0.199 |        0.010 |
| C2L   | KAP-0.75               |  512 |               0.023 |           0.765 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.305 |           0.004 |             0.004 |                0.000 |              0.000 |        0.035 |        0.000 |
| C2L   | KAP-1.25               |  512 |               0.039 |           1.275 |      0.112 |        0.360 |        -0.058 |           0.029 |          0.065 |       1.305 |           0.262 |             0.182 |                0.096 |              0.078 |        0.395 |        0.080 |
| C2L   | TRN-2                  |  512 |               0.025 |           0.888 |      0.170 |        0.776 |        -0.615 |           0.038 |          0.080 |       1.327 |           0.000 |             0.000 |                0.000 |              0.000 |        0.000 |        0.000 |
| C2L   | TRN-3                  |  512 |               0.024 |           0.975 |      0.238 |        0.380 |        -0.218 |           0.025 |          0.042 |       1.257 |           0.059 |             0.020 |                0.006 |              0.008 |        0.143 |        0.000 |
| C3    | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.082 |        0.498 |        -0.385 |           0.014 |          0.053 |       1.187 |           0.119 |             0.060 |                0.013 |              0.006 |        0.224 |        0.004 |
| C3    | empirical VALIDATION Z | 1200 |               0.025 |           0.999 |      0.130 |        0.552 |        -0.396 |           0.028 |          0.071 |       1.302 |           0.116 |             0.054 |                0.010 |              0.010 |        0.225 |        0.005 |
| C3    | Q01                    |  512 |               0.016 |           0.997 |      0.317 |        0.549 |        -0.362 |           0.038 |          0.106 |       1.610 |           0.117 |             0.059 |                0.016 |              0.006 |        0.205 |        0.008 |
| C3    | Q02                    |  512 |               0.016 |           0.997 |      0.317 |        0.549 |        -0.362 |           0.038 |          0.106 |       1.610 |           0.117 |             0.059 |                0.016 |              0.006 |        0.205 |        0.008 |
| C3    | Q03                    |  512 |               0.030 |           1.001 |      0.262 |        0.162 |        -0.009 |           0.049 |          0.097 |       1.547 |           0.148 |             0.105 |                0.078 |              0.068 |        0.240 |        0.020 |
| C3    | M-E                    |  512 |               0.016 |           0.997 |      0.317 |        0.549 |        -0.362 |           0.038 |          0.106 |       1.610 |           0.117 |             0.059 |                0.016 |              0.006 |        0.205 |        0.008 |
| C3    | M-J                    |  512 |               0.042 |           1.013 |      0.161 |        0.528 |        -0.110 |           0.044 |          0.087 |       1.373 |           0.170 |             0.102 |                0.066 |              0.074 |        0.262 |        0.014 |
| C3    | M-D                    |  512 |               0.041 |           1.013 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.362 |           0.168 |             0.105 |                0.066 |              0.076 |        0.262 |        0.012 |
| C3    | M-I                    |  512 |               0.023 |           0.981 |      0.233 |        0.561 |        -0.390 |           0.045 |          0.106 |       1.431 |           0.105 |             0.057 |                0.041 |              0.041 |        0.182 |        0.002 |
| C3    | RHO-0.25               |  512 |               0.041 |           1.013 |      0.163 |        0.525 |        -0.110 |           0.041 |          0.077 |       1.360 |           0.168 |             0.105 |                0.066 |              0.076 |        0.264 |        0.014 |
| C3    | RHO-0.5                |  512 |               0.041 |           1.013 |      0.162 |        0.526 |        -0.110 |           0.042 |          0.080 |       1.360 |           0.168 |             0.105 |                0.064 |              0.078 |        0.264 |        0.014 |
| C3    | RHO-0.75               |  512 |               0.042 |           1.013 |      0.161 |        0.527 |        -0.110 |           0.042 |          0.084 |       1.364 |           0.170 |             0.104 |                0.066 |              0.078 |        0.264 |        0.014 |
| C3    | KAP-0.75               |  512 |               0.031 |           0.760 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.362 |           0.016 |             0.006 |                0.000 |              0.000 |        0.045 |        0.000 |
| C3    | KAP-1.25               |  512 |               0.051 |           1.266 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.362 |           0.422 |             0.307 |                0.309 |              0.264 |        0.527 |        0.096 |
| C3    | TRN-2                  |  512 |               0.019 |           0.881 |      0.158 |        0.680 |        -0.632 |           0.038 |          0.081 |       1.487 |           0.000 |             0.000 |                0.002 |              0.006 |        0.000 |        0.000 |
| C3    | TRN-3                  |  512 |               0.023 |           0.979 |      0.255 |        0.354 |        -0.213 |           0.032 |          0.059 |       1.365 |           0.129 |             0.068 |                0.051 |              0.043 |        0.215 |        0.000 |
| C3L   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.064 |        0.189 |        -0.019 |           0.012 |          0.030 |       1.138 |           0.113 |             0.059 |                0.011 |              0.011 |        0.245 |        0.015 |
| C3L   | empirical VALIDATION Z | 1200 |               0.021 |           1.001 |      0.185 |        0.290 |        -0.029 |           0.025 |          0.081 |       1.311 |           0.111 |             0.058 |                0.010 |              0.010 |        0.251 |        0.018 |
| C3L   | Q01                    |  512 |               0.024 |           1.001 |      0.425 |        0.589 |         0.173 |           0.043 |          0.117 |       1.620 |           0.125 |             0.072 |                0.014 |              0.006 |        0.225 |        0.023 |
| C3L   | Q02                    |  512 |               0.024 |           1.001 |      0.425 |        0.589 |         0.173 |           0.043 |          0.117 |       1.620 |           0.125 |             0.072 |                0.014 |              0.006 |        0.225 |        0.023 |
| C3L   | Q03                    |  512 |               0.029 |           1.001 |      0.255 |        0.167 |        -0.014 |           0.043 |          0.089 |       1.498 |           0.115 |             0.082 |                0.006 |              0.010 |        0.230 |        0.020 |
| C3L   | M-E                    |  512 |               0.024 |           1.001 |      0.425 |        0.589 |         0.173 |           0.043 |          0.117 |       1.620 |           0.125 |             0.072 |                0.014 |              0.006 |        0.225 |        0.023 |
| C3L   | M-J                    |  512 |               0.042 |           1.013 |      0.166 |        0.536 |        -0.117 |           0.041 |          0.084 |       1.345 |           0.119 |             0.059 |                0.010 |              0.020 |        0.260 |        0.012 |
| C3L   | M-D                    |  512 |               0.041 |           1.013 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.342 |           0.119 |             0.061 |                0.010 |              0.021 |        0.262 |        0.012 |
| C3L   | M-I                    |  512 |               0.029 |           0.990 |      0.179 |        0.265 |         0.040 |           0.036 |          0.106 |       1.323 |           0.127 |             0.070 |                0.010 |              0.014 |        0.266 |        0.014 |
| C3L   | RHO-0.25               |  512 |               0.041 |           1.013 |      0.165 |        0.527 |        -0.112 |           0.040 |          0.072 |       1.338 |           0.119 |             0.061 |                0.010 |              0.021 |        0.262 |        0.012 |
| C3L   | RHO-0.5                |  512 |               0.041 |           1.013 |      0.165 |        0.530 |        -0.114 |           0.040 |          0.076 |       1.337 |           0.119 |             0.061 |                0.010 |              0.021 |        0.262 |        0.012 |
| C3L   | RHO-0.75               |  512 |               0.041 |           1.013 |      0.165 |        0.533 |        -0.115 |           0.041 |          0.080 |       1.339 |           0.119 |             0.059 |                0.010 |              0.020 |        0.260 |        0.012 |
| C3L   | KAP-0.75               |  512 |               0.031 |           0.760 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.342 |           0.006 |             0.004 |                0.000 |              0.000 |        0.045 |        0.000 |
| C3L   | KAP-1.25               |  512 |               0.051 |           1.266 |      0.164 |        0.524 |        -0.111 |           0.041 |          0.074 |       1.342 |           0.363 |             0.236 |                0.100 |              0.102 |        0.527 |        0.096 |
| C3L   | TRN-2                  |  512 |               0.019 |           0.881 |      0.158 |        0.680 |        -0.632 |           0.038 |          0.081 |       1.458 |           0.000 |             0.000 |                0.000 |              0.000 |        0.000 |        0.000 |
| C3L   | TRN-3                  |  512 |               0.023 |           0.979 |      0.255 |        0.354 |        -0.213 |           0.032 |          0.059 |       1.343 |           0.084 |             0.043 |                0.010 |              0.000 |        0.215 |        0.000 |
| CR3   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.265 |        0.253 |        -0.017 |           0.019 |          0.051 |       1.519 |           0.076 |             0.039 |                0.008 |              0.003 |        0.164 |        0.011 |
| CR3   | empirical VALIDATION Z | 1200 |               0.019 |           1.003 |      0.175 |        0.612 |         0.094 |           0.027 |          0.042 |       1.634 |           0.086 |             0.047 |                0.010 |              0.010 |        0.172 |        0.017 |
| CR3   | Q01                    |  512 |               0.043 |           0.976 |      0.260 |        0.276 |         0.014 |           0.052 |          0.089 |       1.655 |           0.064 |             0.041 |                0.008 |              0.004 |        0.139 |        0.012 |
| CR3   | Q02                    |  128 |               0.101 |           0.956 |      0.320 |        0.432 |        -0.319 |           0.099 |          0.183 |       2.716 |           0.055 |             0.047 |                0.000 |              0.000 |        0.109 |        0.000 |
| CR3   | Q03                    |  512 |               0.030 |           1.000 |      0.250 |        0.343 |        -0.045 |           0.052 |          0.070 |       1.824 |           0.082 |             0.057 |                0.002 |              0.029 |        0.156 |        0.014 |
| CR3   | M-E                    |  128 |               0.101 |           0.956 |      0.320 |        0.432 |        -0.319 |           0.099 |          0.183 |       2.716 |           0.055 |             0.047 |                0.000 |              0.000 |        0.109 |        0.000 |
| CR3   | M-J                    |  128 |               0.111 |           1.001 |      0.076 |        0.369 |        -0.219 |           0.075 |          0.188 |       1.732 |           0.102 |             0.055 |                0.000 |              0.023 |        0.180 |        0.000 |
| CR3   | M-D                    |  128 |               0.112 |           1.001 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       1.710 |           0.102 |             0.055 |                0.000 |              0.023 |        0.180 |        0.000 |
| CR3   | M-I                    |  128 |               0.091 |           0.997 |      0.323 |        0.366 |        -0.107 |           0.068 |          0.172 |       2.592 |           0.086 |             0.047 |                0.000 |              0.016 |        0.203 |        0.008 |
| CR3   | RHO-0.25               |  128 |               0.112 |           1.001 |      0.076 |        0.365 |        -0.218 |           0.079 |          0.165 |       1.711 |           0.102 |             0.055 |                0.000 |              0.023 |        0.180 |        0.000 |
| CR3   | RHO-0.5                |  128 |               0.112 |           1.001 |      0.076 |        0.366 |        -0.219 |           0.078 |          0.173 |       1.715 |           0.102 |             0.055 |                0.000 |              0.023 |        0.180 |        0.000 |
| CR3   | RHO-0.75               |  128 |               0.111 |           1.001 |      0.076 |        0.367 |        -0.219 |           0.077 |          0.181 |       1.722 |           0.102 |             0.055 |                0.000 |              0.023 |        0.180 |        0.000 |
| CR3   | KAP-0.75               |  128 |               0.084 |           0.751 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       1.710 |           0.000 |             0.000 |                0.000 |              0.000 |        0.016 |        0.000 |
| CR3   | KAP-1.25               |  128 |               0.141 |           1.251 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       1.710 |           0.219 |             0.156 |                0.039 |              0.094 |        0.375 |        0.086 |
| CR3   | TRN-2                  |  128 |               0.027 |           0.868 |      0.220 |        0.709 |        -0.545 |           0.075 |          0.166 |       1.888 |           0.000 |             0.000 |                0.000 |              0.008 |        0.000 |        0.000 |
| CR3   | TRN-3                  |  128 |               0.077 |           0.988 |      0.077 |        0.429 |        -0.147 |           0.082 |          0.131 |       1.633 |           0.086 |             0.039 |                0.008 |              0.008 |        0.156 |        0.000 |
| CR3L  | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.488 |        0.377 |         0.153 |           0.020 |          0.065 |       1.482 |           0.077 |             0.039 |                0.013 |              0.005 |        0.170 |        0.016 |
| CR3L  | empirical VALIDATION Z | 1200 |               0.013 |           1.009 |      0.458 |        0.471 |         0.207 |           0.032 |          0.083 |       1.566 |           0.089 |             0.045 |                0.010 |              0.010 |        0.184 |        0.018 |
| CR3L  | Q01                    |  512 |               0.048 |           0.977 |      0.552 |        0.508 |         0.156 |           0.024 |          0.098 |       1.525 |           0.066 |             0.029 |                0.012 |              0.008 |        0.156 |        0.021 |
| CR3L  | Q02                    |  128 |               0.110 |           0.986 |      0.741 |        0.434 |        -0.099 |           0.100 |          0.236 |       1.954 |           0.062 |             0.016 |                0.008 |              0.008 |        0.180 |        0.016 |
| CR3L  | Q03                    |  512 |               0.029 |           0.999 |      0.251 |        0.343 |        -0.030 |           0.038 |          0.060 |       1.384 |           0.090 |             0.061 |                0.006 |              0.035 |        0.158 |        0.016 |
| CR3L  | M-E                    |  128 |               0.110 |           0.986 |      0.741 |        0.434 |        -0.099 |           0.100 |          0.236 |       1.954 |           0.062 |             0.016 |                0.008 |              0.008 |        0.180 |        0.016 |
| CR3L  | M-J                    |  128 |               0.114 |           0.997 |      0.112 |        0.363 |        -0.208 |           0.061 |          0.113 |       2.327 |           0.094 |             0.047 |                0.000 |              0.023 |        0.172 |        0.000 |
| CR3L  | M-D                    |  128 |               0.112 |           1.001 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       2.404 |           0.094 |             0.047 |                0.000 |              0.031 |        0.180 |        0.000 |
| CR3L  | M-I                    |  128 |               0.082 |           1.000 |      0.413 |        0.350 |         0.076 |           0.048 |          0.082 |       1.567 |           0.094 |             0.055 |                0.000 |              0.031 |        0.203 |        0.008 |
| CR3L  | RHO-0.25               |  128 |               0.113 |           1.000 |      0.078 |        0.363 |        -0.216 |           0.076 |          0.147 |       2.379 |           0.094 |             0.047 |                0.000 |              0.031 |        0.180 |        0.000 |
| CR3L  | RHO-0.5                |  128 |               0.113 |           0.999 |      0.089 |        0.363 |        -0.213 |           0.071 |          0.136 |       2.357 |           0.094 |             0.047 |                0.000 |              0.031 |        0.180 |        0.000 |
| CR3L  | RHO-0.75               |  128 |               0.113 |           0.998 |      0.100 |        0.363 |        -0.211 |           0.066 |          0.124 |       2.340 |           0.094 |             0.047 |                0.000 |              0.031 |        0.172 |        0.000 |
| CR3L  | KAP-0.75               |  128 |               0.084 |           0.751 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       2.404 |           0.000 |             0.000 |                0.000 |              0.000 |        0.016 |        0.000 |
| CR3L  | KAP-1.25               |  128 |               0.141 |           1.251 |      0.076 |        0.363 |        -0.218 |           0.081 |          0.158 |       2.404 |           0.234 |             0.156 |                0.062 |              0.109 |        0.375 |        0.086 |
| CR3L  | TRN-2                  |  128 |               0.027 |           0.868 |      0.220 |        0.709 |        -0.545 |           0.075 |          0.166 |       1.726 |           0.000 |             0.000 |                0.000 |              0.008 |        0.000 |        0.000 |
| CR3L  | TRN-3                  |  128 |               0.077 |           0.988 |      0.077 |        0.429 |        -0.147 |           0.082 |          0.131 |       2.030 |           0.070 |             0.031 |                0.008 |              0.016 |        0.156 |        0.000 |
| CR4   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.240 |        0.147 |         0.042 |           0.026 |          0.035 |       1.625 |           0.060 |             0.030 |                0.011 |              0.006 |        0.128 |        0.009 |
| CR4   | empirical VALIDATION Z | 1200 |               0.014 |           1.013 |      0.209 |        0.144 |         0.051 |           0.049 |          0.056 |       1.732 |           0.071 |             0.044 |                0.010 |              0.010 |        0.145 |        0.011 |
| CR4   | Q01                    |  512 |               0.034 |           0.977 |      0.075 |        0.369 |        -0.130 |           0.040 |          0.050 |       1.532 |           0.055 |             0.027 |                0.002 |              0.004 |        0.104 |        0.006 |
| CR4   | Q02                    |  512 |               0.034 |           0.977 |      0.075 |        0.369 |        -0.130 |           0.040 |          0.050 |       1.532 |           0.055 |             0.027 |                0.002 |              0.004 |        0.104 |        0.006 |
| CR4   | Q03                    |  512 |               0.026 |           1.006 |      0.105 |        0.020 |         0.008 |           0.025 |          0.030 |       1.739 |           0.072 |             0.051 |                0.004 |              0.018 |        0.131 |        0.012 |
| CR4   | M-E                    |  512 |               0.034 |           0.977 |      0.075 |        0.369 |        -0.130 |           0.040 |          0.050 |       1.532 |           0.055 |             0.027 |                0.002 |              0.004 |        0.104 |        0.006 |
| CR4   | M-J                    |  512 |               0.036 |           1.012 |      0.037 |        0.281 |        -0.035 |           0.066 |          0.103 |       1.806 |           0.074 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4   | M-D                    |  512 |               0.036 |           1.011 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.753 |           0.074 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4   | M-I                    |  512 |               0.035 |           0.997 |      0.210 |        0.617 |         0.153 |           0.017 |          0.045 |       1.685 |           0.062 |             0.031 |                0.012 |              0.025 |        0.123 |        0.010 |
| CR4   | RHO-0.25               |  512 |               0.036 |           1.011 |      0.040 |        0.291 |        -0.039 |           0.047 |          0.080 |       1.762 |           0.074 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4   | RHO-0.5                |  512 |               0.036 |           1.011 |      0.039 |        0.287 |        -0.038 |           0.053 |          0.088 |       1.775 |           0.074 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4   | RHO-0.75               |  512 |               0.036 |           1.012 |      0.038 |        0.284 |        -0.036 |           0.059 |          0.095 |       1.789 |           0.074 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4   | KAP-0.75               |  512 |               0.027 |           0.758 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.753 |           0.004 |             0.000 |                0.000 |              0.000 |        0.025 |        0.000 |
| CR4   | KAP-1.25               |  512 |               0.045 |           1.264 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.753 |           0.189 |             0.139 |                0.066 |              0.072 |        0.311 |        0.062 |
| CR4   | TRN-2                  |  512 |               0.013 |           0.884 |      0.048 |        0.630 |        -0.584 |           0.054 |          0.098 |       1.719 |           0.000 |             0.000 |                0.000 |              0.002 |        0.000 |        0.000 |
| CR4   | TRN-3                  |  512 |               0.056 |           0.977 |      0.059 |        0.342 |        -0.165 |           0.043 |          0.065 |       1.363 |           0.049 |             0.020 |                0.006 |              0.023 |        0.113 |        0.000 |
| CR4L  | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.415 |        0.388 |         0.306 |           0.020 |          0.053 |       1.821 |           0.060 |             0.030 |                0.010 |              0.004 |        0.135 |        0.013 |
| CR4L  | empirical VALIDATION Z | 1200 |               0.018 |           1.017 |      0.300 |        0.371 |         0.208 |           0.031 |          0.060 |       1.856 |           0.065 |             0.037 |                0.010 |              0.010 |        0.161 |        0.012 |
| CR4L  | Q01                    |  512 |               0.049 |           0.976 |      0.248 |        0.155 |         0.039 |           0.040 |          0.052 |       1.745 |           0.055 |             0.029 |                0.004 |              0.006 |        0.119 |        0.010 |
| CR4L  | Q02                    |  512 |               0.049 |           0.976 |      0.248 |        0.155 |         0.039 |           0.040 |          0.052 |       1.745 |           0.055 |             0.029 |                0.004 |              0.006 |        0.119 |        0.010 |
| CR4L  | Q03                    |  512 |               0.026 |           1.006 |      0.105 |        0.015 |         0.007 |           0.033 |          0.045 |       1.970 |           0.059 |             0.033 |                0.002 |              0.020 |        0.133 |        0.012 |
| CR4L  | M-E                    |  512 |               0.049 |           0.976 |      0.248 |        0.155 |         0.039 |           0.040 |          0.052 |       1.745 |           0.055 |             0.029 |                0.004 |              0.006 |        0.119 |        0.010 |
| CR4L  | M-J                    |  512 |               0.035 |           1.012 |      0.035 |        0.295 |        -0.041 |           0.060 |          0.125 |       2.020 |           0.072 |             0.039 |                0.002 |              0.012 |        0.148 |        0.004 |
| CR4L  | M-D                    |  512 |               0.036 |           1.011 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.963 |           0.070 |             0.039 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4L  | M-I                    |  512 |               0.046 |           1.015 |      0.470 |        0.617 |         0.351 |           0.017 |          0.030 |       1.951 |           0.061 |             0.027 |                0.014 |              0.021 |        0.137 |        0.016 |
| CR4L  | RHO-0.25               |  512 |               0.036 |           1.011 |      0.039 |        0.294 |        -0.040 |           0.048 |          0.086 |       1.974 |           0.070 |             0.041 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4L  | RHO-0.5                |  512 |               0.036 |           1.011 |      0.038 |        0.294 |        -0.040 |           0.052 |          0.099 |       1.987 |           0.070 |             0.039 |                0.002 |              0.014 |        0.148 |        0.004 |
| CR4L  | RHO-0.75               |  512 |               0.036 |           1.012 |      0.036 |        0.294 |        -0.041 |           0.056 |          0.112 |       2.003 |           0.072 |             0.039 |                0.002 |              0.012 |        0.148 |        0.004 |
| CR4L  | KAP-0.75               |  512 |               0.027 |           0.758 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.963 |           0.008 |             0.000 |                0.000 |              0.000 |        0.025 |        0.000 |
| CR4L  | KAP-1.25               |  512 |               0.045 |           1.264 |      0.040 |        0.294 |        -0.040 |           0.044 |          0.073 |       1.963 |           0.178 |             0.129 |                0.059 |              0.098 |        0.311 |        0.062 |
| CR4L  | TRN-2                  |  512 |               0.013 |           0.884 |      0.048 |        0.630 |        -0.584 |           0.054 |          0.098 |       1.907 |           0.000 |             0.000 |                0.000 |              0.004 |        0.000 |        0.000 |
| CR4L  | TRN-3                  |  512 |               0.056 |           0.977 |      0.059 |        0.342 |        -0.165 |           0.043 |          0.065 |       1.565 |           0.045 |             0.018 |                0.002 |              0.018 |        0.113 |        0.000 |
| CR5   | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.173 |        0.268 |        -0.181 |           0.025 |          0.048 |       1.361 |           0.059 |             0.030 |                0.013 |              0.006 |        0.127 |        0.005 |
| CR5   | empirical VALIDATION Z | 1200 |               0.023 |           1.004 |      0.244 |        0.254 |        -0.203 |           0.021 |          0.039 |       1.401 |           0.058 |             0.028 |                0.010 |              0.010 |        0.126 |        0.003 |
| CR5   | Q01                    |  512 |               0.045 |           0.966 |      0.146 |        0.342 |        -0.233 |           0.036 |          0.091 |       1.423 |           0.043 |             0.020 |                0.008 |              0.002 |        0.117 |        0.000 |
| CR5   | Q02                    |  128 |               0.114 |           0.970 |      0.187 |        0.438 |        -0.334 |           0.154 |          0.217 |       2.043 |           0.070 |             0.008 |                0.000 |              0.000 |        0.148 |        0.000 |
| CR5   | Q03                    |  512 |               0.024 |           1.006 |      0.105 |        0.009 |         0.002 |           0.042 |          0.057 |       1.312 |           0.082 |             0.053 |                0.008 |              0.033 |        0.135 |        0.010 |
| CR5   | M-E                    |  128 |               0.114 |           0.970 |      0.187 |        0.438 |        -0.334 |           0.154 |          0.217 |       2.043 |           0.070 |             0.008 |                0.000 |              0.000 |        0.148 |        0.000 |
| CR5   | M-J                    |  128 |               0.122 |           1.000 |      0.230 |        0.236 |        -0.090 |           0.071 |          0.133 |       1.439 |           0.109 |             0.047 |                0.000 |              0.039 |        0.148 |        0.000 |
| CR5   | M-D                    |  128 |               0.124 |           1.003 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.548 |           0.109 |             0.047 |                0.000 |              0.031 |        0.148 |        0.000 |
| CR5   | M-I                    |  128 |               0.078 |           1.006 |      0.400 |        0.599 |        -0.331 |           0.051 |          0.085 |       1.824 |           0.055 |             0.031 |                0.008 |              0.055 |        0.117 |        0.000 |
| CR5   | RHO-0.25               |  128 |               0.124 |           1.002 |      0.232 |        0.231 |        -0.087 |           0.084 |          0.168 |       1.518 |           0.109 |             0.047 |                0.000 |              0.031 |        0.148 |        0.000 |
| CR5   | RHO-0.5                |  128 |               0.123 |           1.002 |      0.232 |        0.232 |        -0.088 |           0.079 |          0.157 |       1.490 |           0.109 |             0.047 |                0.000 |              0.031 |        0.148 |        0.000 |
| CR5   | RHO-0.75               |  128 |               0.123 |           1.001 |      0.231 |        0.234 |        -0.089 |           0.075 |          0.145 |       1.464 |           0.109 |             0.047 |                0.000 |              0.031 |        0.148 |        0.000 |
| CR5   | KAP-0.75               |  128 |               0.093 |           0.752 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.548 |           0.016 |             0.000 |                0.000 |              0.000 |        0.016 |        0.000 |
| CR5   | KAP-1.25               |  128 |               0.155 |           1.254 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.548 |           0.195 |             0.156 |                0.094 |              0.141 |        0.281 |        0.070 |
| CR5   | TRN-2                  |  128 |               0.015 |           0.868 |      0.133 |        0.753 |        -0.553 |           0.099 |          0.211 |       1.633 |           0.000 |             0.000 |                0.000 |              0.000 |        0.000 |        0.000 |
| CR5   | TRN-3                  |  128 |               0.023 |           1.025 |      0.318 |        0.712 |        -0.251 |           0.059 |          0.085 |       1.446 |           0.102 |             0.070 |                0.008 |              0.047 |        0.148 |        0.000 |
| CR5L  | empirical TRAIN Z      | 5600 |               0.000 |           1.000 |      0.569 |        0.441 |         0.053 |           0.019 |          0.029 |       1.632 |           0.060 |             0.030 |                0.014 |              0.004 |        0.125 |        0.011 |
| CR5L  | empirical VALIDATION Z | 1200 |               0.014 |           1.013 |      0.421 |        0.129 |        -0.047 |           0.029 |          0.048 |       1.727 |           0.067 |             0.037 |                0.010 |              0.010 |        0.142 |        0.009 |
| CR5L  | Q01                    |  512 |               0.051 |           0.963 |      0.463 |        0.559 |         0.022 |           0.035 |          0.060 |       1.728 |           0.051 |             0.023 |                0.008 |              0.006 |        0.102 |        0.010 |
| CR5L  | Q02                    |  128 |               0.105 |           0.982 |      0.555 |        0.760 |        -0.013 |           0.098 |          0.135 |       1.988 |           0.047 |             0.008 |                0.008 |              0.000 |        0.117 |        0.016 |
| CR5L  | Q03                    |  512 |               0.025 |           1.006 |      0.105 |        0.019 |         0.007 |           0.022 |          0.028 |       1.548 |           0.084 |             0.059 |                0.012 |              0.035 |        0.133 |        0.010 |
| CR5L  | M-E                    |  128 |               0.105 |           0.982 |      0.555 |        0.760 |        -0.013 |           0.098 |          0.135 |       1.988 |           0.047 |             0.008 |                0.008 |              0.000 |        0.117 |        0.016 |
| CR5L  | M-J                    |  128 |               0.122 |           1.003 |      0.235 |        0.207 |        -0.081 |           0.097 |          0.166 |       1.801 |           0.086 |             0.078 |                0.000 |              0.070 |        0.148 |        0.000 |
| CR5L  | M-D                    |  128 |               0.124 |           1.003 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.797 |           0.086 |             0.078 |                0.000 |              0.070 |        0.148 |        0.000 |
| CR5L  | M-I                    |  128 |               0.038 |           1.037 |      0.233 |        0.492 |        -0.375 |           0.069 |          0.131 |       1.698 |           0.047 |             0.031 |                0.016 |              0.039 |        0.164 |        0.000 |
| CR5L  | RHO-0.25               |  128 |               0.124 |           1.003 |      0.234 |        0.224 |        -0.084 |           0.090 |          0.176 |       1.794 |           0.086 |             0.078 |                0.000 |              0.070 |        0.148 |        0.000 |
| CR5L  | RHO-0.5                |  128 |               0.123 |           1.003 |      0.234 |        0.218 |        -0.083 |           0.093 |          0.173 |       1.794 |           0.086 |             0.078 |                0.000 |              0.070 |        0.148 |        0.000 |
| CR5L  | RHO-0.75               |  128 |               0.123 |           1.003 |      0.235 |        0.212 |        -0.082 |           0.095 |          0.170 |       1.796 |           0.086 |             0.078 |                0.000 |              0.070 |        0.148 |        0.000 |
| CR5L  | KAP-0.75               |  128 |               0.093 |           0.752 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.797 |           0.008 |             0.000 |                0.000 |              0.000 |        0.016 |        0.000 |
| CR5L  | KAP-1.25               |  128 |               0.155 |           1.254 |      0.233 |        0.229 |        -0.085 |           0.088 |          0.180 |       1.797 |           0.219 |             0.141 |                0.094 |              0.125 |        0.281 |        0.070 |
| CR5L  | TRN-2                  |  128 |               0.015 |           0.868 |      0.133 |        0.753 |        -0.553 |           0.099 |          0.211 |       2.233 |           0.016 |             0.000 |                0.000 |              0.000 |        0.000 |        0.000 |
| CR5L  | TRN-3                  |  128 |               0.023 |           1.025 |      0.318 |        0.712 |        -0.251 |           0.059 |          0.085 |       2.257 |           0.086 |             0.047 |                0.008 |              0.031 |        0.148 |        0.000 |


## Tail energy share of the base-frame UX energy

| toy   | law      | mask     |   n_blocks |   energy_tail2 |   energy_tail3 |   blocks_tail2 |   blocks_tail3 |   top1pct_block_energy |   valid_pairs |
|:------|:---------|:---------|-----------:|---------------:|---------------:|---------------:|---------------:|-----------------------:|--------------:|
| C1    | Q01      | FULL     |        512 |          0.192 |          0.007 |          0.180 |          0.006 |                  0.012 |     32768.000 |
| C1    | Q02      | FULL     |        512 |          0.193 |          0.007 |          0.180 |          0.006 |                  0.012 |     32768.000 |
| C1    | Q03      | FULL     |        512 |          0.169 |          0.016 |          0.156 |          0.014 |                  0.013 |     32768.000 |
| C1    | M-E      | FULL     |        512 |          0.193 |          0.007 |          0.180 |          0.006 |                  0.012 |     32768.000 |
| C1    | M-J      | FULL     |        512 |          0.213 |          0.011 |          0.199 |          0.010 |                  0.012 |     32768.000 |
| C1    | M-D      | FULL     |        512 |          0.213 |          0.008 |          0.199 |          0.008 |                  0.012 |     32768.000 |
| C1    | M-I      | FULL     |        512 |          0.163 |          0.005 |          0.150 |          0.004 |                  0.013 |     32768.000 |
| C1    | RHO-0.25 | FULL     |        512 |          0.213 |          0.008 |          0.199 |          0.008 |                  0.012 |     32768.000 |
| C1    | RHO-0.5  | FULL     |        512 |          0.213 |          0.011 |          0.199 |          0.010 |                  0.012 |     32768.000 |
| C1    | RHO-0.75 | FULL     |        512 |          0.215 |          0.011 |          0.201 |          0.010 |                  0.012 |     32768.000 |
| C1    | KAP-0.75 | FULL     |        512 |          0.039 |          0.000 |          0.035 |          0.000 |                  0.012 |     32768.000 |
| C1    | KAP-1.25 | FULL     |        512 |          0.415 |          0.085 |          0.395 |          0.080 |                  0.013 |     32768.000 |
| C1    | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.012 |     32768.000 |
| C1    | TRN-3    | FULL     |        512 |          0.157 |          0.000 |          0.143 |          0.000 |                  0.013 |     32768.000 |
| C1L   | Q01      | FULL     |        512 |          0.164 |          0.007 |          0.166 |          0.008 |                  0.011 |     32768.000 |
| C1L   | Q02      | FULL     |        512 |          0.163 |          0.007 |          0.166 |          0.008 |                  0.011 |     32768.000 |
| C1L   | Q03      | FULL     |        512 |          0.156 |          0.015 |          0.158 |          0.016 |                  0.011 |     32768.000 |
| C1L   | M-E      | FULL     |        512 |          0.164 |          0.007 |          0.166 |          0.008 |                  0.011 |     32768.000 |
| C1L   | M-J      | FULL     |        512 |          0.196 |          0.008 |          0.197 |          0.008 |                  0.011 |     32768.000 |
| C1L   | M-D      | FULL     |        512 |          0.198 |          0.008 |          0.199 |          0.008 |                  0.011 |     32768.000 |
| C1L   | M-I      | FULL     |        512 |          0.164 |          0.010 |          0.166 |          0.012 |                  0.011 |     32768.000 |
| C1L   | RHO-0.25 | FULL     |        512 |          0.198 |          0.008 |          0.199 |          0.008 |                  0.011 |     32768.000 |
| C1L   | RHO-0.5  | FULL     |        512 |          0.196 |          0.008 |          0.197 |          0.008 |                  0.011 |     32768.000 |
| C1L   | RHO-0.75 | FULL     |        512 |          0.196 |          0.008 |          0.197 |          0.008 |                  0.011 |     32768.000 |
| C1L   | KAP-0.75 | FULL     |        512 |          0.036 |          0.000 |          0.035 |          0.000 |                  0.011 |     32768.000 |
| C1L   | KAP-1.25 | FULL     |        512 |          0.389 |          0.076 |          0.395 |          0.080 |                  0.011 |     32768.000 |
| C1L   | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.011 |     32768.000 |
| C1L   | TRN-3    | FULL     |        512 |          0.142 |          0.000 |          0.143 |          0.000 |                  0.011 |     32768.000 |
| C2    | Q01      | FULL     |        512 |          0.182 |          0.000 |          0.160 |          0.000 |                  0.015 |     32768.000 |
| C2    | Q02      | FULL     |        512 |          0.183 |          0.000 |          0.160 |          0.000 |                  0.016 |     32768.000 |
| C2    | Q03      | FULL     |        512 |          0.193 |          0.011 |          0.164 |          0.010 |                  0.016 |     32768.000 |
| C2    | M-E      | FULL     |        512 |          0.183 |          0.000 |          0.160 |          0.000 |                  0.015 |     32768.000 |
| C2    | M-J      | FULL     |        512 |          0.221 |          0.009 |          0.189 |          0.008 |                  0.016 |     32768.000 |
| C2    | M-D      | FULL     |        512 |          0.231 |          0.009 |          0.199 |          0.008 |                  0.016 |     32768.000 |
| C2    | M-I      | FULL     |        512 |          0.198 |          0.007 |          0.166 |          0.006 |                  0.016 |     32768.000 |
| C2    | RHO-0.25 | FULL     |        512 |          0.225 |          0.009 |          0.193 |          0.008 |                  0.016 |     32768.000 |
| C2    | RHO-0.5  | FULL     |        512 |          0.223 |          0.009 |          0.191 |          0.008 |                  0.016 |     32768.000 |
| C2    | RHO-0.75 | FULL     |        512 |          0.218 |          0.009 |          0.188 |          0.008 |                  0.016 |     32768.000 |
| C2    | KAP-0.75 | FULL     |        512 |          0.042 |          0.000 |          0.035 |          0.000 |                  0.015 |     32768.000 |
| C2    | KAP-1.25 | FULL     |        512 |          0.435 |          0.083 |          0.395 |          0.080 |                  0.016 |     32768.000 |
| C2    | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.015 |     32768.000 |
| C2    | TRN-3    | FULL     |        512 |          0.174 |          0.000 |          0.143 |          0.000 |                  0.017 |     32768.000 |
| C2L   | Q01      | FULL     |        512 |          0.186 |          0.022 |          0.191 |          0.023 |                  0.012 |     32768.000 |
| C2L   | Q02      | FULL     |        512 |          0.186 |          0.021 |          0.191 |          0.023 |                  0.012 |     32768.000 |
| C2L   | Q03      | FULL     |        512 |          0.153 |          0.013 |          0.156 |          0.014 |                  0.012 |     32768.000 |
| C2L   | M-E      | FULL     |        512 |          0.187 |          0.022 |          0.191 |          0.023 |                  0.012 |     32768.000 |
| C2L   | M-J      | FULL     |        512 |          0.194 |          0.009 |          0.201 |          0.010 |                  0.012 |     32768.000 |
| C2L   | M-D      | FULL     |        512 |          0.192 |          0.007 |          0.199 |          0.008 |                  0.012 |     32768.000 |
| C2L   | M-I      | FULL     |        512 |          0.165 |          0.006 |          0.168 |          0.006 |                  0.012 |     32768.000 |
| C2L   | RHO-0.25 | FULL     |        512 |          0.194 |          0.007 |          0.201 |          0.008 |                  0.012 |     32768.000 |
| C2L   | RHO-0.5  | FULL     |        512 |          0.194 |          0.009 |          0.201 |          0.010 |                  0.012 |     32768.000 |
| C2L   | RHO-0.75 | FULL     |        512 |          0.192 |          0.009 |          0.199 |          0.010 |                  0.012 |     32768.000 |
| C2L   | KAP-0.75 | FULL     |        512 |          0.034 |          0.000 |          0.035 |          0.000 |                  0.012 |     32768.000 |
| C2L   | KAP-1.25 | FULL     |        512 |          0.379 |          0.074 |          0.395 |          0.080 |                  0.013 |     32768.000 |
| C2L   | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.012 |     32768.000 |
| C2L   | TRN-3    | FULL     |        512 |          0.141 |          0.000 |          0.143 |          0.000 |                  0.012 |     32768.000 |
| C3    | Q01      | FULL     |        512 |          0.227 |          0.009 |          0.205 |          0.008 |                  0.014 |     32768.000 |
| C3    | Q02      | FULL     |        512 |          0.228 |          0.010 |          0.205 |          0.008 |                  0.014 |     32768.000 |
| C3    | Q03      | FULL     |        512 |          0.275 |          0.022 |          0.240 |          0.020 |                  0.015 |     32768.000 |
| C3    | M-E      | FULL     |        512 |          0.229 |          0.010 |          0.205 |          0.008 |                  0.014 |     32768.000 |
| C3    | M-J      | FULL     |        512 |          0.303 |          0.015 |          0.262 |          0.014 |                  0.015 |     32768.000 |
| C3    | M-D      | FULL     |        512 |          0.302 |          0.013 |          0.262 |          0.012 |                  0.015 |     32768.000 |
| C3    | M-I      | FULL     |        512 |          0.216 |          0.003 |          0.182 |          0.002 |                  0.015 |     32768.000 |
| C3    | RHO-0.25 | FULL     |        512 |          0.305 |          0.015 |          0.264 |          0.014 |                  0.015 |     32768.000 |
| C3    | RHO-0.5  | FULL     |        512 |          0.305 |          0.015 |          0.264 |          0.014 |                  0.015 |     32768.000 |
| C3    | RHO-0.75 | FULL     |        512 |          0.305 |          0.015 |          0.264 |          0.014 |                  0.015 |     32768.000 |
| C3    | KAP-0.75 | FULL     |        512 |          0.056 |          0.000 |          0.045 |          0.000 |                  0.016 |     32768.000 |
| C3    | KAP-1.25 | FULL     |        512 |          0.569 |          0.104 |          0.527 |          0.096 |                  0.014 |     32768.000 |
| C3    | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.015 |     32768.000 |
| C3    | TRN-3    | FULL     |        512 |          0.250 |          0.000 |          0.215 |          0.000 |                  0.015 |     32768.000 |
| C3L   | Q01      | FULL     |        512 |          0.222 |          0.023 |          0.225 |          0.023 |                  0.012 |     32768.000 |
| C3L   | Q02      | FULL     |        512 |          0.222 |          0.023 |          0.225 |          0.023 |                  0.012 |     32768.000 |
| C3L   | Q03      | FULL     |        512 |          0.230 |          0.019 |          0.230 |          0.020 |                  0.013 |     32768.000 |
| C3L   | M-E      | FULL     |        512 |          0.222 |          0.023 |          0.225 |          0.023 |                  0.012 |     32768.000 |
| C3L   | M-J      | FULL     |        512 |          0.260 |          0.011 |          0.260 |          0.012 |                  0.012 |     32768.000 |
| C3L   | M-D      | FULL     |        512 |          0.262 |          0.011 |          0.262 |          0.012 |                  0.012 |     32768.000 |
| C3L   | M-I      | FULL     |        512 |          0.266 |          0.015 |          0.266 |          0.014 |                  0.013 |     32768.000 |
| C3L   | RHO-0.25 | FULL     |        512 |          0.262 |          0.011 |          0.262 |          0.012 |                  0.012 |     32768.000 |
| C3L   | RHO-0.5  | FULL     |        512 |          0.262 |          0.011 |          0.262 |          0.012 |                  0.012 |     32768.000 |
| C3L   | RHO-0.75 | FULL     |        512 |          0.260 |          0.011 |          0.260 |          0.012 |                  0.012 |     32768.000 |
| C3L   | KAP-0.75 | FULL     |        512 |          0.045 |          0.000 |          0.045 |          0.000 |                  0.012 |     32768.000 |
| C3L   | KAP-1.25 | FULL     |        512 |          0.523 |          0.094 |          0.527 |          0.096 |                  0.013 |     32768.000 |
| C3L   | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.012 |     32768.000 |
| C3L   | TRN-3    | FULL     |        512 |          0.215 |          0.000 |          0.215 |          0.000 |                  0.012 |     32768.000 |
| CR3   | Q01      | MASK-ALL |        512 |          0.165 |          0.021 |          0.139 |          0.012 |                  0.030 |      7370.000 |
| CR3   | Q01      | MASK-COL |        512 |          0.160 |          0.018 |          0.139 |          0.012 |                  0.026 |     25237.833 |
| CR3   | Q01      | FULL     |        512 |          0.162 |          0.019 |          0.139 |          0.012 |                  0.025 |     32768.000 |
| CR3   | Q02      | MASK-ALL |        128 |          0.121 |          0.000 |          0.109 |          0.000 |                  0.022 |     14336.000 |
| CR3   | Q02      | MASK-COL |        128 |          0.119 |          0.000 |          0.109 |          0.000 |                  0.021 |     49280.000 |
| CR3   | Q02      | FULL     |        128 |          0.121 |          0.000 |          0.109 |          0.000 |                  0.021 |     64000.000 |
| CR3   | Q03      | MASK-ALL |        512 |          0.161 |          0.017 |          0.156 |          0.014 |                  0.037 |      7418.000 |
| CR3   | Q03      | MASK-COL |        512 |          0.161 |          0.018 |          0.156 |          0.014 |                  0.027 |     25264.667 |
| CR3   | Q03      | FULL     |        512 |          0.161 |          0.018 |          0.156 |          0.014 |                  0.028 |     32768.000 |
| CR3   | M-E      | MASK-ALL |        128 |          0.121 |          0.000 |          0.109 |          0.000 |                  0.022 |     14336.000 |
| CR3   | M-E      | MASK-COL |        128 |          0.119 |          0.000 |          0.109 |          0.000 |                  0.021 |     49280.000 |
| CR3   | M-E      | FULL     |        128 |          0.121 |          0.000 |          0.109 |          0.000 |                  0.021 |     64000.000 |
| CR3   | M-J      | MASK-ALL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3   | M-J      | MASK-COL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.018 |     49280.000 |
| CR3   | M-J      | FULL     |        128 |          0.202 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3   | M-D      | MASK-ALL |        128 |          0.199 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3   | M-D      | MASK-COL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.018 |     49280.000 |
| CR3   | M-D      | FULL     |        128 |          0.202 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3   | M-I      | MASK-ALL |        128 |          0.229 |          0.012 |          0.203 |          0.008 |                  0.021 |     14336.000 |
| CR3   | M-I      | MASK-COL |        128 |          0.225 |          0.011 |          0.203 |          0.008 |                  0.020 |     49280.000 |
| CR3   | M-I      | FULL     |        128 |          0.228 |          0.011 |          0.203 |          0.008 |                  0.020 |     64000.000 |
| CR3   | RHO-0.25 | MASK-ALL |        128 |          0.199 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3   | RHO-0.25 | MASK-COL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.018 |     49280.000 |
| CR3   | RHO-0.25 | FULL     |        128 |          0.202 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3   | RHO-0.5  | MASK-ALL |        128 |          0.199 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3   | RHO-0.5  | MASK-COL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.018 |     49280.000 |
| CR3   | RHO-0.5  | FULL     |        128 |          0.202 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3   | RHO-0.75 | MASK-ALL |        128 |          0.199 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3   | RHO-0.75 | MASK-COL |        128 |          0.198 |          0.000 |          0.180 |          0.000 |                  0.018 |     49280.000 |
| CR3   | RHO-0.75 | FULL     |        128 |          0.202 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3   | KAP-0.75 | MASK-ALL |        128 |          0.015 |          0.000 |          0.016 |          0.000 |                  0.015 |     14336.000 |
| CR3   | KAP-0.75 | MASK-COL |        128 |          0.014 |          0.000 |          0.016 |          0.000 |                  0.015 |     49280.000 |
| CR3   | KAP-0.75 | FULL     |        128 |          0.014 |          0.000 |          0.016 |          0.000 |                  0.016 |     64000.000 |
| CR3   | KAP-1.25 | MASK-ALL |        128 |          0.406 |          0.089 |          0.375 |          0.086 |                  0.021 |     14336.000 |
| CR3   | KAP-1.25 | MASK-COL |        128 |          0.400 |          0.087 |          0.375 |          0.086 |                  0.019 |     49280.000 |
| CR3   | KAP-1.25 | FULL     |        128 |          0.403 |          0.087 |          0.375 |          0.086 |                  0.020 |     64000.000 |
| CR3   | TRN-2    | MASK-ALL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.021 |     14336.000 |
| CR3   | TRN-2    | MASK-COL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.021 |     49280.000 |
| CR3   | TRN-2    | FULL     |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.021 |     64000.000 |
| CR3   | TRN-3    | MASK-ALL |        128 |          0.188 |          0.000 |          0.156 |          0.000 |                  0.021 |     14336.000 |
| CR3   | TRN-3    | MASK-COL |        128 |          0.184 |          0.000 |          0.156 |          0.000 |                  0.020 |     49280.000 |
| CR3   | TRN-3    | FULL     |        128 |          0.187 |          0.000 |          0.156 |          0.000 |                  0.021 |     64000.000 |
| CR3L  | Q01      | MASK-ALL |        512 |          0.180 |          0.037 |          0.156 |          0.021 |                  0.031 |      7370.000 |
| CR3L  | Q01      | MASK-COL |        512 |          0.185 |          0.033 |          0.156 |          0.021 |                  0.027 |     25237.833 |
| CR3L  | Q01      | FULL     |        512 |          0.191 |          0.034 |          0.156 |          0.021 |                  0.025 |     32768.000 |
| CR3L  | Q02      | MASK-ALL |        128 |          0.197 |          0.024 |          0.180 |          0.016 |                  0.021 |     14336.000 |
| CR3L  | Q02      | MASK-COL |        128 |          0.199 |          0.025 |          0.180 |          0.016 |                  0.021 |     49280.000 |
| CR3L  | Q02      | FULL     |        128 |          0.203 |          0.027 |          0.180 |          0.016 |                  0.022 |     64000.000 |
| CR3L  | Q03      | MASK-ALL |        512 |          0.164 |          0.015 |          0.158 |          0.016 |                  0.029 |      7418.000 |
| CR3L  | Q03      | MASK-COL |        512 |          0.169 |          0.017 |          0.158 |          0.016 |                  0.026 |     25264.667 |
| CR3L  | Q03      | FULL     |        512 |          0.167 |          0.018 |          0.158 |          0.016 |                  0.025 |     32768.000 |
| CR3L  | M-E      | MASK-ALL |        128 |          0.197 |          0.024 |          0.180 |          0.016 |                  0.021 |     14336.000 |
| CR3L  | M-E      | MASK-COL |        128 |          0.199 |          0.025 |          0.180 |          0.016 |                  0.021 |     49280.000 |
| CR3L  | M-E      | FULL     |        128 |          0.203 |          0.027 |          0.180 |          0.016 |                  0.022 |     64000.000 |
| CR3L  | M-J      | MASK-ALL |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     14336.000 |
| CR3L  | M-J      | MASK-COL |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     49280.000 |
| CR3L  | M-J      | FULL     |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     64000.000 |
| CR3L  | M-D      | MASK-ALL |        128 |          0.205 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3L  | M-D      | MASK-COL |        128 |          0.206 |          0.000 |          0.180 |          0.000 |                  0.019 |     49280.000 |
| CR3L  | M-D      | FULL     |        128 |          0.207 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3L  | M-I      | MASK-ALL |        128 |          0.232 |          0.004 |          0.203 |          0.008 |                  0.026 |     14336.000 |
| CR3L  | M-I      | MASK-COL |        128 |          0.235 |          0.004 |          0.203 |          0.008 |                  0.027 |     49280.000 |
| CR3L  | M-I      | FULL     |        128 |          0.239 |          0.003 |          0.203 |          0.008 |                  0.027 |     64000.000 |
| CR3L  | RHO-0.25 | MASK-ALL |        128 |          0.205 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3L  | RHO-0.25 | MASK-COL |        128 |          0.206 |          0.000 |          0.180 |          0.000 |                  0.019 |     49280.000 |
| CR3L  | RHO-0.25 | FULL     |        128 |          0.207 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3L  | RHO-0.5  | MASK-ALL |        128 |          0.205 |          0.000 |          0.180 |          0.000 |                  0.019 |     14336.000 |
| CR3L  | RHO-0.5  | MASK-COL |        128 |          0.206 |          0.000 |          0.180 |          0.000 |                  0.019 |     49280.000 |
| CR3L  | RHO-0.5  | FULL     |        128 |          0.207 |          0.000 |          0.180 |          0.000 |                  0.019 |     64000.000 |
| CR3L  | RHO-0.75 | MASK-ALL |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     14336.000 |
| CR3L  | RHO-0.75 | MASK-COL |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     49280.000 |
| CR3L  | RHO-0.75 | FULL     |        128 |          0.201 |          0.000 |          0.172 |          0.000 |                  0.019 |     64000.000 |
| CR3L  | KAP-0.75 | MASK-ALL |        128 |          0.008 |          0.000 |          0.016 |          0.000 |                  0.016 |     14336.000 |
| CR3L  | KAP-0.75 | MASK-COL |        128 |          0.008 |          0.000 |          0.016 |          0.000 |                  0.016 |     49280.000 |
| CR3L  | KAP-0.75 | FULL     |        128 |          0.007 |          0.000 |          0.016 |          0.000 |                  0.016 |     64000.000 |
| CR3L  | KAP-1.25 | MASK-ALL |        128 |          0.429 |          0.057 |          0.375 |          0.086 |                  0.022 |     14336.000 |
| CR3L  | KAP-1.25 | MASK-COL |        128 |          0.429 |          0.057 |          0.375 |          0.086 |                  0.022 |     49280.000 |
| CR3L  | KAP-1.25 | FULL     |        128 |          0.427 |          0.057 |          0.375 |          0.086 |                  0.022 |     64000.000 |
| CR3L  | TRN-2    | MASK-ALL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.016 |     14336.000 |
| CR3L  | TRN-2    | MASK-COL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.016 |     49280.000 |
| CR3L  | TRN-2    | FULL     |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.016 |     64000.000 |
| CR3L  | TRN-3    | MASK-ALL |        128 |          0.166 |          0.000 |          0.156 |          0.000 |                  0.022 |     14336.000 |
| CR3L  | TRN-3    | MASK-COL |        128 |          0.168 |          0.000 |          0.156 |          0.000 |                  0.022 |     49280.000 |
| CR3L  | TRN-3    | FULL     |        128 |          0.172 |          0.000 |          0.156 |          0.000 |                  0.021 |     64000.000 |
| CR4   | Q01      | FULL     |        512 |          0.104 |          0.004 |          0.104 |          0.006 |                  0.020 |     32768.000 |
| CR4   | Q02      | FULL     |        512 |          0.105 |          0.004 |          0.104 |          0.006 |                  0.020 |     32768.000 |
| CR4   | Q03      | FULL     |        512 |          0.123 |          0.011 |          0.131 |          0.012 |                  0.019 |     32768.000 |
| CR4   | M-E      | FULL     |        512 |          0.105 |          0.004 |          0.104 |          0.006 |                  0.020 |     32768.000 |
| CR4   | M-J      | FULL     |        512 |          0.150 |          0.002 |          0.148 |          0.004 |                  0.020 |     32768.000 |
| CR4   | M-D      | FULL     |        512 |          0.150 |          0.002 |          0.148 |          0.004 |                  0.020 |     32768.000 |
| CR4   | M-I      | FULL     |        512 |          0.115 |          0.007 |          0.123 |          0.010 |                  0.020 |     32768.000 |
| CR4   | RHO-0.25 | FULL     |        512 |          0.150 |          0.002 |          0.148 |          0.004 |                  0.020 |     32768.000 |
| CR4   | RHO-0.5  | FULL     |        512 |          0.150 |          0.002 |          0.148 |          0.004 |                  0.020 |     32768.000 |
| CR4   | RHO-0.75 | FULL     |        512 |          0.150 |          0.002 |          0.148 |          0.004 |                  0.020 |     32768.000 |
| CR4   | KAP-0.75 | FULL     |        512 |          0.023 |          0.000 |          0.025 |          0.000 |                  0.018 |     32768.000 |
| CR4   | KAP-1.25 | FULL     |        512 |          0.322 |          0.063 |          0.311 |          0.062 |                  0.021 |     32768.000 |
| CR4   | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.017 |     32768.000 |
| CR4   | TRN-3    | FULL     |        512 |          0.118 |          0.000 |          0.113 |          0.000 |                  0.020 |     32768.000 |
| CR4L  | Q01      | FULL     |        512 |          0.144 |          0.012 |          0.119 |          0.010 |                  0.024 |     32768.000 |
| CR4L  | Q02      | FULL     |        512 |          0.144 |          0.011 |          0.119 |          0.010 |                  0.024 |     32768.000 |
| CR4L  | Q03      | FULL     |        512 |          0.152 |          0.013 |          0.133 |          0.012 |                  0.024 |     32768.000 |
| CR4L  | M-E      | FULL     |        512 |          0.145 |          0.011 |          0.119 |          0.010 |                  0.025 |     32768.000 |
| CR4L  | M-J      | FULL     |        512 |          0.166 |          0.007 |          0.148 |          0.004 |                  0.026 |     32768.000 |
| CR4L  | M-D      | FULL     |        512 |          0.166 |          0.007 |          0.148 |          0.004 |                  0.026 |     32768.000 |
| CR4L  | M-I      | FULL     |        512 |          0.150 |          0.025 |          0.137 |          0.016 |                  0.026 |     32768.000 |
| CR4L  | RHO-0.25 | FULL     |        512 |          0.166 |          0.007 |          0.148 |          0.004 |                  0.026 |     32768.000 |
| CR4L  | RHO-0.5  | FULL     |        512 |          0.166 |          0.007 |          0.148 |          0.004 |                  0.026 |     32768.000 |
| CR4L  | RHO-0.75 | FULL     |        512 |          0.166 |          0.007 |          0.148 |          0.004 |                  0.026 |     32768.000 |
| CR4L  | KAP-0.75 | FULL     |        512 |          0.032 |          0.000 |          0.025 |          0.000 |                  0.023 |     32768.000 |
| CR4L  | KAP-1.25 | FULL     |        512 |          0.328 |          0.069 |          0.311 |          0.062 |                  0.027 |     32768.000 |
| CR4L  | TRN-2    | FULL     |        512 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.021 |     32768.000 |
| CR4L  | TRN-3    | FULL     |        512 |          0.119 |          0.000 |          0.113 |          0.000 |                  0.023 |     32768.000 |
| CR5   | Q01      | MASK-ALL |        512 |          0.123 |          0.000 |          0.117 |          0.000 |                  0.025 |      7370.000 |
| CR5   | Q01      | MASK-COL |        512 |          0.125 |          0.000 |          0.117 |          0.000 |                  0.021 |     25237.833 |
| CR5   | Q01      | FULL     |        512 |          0.127 |          0.000 |          0.117 |          0.000 |                  0.022 |     32768.000 |
| CR5   | Q02      | MASK-ALL |        128 |          0.154 |          0.000 |          0.148 |          0.000 |                  0.016 |     14336.000 |
| CR5   | Q02      | MASK-COL |        128 |          0.153 |          0.000 |          0.148 |          0.000 |                  0.016 |     49280.000 |
| CR5   | Q02      | FULL     |        128 |          0.155 |          0.000 |          0.148 |          0.000 |                  0.017 |     64000.000 |
| CR5   | Q03      | MASK-ALL |        512 |          0.127 |          0.011 |          0.135 |          0.010 |                  0.029 |      7418.000 |
| CR5   | Q03      | MASK-COL |        512 |          0.130 |          0.011 |          0.135 |          0.010 |                  0.022 |     25264.667 |
| CR5   | Q03      | FULL     |        512 |          0.133 |          0.011 |          0.135 |          0.010 |                  0.022 |     32768.000 |
| CR5   | M-E      | MASK-ALL |        128 |          0.154 |          0.000 |          0.148 |          0.000 |                  0.016 |     14336.000 |
| CR5   | M-E      | MASK-COL |        128 |          0.153 |          0.000 |          0.148 |          0.000 |                  0.016 |     49280.000 |
| CR5   | M-E      | FULL     |        128 |          0.155 |          0.000 |          0.148 |          0.000 |                  0.017 |     64000.000 |
| CR5   | M-J      | MASK-ALL |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | M-J      | MASK-COL |        128 |          0.160 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | M-J      | FULL     |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     64000.000 |
| CR5   | M-D      | MASK-ALL |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | M-D      | MASK-COL |        128 |          0.159 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | M-D      | FULL     |        128 |          0.161 |          0.000 |          0.148 |          0.000 |                  0.018 |     64000.000 |
| CR5   | M-I      | MASK-ALL |        128 |          0.155 |          0.000 |          0.117 |          0.000 |                  0.019 |     14336.000 |
| CR5   | M-I      | MASK-COL |        128 |          0.151 |          0.000 |          0.117 |          0.000 |                  0.018 |     49280.000 |
| CR5   | M-I      | FULL     |        128 |          0.155 |          0.000 |          0.117 |          0.000 |                  0.019 |     64000.000 |
| CR5   | RHO-0.25 | MASK-ALL |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | RHO-0.25 | MASK-COL |        128 |          0.159 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | RHO-0.25 | FULL     |        128 |          0.161 |          0.000 |          0.148 |          0.000 |                  0.018 |     64000.000 |
| CR5   | RHO-0.5  | MASK-ALL |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | RHO-0.5  | MASK-COL |        128 |          0.159 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | RHO-0.5  | FULL     |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     64000.000 |
| CR5   | RHO-0.75 | MASK-ALL |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | RHO-0.75 | MASK-COL |        128 |          0.160 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | RHO-0.75 | FULL     |        128 |          0.162 |          0.000 |          0.148 |          0.000 |                  0.018 |     64000.000 |
| CR5   | KAP-0.75 | MASK-ALL |        128 |          0.019 |          0.000 |          0.016 |          0.000 |                  0.016 |     14336.000 |
| CR5   | KAP-0.75 | MASK-COL |        128 |          0.019 |          0.000 |          0.016 |          0.000 |                  0.015 |     49280.000 |
| CR5   | KAP-0.75 | FULL     |        128 |          0.019 |          0.000 |          0.016 |          0.000 |                  0.016 |     64000.000 |
| CR5   | KAP-1.25 | MASK-ALL |        128 |          0.304 |          0.088 |          0.281 |          0.070 |                  0.020 |     14336.000 |
| CR5   | KAP-1.25 | MASK-COL |        128 |          0.299 |          0.085 |          0.281 |          0.070 |                  0.019 |     49280.000 |
| CR5   | KAP-1.25 | FULL     |        128 |          0.303 |          0.087 |          0.281 |          0.070 |                  0.020 |     64000.000 |
| CR5   | TRN-2    | MASK-ALL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.018 |     14336.000 |
| CR5   | TRN-2    | MASK-COL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.018 |     49280.000 |
| CR5   | TRN-2    | FULL     |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.018 |     64000.000 |
| CR5   | TRN-3    | MASK-ALL |        128 |          0.187 |          0.000 |          0.148 |          0.000 |                  0.018 |     14336.000 |
| CR5   | TRN-3    | MASK-COL |        128 |          0.184 |          0.000 |          0.148 |          0.000 |                  0.018 |     49280.000 |
| CR5   | TRN-3    | FULL     |        128 |          0.187 |          0.000 |          0.148 |          0.000 |                  0.019 |     64000.000 |
| CR5L  | Q01      | MASK-ALL |        512 |          0.128 |          0.007 |          0.102 |          0.010 |                  0.031 |      7370.000 |
| CR5L  | Q01      | MASK-COL |        512 |          0.131 |          0.008 |          0.102 |          0.010 |                  0.027 |     25237.833 |
| CR5L  | Q01      | FULL     |        512 |          0.132 |          0.008 |          0.102 |          0.010 |                  0.027 |     32768.000 |
| CR5L  | Q02      | MASK-ALL |        128 |          0.138 |          0.009 |          0.117 |          0.016 |                  0.020 |     14336.000 |
| CR5L  | Q02      | MASK-COL |        128 |          0.137 |          0.009 |          0.117 |          0.016 |                  0.020 |     49280.000 |
| CR5L  | Q02      | FULL     |        128 |          0.138 |          0.009 |          0.117 |          0.016 |                  0.020 |     64000.000 |
| CR5L  | Q03      | MASK-ALL |        512 |          0.144 |          0.009 |          0.133 |          0.010 |                  0.030 |      7418.000 |
| CR5L  | Q03      | MASK-COL |        512 |          0.152 |          0.010 |          0.133 |          0.010 |                  0.028 |     25264.667 |
| CR5L  | Q03      | FULL     |        512 |          0.153 |          0.010 |          0.133 |          0.010 |                  0.028 |     32768.000 |
| CR5L  | M-E      | MASK-ALL |        128 |          0.138 |          0.009 |          0.117 |          0.016 |                  0.020 |     14336.000 |
| CR5L  | M-E      | MASK-COL |        128 |          0.137 |          0.009 |          0.117 |          0.016 |                  0.020 |     49280.000 |
| CR5L  | M-E      | FULL     |        128 |          0.138 |          0.009 |          0.117 |          0.016 |                  0.020 |     64000.000 |
| CR5L  | M-J      | MASK-ALL |        128 |          0.166 |          0.000 |          0.148 |          0.000 |                  0.024 |     14336.000 |
| CR5L  | M-J      | MASK-COL |        128 |          0.165 |          0.000 |          0.148 |          0.000 |                  0.023 |     49280.000 |
| CR5L  | M-J      | FULL     |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     64000.000 |
| CR5L  | M-D      | MASK-ALL |        128 |          0.166 |          0.000 |          0.148 |          0.000 |                  0.024 |     14336.000 |
| CR5L  | M-D      | MASK-COL |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     49280.000 |
| CR5L  | M-D      | FULL     |        128 |          0.163 |          0.000 |          0.148 |          0.000 |                  0.023 |     64000.000 |
| CR5L  | M-I      | MASK-ALL |        128 |          0.244 |          0.000 |          0.164 |          0.000 |                  0.021 |     14336.000 |
| CR5L  | M-I      | MASK-COL |        128 |          0.241 |          0.000 |          0.164 |          0.000 |                  0.021 |     49280.000 |
| CR5L  | M-I      | FULL     |        128 |          0.241 |          0.000 |          0.164 |          0.000 |                  0.021 |     64000.000 |
| CR5L  | RHO-0.25 | MASK-ALL |        128 |          0.166 |          0.000 |          0.148 |          0.000 |                  0.024 |     14336.000 |
| CR5L  | RHO-0.25 | MASK-COL |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     49280.000 |
| CR5L  | RHO-0.25 | FULL     |        128 |          0.163 |          0.000 |          0.148 |          0.000 |                  0.023 |     64000.000 |
| CR5L  | RHO-0.5  | MASK-ALL |        128 |          0.166 |          0.000 |          0.148 |          0.000 |                  0.024 |     14336.000 |
| CR5L  | RHO-0.5  | MASK-COL |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     49280.000 |
| CR5L  | RHO-0.5  | FULL     |        128 |          0.163 |          0.000 |          0.148 |          0.000 |                  0.023 |     64000.000 |
| CR5L  | RHO-0.75 | MASK-ALL |        128 |          0.166 |          0.000 |          0.148 |          0.000 |                  0.024 |     14336.000 |
| CR5L  | RHO-0.75 | MASK-COL |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     49280.000 |
| CR5L  | RHO-0.75 | FULL     |        128 |          0.164 |          0.000 |          0.148 |          0.000 |                  0.023 |     64000.000 |
| CR5L  | KAP-0.75 | MASK-ALL |        128 |          0.013 |          0.000 |          0.016 |          0.000 |                  0.019 |     14336.000 |
| CR5L  | KAP-0.75 | MASK-COL |        128 |          0.013 |          0.000 |          0.016 |          0.000 |                  0.019 |     49280.000 |
| CR5L  | KAP-0.75 | FULL     |        128 |          0.013 |          0.000 |          0.016 |          0.000 |                  0.019 |     64000.000 |
| CR5L  | KAP-1.25 | MASK-ALL |        128 |          0.313 |          0.064 |          0.281 |          0.070 |                  0.028 |     14336.000 |
| CR5L  | KAP-1.25 | MASK-COL |        128 |          0.309 |          0.064 |          0.281 |          0.070 |                  0.027 |     49280.000 |
| CR5L  | KAP-1.25 | FULL     |        128 |          0.309 |          0.063 |          0.281 |          0.070 |                  0.027 |     64000.000 |
| CR5L  | TRN-2    | MASK-ALL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.019 |     14336.000 |
| CR5L  | TRN-2    | MASK-COL |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.019 |     49280.000 |
| CR5L  | TRN-2    | FULL     |        128 |          0.000 |          0.000 |          0.000 |          0.000 |                  0.019 |     64000.000 |
| CR5L  | TRN-3    | MASK-ALL |        128 |          0.177 |          0.000 |          0.148 |          0.000 |                  0.026 |     14336.000 |
| CR5L  | TRN-3    | MASK-COL |        128 |          0.176 |          0.000 |          0.148 |          0.000 |                  0.025 |     49280.000 |
| CR5L  | TRN-3    | FULL     |        128 |          0.177 |          0.000 |          0.148 |          0.000 |                  0.025 |     64000.000 |


## Decoder prediction on actual VALIDATION rows by latent tail stratum

| toy   | stratum   |   n_rows |   frac_subjects |   logloss |     acc |      R2 |
|:------|:----------|---------:|----------------:|----------:|--------:|--------:|
| C1    | (0,2]     |   402336 |           0.835 |     0.614 |   0.659 | nan     |
| C1    | (2,3]     |    79805 |           0.163 |     0.520 |   0.746 | nan     |
| C1    | (3,inf]   |     1321 |           0.003 |     0.415 |   0.805 | nan     |
| C1L   | (0,2]     |   394873 |           0.821 |   nan     | nan     |   0.750 |
| C1L   | (2,3]     |    83032 |           0.168 |   nan     | nan     |   0.888 |
| C1L   | (3,inf]   |     5557 |           0.011 |   nan     | nan     |   0.930 |
| C2    | (0,2]     |   399243 |           0.829 |     0.580 |   0.688 | nan     |
| C2    | (2,3]     |    84219 |           0.171 |     0.472 |   0.774 | nan     |
| C2L   | (0,2]     |   395405 |           0.821 |   nan     | nan     |   0.750 |
| C2L   | (2,3]     |    82647 |           0.169 |   nan     | nan     |   0.882 |
| C2L   | (3,inf]   |     5410 |           0.010 |   nan     | nan     |   0.933 |
| C3    | (0,2]     |   374452 |           0.775 |     0.536 |   0.722 | nan     |
| C3    | (2,3]     |   106710 |           0.220 |     0.460 |   0.778 | nan     |
| C3    | (3,inf]   |     2300 |           0.005 |     0.420 |   0.822 | nan     |
| C3L   | (0,2]     |   361747 |           0.749 |   nan     | nan     |   0.753 |
| C3L   | (2,3]     |   113045 |           0.233 |   nan     | nan     |   0.862 |
| C3L   | (3,inf]   |     8670 |           0.018 |   nan     | nan     |   0.905 |
| CR3   | (0,2]     |   497000 |           0.828 |     0.558 |   0.703 | nan     |
| CR3   | (2,3]     |    93000 |           0.155 |     0.539 |   0.711 | nan     |
| CR3   | (3,inf]   |    10000 |           0.017 |     0.484 |   0.749 | nan     |
| CR3L  | (0,2]     |   489500 |           0.816 |   nan     | nan     |   0.784 |
| CR3L  | (2,3]     |    99500 |           0.166 |   nan     | nan     |   0.841 |
| CR3L  | (3,inf]   |    11000 |           0.018 |   nan     | nan     |   0.915 |
| CR4   | (0,2]     |   513000 |           0.855 |     0.389 |   0.817 | nan     |
| CR4   | (2,3]     |    80500 |           0.134 |     0.392 |   0.811 | nan     |
| CR4   | (3,inf]   |     6500 |           0.011 |     0.428 |   0.784 | nan     |
| CR4L  | (0,2]     |   503500 |           0.839 |   nan     | nan     |   0.787 |
| CR4L  | (2,3]     |    89500 |           0.149 |   nan     | nan     |   0.840 |
| CR4L  | (3,inf]   |     7000 |           0.012 |   nan     | nan     |   0.901 |
| CR5   | (0,2]     |   524500 |           0.874 |     0.536 |   0.719 | nan     |
| CR5   | (2,3]     |    74000 |           0.123 |     0.524 |   0.719 | nan     |
| CR5   | (3,inf]   |     1500 |           0.003 |     0.501 |   0.738 | nan     |
| CR5L  | (0,2]     |   514500 |           0.858 |   nan     | nan     |   0.782 |
| CR5L  | (2,3]     |    80000 |           0.133 |   nan     | nan     |   0.860 |
| CR5L  | (3,inf]   |     5500 |           0.009 |   nan     | nan     |   0.884 |


## Exact GL loss differences dL = L(b) - L(R*) = score term + weight term (x .75) + global term (x .25); b_preferred = dL < 0

| toy   | variant   | law      | b     |      dL |   score_term |   weight_term |   d_global | b_preferred   |
|:------|:----------|:---------|:------|--------:|-------------:|--------------:|-----------:|:--------------|
| C1    | s9        | Q01      | R_Q01 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | Q01      | R_Q05 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | Q02      | R_Q01 | -0.0007 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | Q02      | R_Q05 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | M-E      | R_Q01 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | M-E      | R_Q05 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | M-J      | R_Q01 | -0.0006 |      -0.0004 |       -0.0000 |    -0.0001 | True          |
| C1    | s9        | M-J      | R_Q05 | -0.0007 |      -0.0005 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | M-D      | R_Q01 | -0.0006 |      -0.0004 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | M-D      | R_Q05 | -0.0007 |      -0.0005 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | M-I      | R_Q01 | -0.0006 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | M-I      | R_Q05 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | RHO-0.5  | R_Q01 | -0.0006 |      -0.0004 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | RHO-0.5  | R_Q05 | -0.0007 |      -0.0005 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | KAP-0.75 | R_Q01 | -0.0004 |      -0.0003 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | KAP-0.75 | R_Q05 | -0.0005 |      -0.0003 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | KAP-1.25 | R_Q01 | -0.0007 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | KAP-1.25 | R_Q05 | -0.0007 |      -0.0006 |        0.0000 |    -0.0001 | True          |
| C1    | s9        | TRN-2    | R_Q01 | -0.0006 |      -0.0004 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | TRN-2    | R_Q05 | -0.0007 |      -0.0005 |       -0.0000 |    -0.0002 | True          |
| C1    | s9        | TRN-3    | R_Q01 | -0.0007 |      -0.0004 |       -0.0000 |    -0.0003 | True          |
| C1    | s9        | TRN-3    | R_Q05 | -0.0008 |      -0.0005 |       -0.0000 |    -0.0003 | True          |
| C1    | s9_2      | Q01      | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | Q01      | R_Q05 | -0.0004 |      -0.0004 |        0.0000 |    -0.0000 | True          |
| C1    | s9_2      | Q02      | R_Q01 | -0.0005 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | Q02      | R_Q05 | -0.0005 |      -0.0004 |        0.0000 |    -0.0000 | True          |
| C1    | s9_2      | M-E      | R_Q01 | -0.0005 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-E      | R_Q05 | -0.0004 |      -0.0004 |        0.0000 |    -0.0000 | True          |
| C1    | s9_2      | M-J      | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-J      | R_Q05 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-D      | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-D      | R_Q05 | -0.0005 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-I      | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | M-I      | R_Q05 | -0.0005 |      -0.0004 |        0.0000 |    -0.0000 | True          |
| C1    | s9_2      | RHO-0.5  | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | RHO-0.5  | R_Q05 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | KAP-0.75 | R_Q01 | -0.0003 |      -0.0002 |       -0.0000 |    -0.0001 | True          |
| C1    | s9_2      | KAP-0.75 | R_Q05 | -0.0004 |      -0.0003 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | KAP-1.25 | R_Q01 | -0.0005 |      -0.0005 |        0.0000 |    -0.0000 | True          |
| C1    | s9_2      | KAP-1.25 | R_Q05 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | TRN-2    | R_Q01 | -0.0005 |      -0.0004 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | TRN-2    | R_Q05 | -0.0006 |      -0.0005 |        0.0000 |    -0.0001 | True          |
| C1    | s9_2      | TRN-3    | R_Q01 | -0.0005 |      -0.0004 |       -0.0000 |    -0.0002 | True          |
| C1    | s9_2      | TRN-3    | R_Q05 | -0.0006 |      -0.0005 |       -0.0000 |    -0.0002 | True          |
| C1L   | s9        | Q01      | R_Q01 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | Q01      | R_Q05 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | Q02      | R_Q01 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | Q02      | R_Q05 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-E      | R_Q01 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-E      | R_Q05 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-J      | R_Q01 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-J      | R_Q05 | -0.0027 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-D      | R_Q01 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-D      | R_Q05 | -0.0027 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | M-I      | R_Q01 | -0.0028 |      -0.0019 |       -0.0000 |    -0.0009 | True          |
| C1L   | s9        | M-I      | R_Q05 | -0.0028 |      -0.0019 |       -0.0000 |    -0.0009 | True          |
| C1L   | s9        | RHO-0.5  | R_Q01 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | RHO-0.5  | R_Q05 | -0.0027 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | KAP-0.75 | R_Q01 | -0.0031 |      -0.0021 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9        | KAP-0.75 | R_Q05 | -0.0031 |      -0.0021 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9        | KAP-1.25 | R_Q01 | -0.0020 |      -0.0014 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9        | KAP-1.25 | R_Q05 | -0.0020 |      -0.0014 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9        | TRN-2    | R_Q01 | -0.0033 |      -0.0023 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9        | TRN-2    | R_Q05 | -0.0033 |      -0.0023 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9        | TRN-3    | R_Q01 | -0.0024 |      -0.0016 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9        | TRN-3    | R_Q05 | -0.0024 |      -0.0016 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | Q01      | R_Q01 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | Q01      | R_Q05 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9_2      | Q02      | R_Q01 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | Q02      | R_Q05 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9_2      | M-E      | R_Q01 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-E      | R_Q05 | -0.0026 |      -0.0019 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9_2      | M-J      | R_Q01 | -0.0025 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-J      | R_Q05 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-D      | R_Q01 | -0.0025 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-D      | R_Q05 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-I      | R_Q01 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | M-I      | R_Q05 | -0.0027 |      -0.0019 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | RHO-0.5  | R_Q01 | -0.0025 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | RHO-0.5  | R_Q05 | -0.0026 |      -0.0018 |       -0.0000 |    -0.0008 | True          |
| C1L   | s9_2      | KAP-0.75 | R_Q01 | -0.0030 |      -0.0021 |       -0.0000 |    -0.0009 | True          |
| C1L   | s9_2      | KAP-0.75 | R_Q05 | -0.0030 |      -0.0021 |       -0.0000 |    -0.0009 | True          |
| C1L   | s9_2      | KAP-1.25 | R_Q01 | -0.0019 |      -0.0013 |       -0.0000 |    -0.0006 | True          |
| C1L   | s9_2      | KAP-1.25 | R_Q05 | -0.0019 |      -0.0013 |       -0.0000 |    -0.0006 | True          |
| C1L   | s9_2      | TRN-2    | R_Q01 | -0.0032 |      -0.0023 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9_2      | TRN-2    | R_Q05 | -0.0032 |      -0.0023 |       -0.0000 |    -0.0010 | True          |
| C1L   | s9_2      | TRN-3    | R_Q01 | -0.0023 |      -0.0016 |       -0.0000 |    -0.0007 | True          |
| C1L   | s9_2      | TRN-3    | R_Q05 | -0.0023 |      -0.0016 |       -0.0000 |    -0.0007 | True          |
| CR3   | s9        | Q01      | R_Q01 | -0.0103 |       0.0046 |       -0.0163 |     0.0014 | True          |
| CR3   | s9        | Q01      | R_Q05 | -0.0103 |       0.0043 |       -0.0164 |     0.0018 | True          |
| CR3   | s9        | Q02      | R_Q01 | -0.0115 |       0.0039 |       -0.0164 |     0.0011 | True          |
| CR3   | s9        | Q02      | R_Q05 | -0.0114 |       0.0037 |       -0.0166 |     0.0015 | True          |
| CR3   | s9        | M-E      | R_Q01 | -0.0115 |       0.0039 |       -0.0164 |     0.0011 | True          |
| CR3   | s9        | M-E      | R_Q05 | -0.0114 |       0.0037 |       -0.0166 |     0.0015 | True          |
| CR3   | s9        | M-J      | R_Q01 | -0.0108 |       0.0043 |       -0.0163 |     0.0013 | True          |
| CR3   | s9        | M-J      | R_Q05 | -0.0108 |       0.0040 |       -0.0165 |     0.0016 | True          |
| CR3   | s9        | M-D      | R_Q01 | -0.0107 |       0.0044 |       -0.0164 |     0.0013 | True          |
| CR3   | s9        | M-D      | R_Q05 | -0.0108 |       0.0041 |       -0.0165 |     0.0016 | True          |
| CR3   | s9        | M-I      | R_Q01 | -0.0110 |       0.0041 |       -0.0162 |     0.0012 | True          |
| CR3   | s9        | M-I      | R_Q05 | -0.0109 |       0.0038 |       -0.0163 |     0.0016 | True          |
| CR3   | s9        | RHO-0.5  | R_Q01 | -0.0107 |       0.0044 |       -0.0164 |     0.0013 | True          |
| CR3   | s9        | RHO-0.5  | R_Q05 | -0.0108 |       0.0040 |       -0.0165 |     0.0016 | True          |
| CR3   | s9        | KAP-0.75 | R_Q01 | -0.0115 |       0.0047 |       -0.0173 |     0.0012 | True          |
| CR3   | s9        | KAP-0.75 | R_Q05 | -0.0114 |       0.0045 |       -0.0175 |     0.0016 | True          |
| CR3   | s9        | KAP-1.25 | R_Q01 | -0.0102 |       0.0038 |       -0.0154 |     0.0013 | True          |
| CR3   | s9        | KAP-1.25 | R_Q05 | -0.0104 |       0.0034 |       -0.0154 |     0.0016 | True          |
| CR3   | s9        | TRN-2    | R_Q01 | -0.0101 |       0.0054 |       -0.0168 |     0.0013 | True          |
| CR3   | s9        | TRN-2    | R_Q05 | -0.0100 |       0.0052 |       -0.0169 |     0.0018 | True          |
| CR3   | s9        | TRN-3    | R_Q01 | -0.0112 |       0.0039 |       -0.0163 |     0.0012 | True          |
| CR3   | s9        | TRN-3    | R_Q05 | -0.0111 |       0.0037 |       -0.0164 |     0.0016 | True          |
| CR3   | s9_2      | Q01      | R_Q01 | -0.0077 |       0.0066 |       -0.0137 |    -0.0006 | True          |
| CR3   | s9_2      | Q01      | R_Q05 | -0.0076 |       0.0066 |       -0.0139 |    -0.0003 | True          |
| CR3   | s9_2      | Q02      | R_Q01 | -0.0086 |       0.0060 |       -0.0138 |    -0.0008 | True          |
| CR3   | s9_2      | Q02      | R_Q05 | -0.0086 |       0.0060 |       -0.0141 |    -0.0005 | True          |
| CR3   | s9_2      | M-E      | R_Q01 | -0.0086 |       0.0060 |       -0.0138 |    -0.0008 | True          |
| CR3   | s9_2      | M-E      | R_Q05 | -0.0086 |       0.0060 |       -0.0141 |    -0.0005 | True          |
| CR3   | s9_2      | M-J      | R_Q01 | -0.0078 |       0.0065 |       -0.0137 |    -0.0007 | True          |
| CR3   | s9_2      | M-J      | R_Q05 | -0.0079 |       0.0064 |       -0.0139 |    -0.0004 | True          |
| CR3   | s9_2      | M-D      | R_Q01 | -0.0078 |       0.0066 |       -0.0137 |    -0.0007 | True          |
| CR3   | s9_2      | M-D      | R_Q05 | -0.0079 |       0.0065 |       -0.0140 |    -0.0004 | True          |
| CR3   | s9_2      | M-I      | R_Q01 | -0.0085 |       0.0061 |       -0.0136 |    -0.0010 | True          |
| CR3   | s9_2      | M-I      | R_Q05 | -0.0085 |       0.0060 |       -0.0138 |    -0.0007 | True          |
| CR3   | s9_2      | RHO-0.5  | R_Q01 | -0.0078 |       0.0065 |       -0.0137 |    -0.0007 | True          |
| CR3   | s9_2      | RHO-0.5  | R_Q05 | -0.0079 |       0.0064 |       -0.0139 |    -0.0004 | True          |
| CR3   | s9_2      | KAP-0.75 | R_Q01 | -0.0086 |       0.0069 |       -0.0144 |    -0.0011 | True          |
| CR3   | s9_2      | KAP-0.75 | R_Q05 | -0.0085 |       0.0069 |       -0.0147 |    -0.0007 | True          |
| CR3   | s9_2      | KAP-1.25 | R_Q01 | -0.0073 |       0.0060 |       -0.0130 |    -0.0003 | True          |
| CR3   | s9_2      | KAP-1.25 | R_Q05 | -0.0074 |       0.0058 |       -0.0132 |    -0.0000 | True          |
| CR3   | s9_2      | TRN-2    | R_Q01 | -0.0075 |       0.0073 |       -0.0140 |    -0.0008 | True          |
| CR3   | s9_2      | TRN-2    | R_Q05 | -0.0074 |       0.0073 |       -0.0142 |    -0.0005 | True          |
| CR3   | s9_2      | TRN-3    | R_Q01 | -0.0087 |       0.0060 |       -0.0137 |    -0.0010 | True          |
| CR3   | s9_2      | TRN-3    | R_Q05 | -0.0086 |       0.0059 |       -0.0139 |    -0.0006 | True          |
| CR3L  | s9        | Q01      | R_Q01 | -0.0094 |       0.0057 |       -0.0126 |    -0.0025 | True          |
| CR3L  | s9        | Q01      | R_Q05 | -0.0092 |       0.0064 |       -0.0139 |    -0.0016 | True          |
| CR3L  | s9        | Q02      | R_Q01 | -0.0092 |       0.0057 |       -0.0128 |    -0.0022 | True          |
| CR3L  | s9        | Q02      | R_Q05 | -0.0092 |       0.0064 |       -0.0141 |    -0.0015 | True          |
| CR3L  | s9        | M-E      | R_Q01 | -0.0092 |       0.0057 |       -0.0128 |    -0.0022 | True          |
| CR3L  | s9        | M-E      | R_Q05 | -0.0092 |       0.0064 |       -0.0141 |    -0.0015 | True          |
| CR3L  | s9        | M-J      | R_Q01 | -0.0088 |       0.0056 |       -0.0126 |    -0.0018 | True          |
| CR3L  | s9        | M-J      | R_Q05 | -0.0091 |       0.0062 |       -0.0140 |    -0.0013 | True          |
| CR3L  | s9        | M-D      | R_Q01 | -0.0089 |       0.0055 |       -0.0126 |    -0.0019 | True          |
| CR3L  | s9        | M-D      | R_Q05 | -0.0092 |       0.0062 |       -0.0140 |    -0.0013 | True          |
| CR3L  | s9        | M-I      | R_Q01 | -0.0099 |       0.0059 |       -0.0127 |    -0.0030 | True          |
| CR3L  | s9        | M-I      | R_Q05 | -0.0092 |       0.0068 |       -0.0141 |    -0.0019 | True          |
| CR3L  | s9        | RHO-0.5  | R_Q01 | -0.0089 |       0.0056 |       -0.0126 |    -0.0018 | True          |
| CR3L  | s9        | RHO-0.5  | R_Q05 | -0.0091 |       0.0062 |       -0.0140 |    -0.0013 | True          |
| CR3L  | s9        | KAP-0.75 | R_Q01 | -0.0082 |       0.0069 |       -0.0132 |    -0.0020 | True          |
| CR3L  | s9        | KAP-0.75 | R_Q05 | -0.0082 |       0.0078 |       -0.0146 |    -0.0013 | True          |
| CR3L  | s9        | KAP-1.25 | R_Q01 | -0.0094 |       0.0044 |       -0.0121 |    -0.0017 | True          |
| CR3L  | s9        | KAP-1.25 | R_Q05 | -0.0099 |       0.0048 |       -0.0134 |    -0.0013 | True          |
| CR3L  | s9        | TRN-2    | R_Q01 | -0.0085 |       0.0070 |       -0.0129 |    -0.0026 | True          |
| CR3L  | s9        | TRN-2    | R_Q05 | -0.0080 |       0.0079 |       -0.0143 |    -0.0017 | True          |
| CR3L  | s9        | TRN-3    | R_Q01 | -0.0073 |       0.0067 |       -0.0123 |    -0.0017 | True          |
| CR3L  | s9        | TRN-3    | R_Q05 | -0.0072 |       0.0075 |       -0.0136 |    -0.0011 | True          |
| CR3L  | s9_2      | Q01      | R_Q01 | -0.0093 |       0.0063 |       -0.0094 |    -0.0063 | True          |
| CR3L  | s9_2      | Q01      | R_Q05 | -0.0091 |       0.0069 |       -0.0106 |    -0.0054 | True          |
| CR3L  | s9_2      | Q02      | R_Q01 | -0.0090 |       0.0063 |       -0.0095 |    -0.0058 | True          |
| CR3L  | s9_2      | Q02      | R_Q05 | -0.0089 |       0.0068 |       -0.0107 |    -0.0051 | True          |
| CR3L  | s9_2      | M-E      | R_Q01 | -0.0090 |       0.0063 |       -0.0095 |    -0.0058 | True          |
| CR3L  | s9_2      | M-E      | R_Q05 | -0.0089 |       0.0068 |       -0.0107 |    -0.0051 | True          |
| CR3L  | s9_2      | M-J      | R_Q01 | -0.0082 |       0.0063 |       -0.0094 |    -0.0051 | True          |
| CR3L  | s9_2      | M-J      | R_Q05 | -0.0084 |       0.0067 |       -0.0106 |    -0.0046 | True          |
| CR3L  | s9_2      | M-D      | R_Q01 | -0.0083 |       0.0062 |       -0.0094 |    -0.0052 | True          |
| CR3L  | s9_2      | M-D      | R_Q05 | -0.0085 |       0.0067 |       -0.0106 |    -0.0047 | True          |
| CR3L  | s9_2      | M-I      | R_Q01 | -0.0104 |       0.0064 |       -0.0095 |    -0.0072 | True          |
| CR3L  | s9_2      | M-I      | R_Q05 | -0.0098 |       0.0071 |       -0.0107 |    -0.0061 | True          |
| CR3L  | s9_2      | RHO-0.5  | R_Q01 | -0.0083 |       0.0063 |       -0.0094 |    -0.0052 | True          |
| CR3L  | s9_2      | RHO-0.5  | R_Q05 | -0.0085 |       0.0067 |       -0.0106 |    -0.0046 | True          |
| CR3L  | s9_2      | KAP-0.75 | R_Q01 | -0.0081 |       0.0073 |       -0.0097 |    -0.0057 | True          |
| CR3L  | s9_2      | KAP-0.75 | R_Q05 | -0.0081 |       0.0080 |       -0.0110 |    -0.0051 | True          |
| CR3L  | s9_2      | KAP-1.25 | R_Q01 | -0.0085 |       0.0053 |       -0.0090 |    -0.0048 | True          |
| CR3L  | s9_2      | KAP-1.25 | R_Q05 | -0.0088 |       0.0056 |       -0.0101 |    -0.0043 | True          |
| CR3L  | s9_2      | TRN-2    | R_Q01 | -0.0090 |       0.0073 |       -0.0095 |    -0.0067 | True          |
| CR3L  | s9_2      | TRN-2    | R_Q05 | -0.0085 |       0.0080 |       -0.0107 |    -0.0058 | True          |
| CR3L  | s9_2      | TRN-3    | R_Q01 | -0.0073 |       0.0071 |       -0.0091 |    -0.0054 | True          |
| CR3L  | s9_2      | TRN-3    | R_Q05 | -0.0073 |       0.0076 |       -0.0102 |    -0.0047 | True          |
| CR4   | s9        | Q01      | R_Q01 | -0.0132 |       0.0068 |       -0.0213 |     0.0013 | True          |
| CR4   | s9        | Q01      | R_Q05 | -0.0131 |       0.0053 |       -0.0184 |     0.0000 | True          |
| CR4   | s9        | Q02      | R_Q01 | -0.0131 |       0.0070 |       -0.0215 |     0.0014 | True          |
| CR4   | s9        | Q02      | R_Q05 | -0.0130 |       0.0055 |       -0.0185 |     0.0001 | True          |
| CR4   | s9        | M-E      | R_Q01 | -0.0130 |       0.0070 |       -0.0215 |     0.0015 | True          |
| CR4   | s9        | M-E      | R_Q05 | -0.0130 |       0.0055 |       -0.0185 |     0.0001 | True          |
| CR4   | s9        | M-J      | R_Q01 | -0.0112 |       0.0081 |       -0.0210 |     0.0017 | True          |
| CR4   | s9        | M-J      | R_Q05 | -0.0114 |       0.0065 |       -0.0181 |     0.0003 | True          |
| CR4   | s9        | M-D      | R_Q01 | -0.0113 |       0.0080 |       -0.0211 |     0.0017 | True          |
| CR4   | s9        | M-D      | R_Q05 | -0.0115 |       0.0064 |       -0.0182 |     0.0003 | True          |
| CR4   | s9        | M-I      | R_Q01 | -0.0113 |       0.0080 |       -0.0211 |     0.0018 | True          |
| CR4   | s9        | M-I      | R_Q05 | -0.0113 |       0.0065 |       -0.0182 |     0.0004 | True          |
| CR4   | s9        | RHO-0.5  | R_Q01 | -0.0112 |       0.0081 |       -0.0211 |     0.0017 | True          |
| CR4   | s9        | RHO-0.5  | R_Q05 | -0.0115 |       0.0064 |       -0.0182 |     0.0003 | True          |
| CR4   | s9        | KAP-0.75 | R_Q01 | -0.0121 |       0.0084 |       -0.0222 |     0.0017 | True          |
| CR4   | s9        | KAP-0.75 | R_Q05 | -0.0123 |       0.0065 |       -0.0191 |     0.0003 | True          |
| CR4   | s9        | KAP-1.25 | R_Q01 | -0.0106 |       0.0075 |       -0.0199 |     0.0017 | True          |
| CR4   | s9        | KAP-1.25 | R_Q05 | -0.0108 |       0.0061 |       -0.0171 |     0.0002 | True          |
| CR4   | s9        | TRN-2    | R_Q01 | -0.0128 |       0.0075 |       -0.0218 |     0.0015 | True          |
| CR4   | s9        | TRN-2    | R_Q05 | -0.0127 |       0.0058 |       -0.0188 |     0.0002 | True          |
| CR4   | s9        | TRN-3    | R_Q01 | -0.0111 |       0.0086 |       -0.0213 |     0.0016 | True          |
| CR4   | s9        | TRN-3    | R_Q05 | -0.0110 |       0.0069 |       -0.0184 |     0.0005 | True          |
| CR4   | s9_2      | Q01      | R_Q01 | -0.0069 |       0.0062 |       -0.0100 |    -0.0030 | True          |
| CR4   | s9_2      | Q01      | R_Q05 | -0.0068 |       0.0050 |       -0.0084 |    -0.0034 | True          |
| CR4   | s9_2      | Q02      | R_Q01 | -0.0068 |       0.0063 |       -0.0101 |    -0.0030 | True          |
| CR4   | s9_2      | Q02      | R_Q05 | -0.0068 |       0.0051 |       -0.0085 |    -0.0033 | True          |
| CR4   | s9_2      | M-E      | R_Q01 | -0.0067 |       0.0064 |       -0.0101 |    -0.0030 | True          |
| CR4   | s9_2      | M-E      | R_Q05 | -0.0067 |       0.0051 |       -0.0085 |    -0.0033 | True          |
| CR4   | s9_2      | M-J      | R_Q01 | -0.0061 |       0.0067 |       -0.0098 |    -0.0030 | True          |
| CR4   | s9_2      | M-J      | R_Q05 | -0.0062 |       0.0054 |       -0.0083 |    -0.0033 | True          |
| CR4   | s9_2      | M-D      | R_Q01 | -0.0062 |       0.0066 |       -0.0099 |    -0.0030 | True          |
| CR4   | s9_2      | M-D      | R_Q05 | -0.0062 |       0.0054 |       -0.0083 |    -0.0033 | True          |
| CR4   | s9_2      | M-I      | R_Q01 | -0.0060 |       0.0066 |       -0.0098 |    -0.0028 | True          |
| CR4   | s9_2      | M-I      | R_Q05 | -0.0061 |       0.0053 |       -0.0083 |    -0.0032 | True          |
| CR4   | s9_2      | RHO-0.5  | R_Q01 | -0.0061 |       0.0067 |       -0.0098 |    -0.0030 | True          |
| CR4   | s9_2      | RHO-0.5  | R_Q05 | -0.0062 |       0.0054 |       -0.0083 |    -0.0033 | True          |
| CR4   | s9_2      | KAP-0.75 | R_Q01 | -0.0060 |       0.0073 |       -0.0104 |    -0.0028 | True          |
| CR4   | s9_2      | KAP-0.75 | R_Q05 | -0.0061 |       0.0059 |       -0.0088 |    -0.0032 | True          |
| CR4   | s9_2      | KAP-1.25 | R_Q01 | -0.0063 |       0.0060 |       -0.0093 |    -0.0030 | True          |
| CR4   | s9_2      | KAP-1.25 | R_Q05 | -0.0063 |       0.0048 |       -0.0078 |    -0.0034 | True          |
| CR4   | s9_2      | TRN-2    | R_Q01 | -0.0064 |       0.0067 |       -0.0102 |    -0.0029 | True          |
| CR4   | s9_2      | TRN-2    | R_Q05 | -0.0064 |       0.0054 |       -0.0086 |    -0.0032 | True          |
| CR4   | s9_2      | TRN-3    | R_Q01 | -0.0060 |       0.0070 |       -0.0100 |    -0.0031 | True          |
| CR4   | s9_2      | TRN-3    | R_Q05 | -0.0062 |       0.0056 |       -0.0084 |    -0.0035 | True          |
| CR4L  | s9        | Q01      | R_Q01 | -0.0065 |       0.0082 |       -0.0136 |    -0.0011 | True          |
| CR4L  | s9        | Q01      | R_Q05 | -0.0064 |       0.0080 |       -0.0129 |    -0.0015 | True          |
| CR4L  | s9        | Q02      | R_Q01 | -0.0065 |       0.0081 |       -0.0136 |    -0.0010 | True          |
| CR4L  | s9        | Q02      | R_Q05 | -0.0064 |       0.0079 |       -0.0129 |    -0.0015 | True          |
| CR4L  | s9        | M-E      | R_Q01 | -0.0064 |       0.0081 |       -0.0135 |    -0.0010 | True          |
| CR4L  | s9        | M-E      | R_Q05 | -0.0064 |       0.0080 |       -0.0129 |    -0.0015 | True          |
| CR4L  | s9        | M-J      | R_Q01 | -0.0059 |       0.0082 |       -0.0131 |    -0.0011 | True          |
| CR4L  | s9        | M-J      | R_Q05 | -0.0059 |       0.0080 |       -0.0124 |    -0.0016 | True          |
| CR4L  | s9        | M-D      | R_Q01 | -0.0059 |       0.0083 |       -0.0131 |    -0.0010 | True          |
| CR4L  | s9        | M-D      | R_Q05 | -0.0059 |       0.0081 |       -0.0125 |    -0.0015 | True          |
| CR4L  | s9        | M-I      | R_Q01 | -0.0057 |       0.0083 |       -0.0130 |    -0.0010 | True          |
| CR4L  | s9        | M-I      | R_Q05 | -0.0056 |       0.0081 |       -0.0123 |    -0.0014 | True          |
| CR4L  | s9        | RHO-0.5  | R_Q01 | -0.0059 |       0.0083 |       -0.0131 |    -0.0010 | True          |
| CR4L  | s9        | RHO-0.5  | R_Q05 | -0.0059 |       0.0081 |       -0.0124 |    -0.0015 | True          |
| CR4L  | s9        | KAP-0.75 | R_Q01 | -0.0076 |       0.0082 |       -0.0145 |    -0.0013 | True          |
| CR4L  | s9        | KAP-0.75 | R_Q05 | -0.0076 |       0.0079 |       -0.0137 |    -0.0018 | True          |
| CR4L  | s9        | KAP-1.25 | R_Q01 | -0.0051 |       0.0078 |       -0.0118 |    -0.0011 | True          |
| CR4L  | s9        | KAP-1.25 | R_Q05 | -0.0052 |       0.0077 |       -0.0112 |    -0.0016 | True          |
| CR4L  | s9        | TRN-2    | R_Q01 | -0.0059 |       0.0087 |       -0.0138 |    -0.0008 | True          |
| CR4L  | s9        | TRN-2    | R_Q05 | -0.0059 |       0.0085 |       -0.0131 |    -0.0013 | True          |
| CR4L  | s9        | TRN-3    | R_Q01 | -0.0078 |       0.0071 |       -0.0137 |    -0.0013 | True          |
| CR4L  | s9        | TRN-3    | R_Q05 | -0.0077 |       0.0070 |       -0.0130 |    -0.0017 | True          |
| CR4L  | s9_2      | Q01      | R_Q01 | -0.0078 |       0.0083 |       -0.0058 |    -0.0103 | True          |
| CR4L  | s9_2      | Q01      | R_Q05 | -0.0077 |       0.0093 |       -0.0056 |    -0.0115 | True          |
| CR4L  | s9_2      | Q02      | R_Q01 | -0.0077 |       0.0083 |       -0.0058 |    -0.0102 | True          |
| CR4L  | s9_2      | Q02      | R_Q05 | -0.0076 |       0.0093 |       -0.0056 |    -0.0114 | True          |
| CR4L  | s9_2      | M-E      | R_Q01 | -0.0077 |       0.0083 |       -0.0058 |    -0.0103 | True          |
| CR4L  | s9_2      | M-E      | R_Q05 | -0.0076 |       0.0093 |       -0.0055 |    -0.0114 | True          |
| CR4L  | s9_2      | M-J      | R_Q01 | -0.0082 |       0.0080 |       -0.0056 |    -0.0107 | True          |
| CR4L  | s9_2      | M-J      | R_Q05 | -0.0083 |       0.0090 |       -0.0054 |    -0.0119 | True          |
| CR4L  | s9_2      | M-D      | R_Q01 | -0.0082 |       0.0080 |       -0.0056 |    -0.0106 | True          |
| CR4L  | s9_2      | M-D      | R_Q05 | -0.0083 |       0.0090 |       -0.0054 |    -0.0119 | True          |
| CR4L  | s9_2      | M-I      | R_Q01 | -0.0077 |       0.0082 |       -0.0055 |    -0.0104 | True          |
| CR4L  | s9_2      | M-I      | R_Q05 | -0.0077 |       0.0092 |       -0.0053 |    -0.0116 | True          |
| CR4L  | s9_2      | RHO-0.5  | R_Q01 | -0.0082 |       0.0080 |       -0.0056 |    -0.0106 | True          |
| CR4L  | s9_2      | RHO-0.5  | R_Q05 | -0.0083 |       0.0090 |       -0.0054 |    -0.0119 | True          |
| CR4L  | s9_2      | KAP-0.75 | R_Q01 | -0.0083 |       0.0083 |       -0.0062 |    -0.0104 | True          |
| CR4L  | s9_2      | KAP-0.75 | R_Q05 | -0.0080 |       0.0092 |       -0.0059 |    -0.0113 | True          |
| CR4L  | s9_2      | KAP-1.25 | R_Q01 | -0.0086 |       0.0076 |       -0.0051 |    -0.0110 | True          |
| CR4L  | s9_2      | KAP-1.25 | R_Q05 | -0.0090 |       0.0085 |       -0.0049 |    -0.0126 | True          |
| CR4L  | s9_2      | TRN-2    | R_Q01 | -0.0074 |       0.0085 |       -0.0058 |    -0.0101 | True          |
| CR4L  | s9_2      | TRN-2    | R_Q05 | -0.0072 |       0.0095 |       -0.0056 |    -0.0111 | True          |
| CR4L  | s9_2      | TRN-3    | R_Q01 | -0.0077 |       0.0085 |       -0.0059 |    -0.0103 | True          |
| CR4L  | s9_2      | TRN-3    | R_Q05 | -0.0076 |       0.0095 |       -0.0056 |    -0.0115 | True          |
| CR5   | s9        | Q01      | R_Q01 | -0.0143 |       0.0022 |       -0.0188 |     0.0023 | True          |
| CR5   | s9        | Q01      | R_Q05 | -0.0141 |       0.0037 |       -0.0203 |     0.0025 | True          |
| CR5   | s9        | Q02      | R_Q01 | -0.0150 |       0.0017 |       -0.0189 |     0.0021 | True          |
| CR5   | s9        | Q02      | R_Q05 | -0.0149 |       0.0032 |       -0.0204 |     0.0023 | True          |
| CR5   | s9        | M-E      | R_Q01 | -0.0150 |       0.0017 |       -0.0189 |     0.0021 | True          |
| CR5   | s9        | M-E      | R_Q05 | -0.0149 |       0.0032 |       -0.0204 |     0.0023 | True          |
| CR5   | s9        | M-J      | R_Q01 | -0.0176 |      -0.0006 |       -0.0187 |     0.0016 | True          |
| CR5   | s9        | M-J      | R_Q05 | -0.0178 |       0.0007 |       -0.0202 |     0.0017 | True          |
| CR5   | s9        | M-D      | R_Q01 | -0.0176 |      -0.0007 |       -0.0186 |     0.0017 | True          |
| CR5   | s9        | M-D      | R_Q05 | -0.0178 |       0.0006 |       -0.0201 |     0.0017 | True          |
| CR5   | s9        | M-I      | R_Q01 | -0.0161 |       0.0009 |       -0.0188 |     0.0018 | True          |
| CR5   | s9        | M-I      | R_Q05 | -0.0162 |       0.0022 |       -0.0204 |     0.0020 | True          |
| CR5   | s9        | RHO-0.5  | R_Q01 | -0.0176 |      -0.0006 |       -0.0186 |     0.0016 | True          |
| CR5   | s9        | RHO-0.5  | R_Q05 | -0.0178 |       0.0007 |       -0.0202 |     0.0017 | True          |
| CR5   | s9        | KAP-0.75 | R_Q01 | -0.0173 |       0.0002 |       -0.0194 |     0.0018 | True          |
| CR5   | s9        | KAP-0.75 | R_Q05 | -0.0175 |       0.0016 |       -0.0209 |     0.0019 | True          |
| CR5   | s9        | KAP-1.25 | R_Q01 | -0.0179 |      -0.0015 |       -0.0178 |     0.0015 | True          |
| CR5   | s9        | KAP-1.25 | R_Q05 | -0.0181 |      -0.0003 |       -0.0194 |     0.0016 | True          |
| CR5   | s9        | TRN-2    | R_Q01 | -0.0143 |       0.0027 |       -0.0193 |     0.0023 | True          |
| CR5   | s9        | TRN-2    | R_Q05 | -0.0142 |       0.0042 |       -0.0208 |     0.0024 | True          |
| CR5   | s9        | TRN-3    | R_Q01 | -0.0144 |       0.0020 |       -0.0187 |     0.0023 | True          |
| CR5   | s9        | TRN-3    | R_Q05 | -0.0142 |       0.0035 |       -0.0202 |     0.0025 | True          |
| CR5   | s9_2      | Q01      | R_Q01 | -0.0079 |       0.0050 |       -0.0148 |     0.0019 | True          |
| CR5   | s9_2      | Q01      | R_Q05 | -0.0077 |       0.0070 |       -0.0168 |     0.0021 | True          |
| CR5   | s9_2      | Q02      | R_Q01 | -0.0085 |       0.0047 |       -0.0149 |     0.0017 | True          |
| CR5   | s9_2      | Q02      | R_Q05 | -0.0083 |       0.0066 |       -0.0169 |     0.0019 | True          |
| CR5   | s9_2      | M-E      | R_Q01 | -0.0085 |       0.0047 |       -0.0149 |     0.0017 | True          |
| CR5   | s9_2      | M-E      | R_Q05 | -0.0083 |       0.0066 |       -0.0169 |     0.0019 | True          |
| CR5   | s9_2      | M-J      | R_Q01 | -0.0107 |       0.0029 |       -0.0148 |     0.0012 | True          |
| CR5   | s9_2      | M-J      | R_Q05 | -0.0109 |       0.0046 |       -0.0167 |     0.0013 | True          |
| CR5   | s9_2      | M-D      | R_Q01 | -0.0106 |       0.0029 |       -0.0147 |     0.0012 | True          |
| CR5   | s9_2      | M-D      | R_Q05 | -0.0108 |       0.0046 |       -0.0167 |     0.0013 | True          |
| CR5   | s9_2      | M-I      | R_Q01 | -0.0095 |       0.0039 |       -0.0148 |     0.0014 | True          |
| CR5   | s9_2      | M-I      | R_Q05 | -0.0096 |       0.0057 |       -0.0168 |     0.0016 | True          |
| CR5   | s9_2      | RHO-0.5  | R_Q01 | -0.0106 |       0.0029 |       -0.0147 |     0.0012 | True          |
| CR5   | s9_2      | RHO-0.5  | R_Q05 | -0.0108 |       0.0046 |       -0.0167 |     0.0013 | True          |
| CR5   | s9_2      | KAP-0.75 | R_Q01 | -0.0104 |       0.0037 |       -0.0153 |     0.0013 | True          |
| CR5   | s9_2      | KAP-0.75 | R_Q05 | -0.0105 |       0.0055 |       -0.0174 |     0.0013 | True          |
| CR5   | s9_2      | KAP-1.25 | R_Q01 | -0.0109 |       0.0021 |       -0.0141 |     0.0012 | True          |
| CR5   | s9_2      | KAP-1.25 | R_Q05 | -0.0112 |       0.0036 |       -0.0161 |     0.0013 | True          |
| CR5   | s9_2      | TRN-2    | R_Q01 | -0.0080 |       0.0055 |       -0.0152 |     0.0017 | True          |
| CR5   | s9_2      | TRN-2    | R_Q05 | -0.0079 |       0.0075 |       -0.0172 |     0.0019 | True          |
| CR5   | s9_2      | TRN-3    | R_Q01 | -0.0080 |       0.0049 |       -0.0147 |     0.0018 | True          |
| CR5   | s9_2      | TRN-3    | R_Q05 | -0.0079 |       0.0068 |       -0.0167 |     0.0020 | True          |
| CR5L  | s9        | Q01      | R_Q01 | -0.0026 |       0.0049 |       -0.0065 |    -0.0010 | True          |
| CR5L  | s9        | Q01      | R_Q05 | -0.0024 |       0.0030 |       -0.0036 |    -0.0018 | True          |
| CR5L  | s9        | Q02      | R_Q01 | -0.0022 |       0.0050 |       -0.0063 |    -0.0009 | True          |
| CR5L  | s9        | Q02      | R_Q05 | -0.0021 |       0.0031 |       -0.0035 |    -0.0016 | True          |
| CR5L  | s9        | M-E      | R_Q01 | -0.0022 |       0.0050 |       -0.0063 |    -0.0009 | True          |
| CR5L  | s9        | M-E      | R_Q05 | -0.0021 |       0.0031 |       -0.0035 |    -0.0016 | True          |
| CR5L  | s9        | M-J      | R_Q01 | -0.0017 |       0.0056 |       -0.0063 |    -0.0009 | True          |
| CR5L  | s9        | M-J      | R_Q05 | -0.0019 |       0.0034 |       -0.0035 |    -0.0017 | True          |
| CR5L  | s9        | M-D      | R_Q01 | -0.0016 |       0.0056 |       -0.0063 |    -0.0009 | True          |
| CR5L  | s9        | M-D      | R_Q05 | -0.0018 |       0.0034 |       -0.0035 |    -0.0017 | True          |
| CR5L  | s9        | M-I      | R_Q01 | -0.0025 |       0.0047 |       -0.0064 |    -0.0008 | True          |
| CR5L  | s9        | M-I      | R_Q05 | -0.0021 |       0.0030 |       -0.0036 |    -0.0015 | True          |
| CR5L  | s9        | RHO-0.5  | R_Q01 | -0.0016 |       0.0056 |       -0.0063 |    -0.0009 | True          |
| CR5L  | s9        | RHO-0.5  | R_Q05 | -0.0018 |       0.0034 |       -0.0035 |    -0.0017 | True          |
| CR5L  | s9        | KAP-0.75 | R_Q01 | -0.0019 |       0.0061 |       -0.0068 |    -0.0013 | True          |
| CR5L  | s9        | KAP-0.75 | R_Q05 | -0.0024 |       0.0035 |       -0.0037 |    -0.0022 | True          |
| CR5L  | s9        | KAP-1.25 | R_Q01 | -0.0015 |       0.0050 |       -0.0059 |    -0.0006 | True          |
| CR5L  | s9        | KAP-1.25 | R_Q05 | -0.0015 |       0.0032 |       -0.0033 |    -0.0013 | True          |
| CR5L  | s9        | TRN-2    | R_Q01 | -0.0029 |       0.0050 |       -0.0066 |    -0.0013 | True          |
| CR5L  | s9        | TRN-2    | R_Q05 | -0.0028 |       0.0030 |       -0.0036 |    -0.0021 | True          |
| CR5L  | s9        | TRN-3    | R_Q01 | -0.0025 |       0.0050 |       -0.0064 |    -0.0011 | True          |
| CR5L  | s9        | TRN-3    | R_Q05 | -0.0024 |       0.0030 |       -0.0036 |    -0.0018 | True          |
| CR5L  | s9_2      | Q01      | R_Q01 | -0.0046 |       0.0058 |       -0.0028 |    -0.0075 | True          |
| CR5L  | s9_2      | Q01      | R_Q05 | -0.0044 |       0.0054 |       -0.0013 |    -0.0086 | True          |
| CR5L  | s9_2      | Q02      | R_Q01 | -0.0042 |       0.0059 |       -0.0028 |    -0.0073 | True          |
| CR5L  | s9_2      | Q02      | R_Q05 | -0.0041 |       0.0055 |       -0.0013 |    -0.0084 | True          |
| CR5L  | s9_2      | M-E      | R_Q01 | -0.0042 |       0.0059 |       -0.0028 |    -0.0073 | True          |
| CR5L  | s9_2      | M-E      | R_Q05 | -0.0041 |       0.0055 |       -0.0013 |    -0.0084 | True          |
| CR5L  | s9_2      | M-J      | R_Q01 | -0.0047 |       0.0059 |       -0.0027 |    -0.0080 | True          |
| CR5L  | s9_2      | M-J      | R_Q05 | -0.0049 |       0.0055 |       -0.0011 |    -0.0093 | True          |
| CR5L  | s9_2      | M-D      | R_Q01 | -0.0046 |       0.0060 |       -0.0027 |    -0.0079 | True          |
| CR5L  | s9_2      | M-D      | R_Q05 | -0.0048 |       0.0055 |       -0.0011 |    -0.0092 | True          |
| CR5L  | s9_2      | M-I      | R_Q01 | -0.0036 |       0.0062 |       -0.0028 |    -0.0070 | True          |
| CR5L  | s9_2      | M-I      | R_Q05 | -0.0033 |       0.0061 |       -0.0013 |    -0.0081 | True          |
| CR5L  | s9_2      | RHO-0.5  | R_Q01 | -0.0047 |       0.0059 |       -0.0027 |    -0.0079 | True          |
| CR5L  | s9_2      | RHO-0.5  | R_Q05 | -0.0049 |       0.0055 |       -0.0011 |    -0.0093 | True          |
| CR5L  | s9_2      | KAP-0.75 | R_Q01 | -0.0061 |       0.0055 |       -0.0027 |    -0.0089 | True          |
| CR5L  | s9_2      | KAP-0.75 | R_Q05 | -0.0065 |       0.0047 |       -0.0010 |    -0.0102 | True          |
| CR5L  | s9_2      | KAP-1.25 | R_Q01 | -0.0036 |       0.0062 |       -0.0026 |    -0.0072 | True          |
| CR5L  | s9_2      | KAP-1.25 | R_Q05 | -0.0036 |       0.0061 |       -0.0012 |    -0.0085 | True          |
| CR5L  | s9_2      | TRN-2    | R_Q01 | -0.0058 |       0.0053 |       -0.0028 |    -0.0083 | True          |
| CR5L  | s9_2      | TRN-2    | R_Q05 | -0.0058 |       0.0048 |       -0.0012 |    -0.0094 | True          |
| CR5L  | s9_2      | TRN-3    | R_Q01 | -0.0050 |       0.0056 |       -0.0028 |    -0.0078 | True          |
| CR5L  | s9_2      | TRN-3    | R_Q05 | -0.0049 |       0.0052 |       -0.0012 |    -0.0089 | True          |


## Latent-block bootstrap of loss differences (400 resamples)

| toy   | variant   | measure   |   seed |   n_blocks |   dL_Q05_star |   dL_Q01_star |   dL_Q05_Q01 |   sd_Q05_star |   sd_Q01_star |   sd_Q05_Q01 |   p_Q05_below_star |   p_Q01_below_star |   p_Q05_below_Q01 |
|:------|:----------|:----------|-------:|-----------:|--------------:|--------------:|-------------:|--------------:|--------------:|-------------:|-------------------:|-------------------:|------------------:|
| C1    | s9        | Q05@C     |     31 |        512 |       -0.0007 |       -0.0006 |      -0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.8350 |
| C1    | s9        | Q05@C     |     73 |        512 |       -0.0007 |       -0.0007 |      -0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.8950 |
| C1    | s9        | Q05@C     |    101 |        512 |       -0.0008 |       -0.0007 |      -0.0000 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.6050 |
| C1    | s9        | Q05@D     |    211 |       1024 |       -0.0007 |       -0.0007 |      -0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.6325 |
| C1    | s9        | Q05@D     |    613 |       1024 |       -0.0007 |       -0.0007 |      -0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.6250 |
| C1    | s9        | Q01@C     |     31 |        512 |       -0.0005 |       -0.0006 |       0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.1450 |
| C1    | s9_2      | Q05@C     |     31 |        512 |       -0.0005 |       -0.0005 |      -0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.8350 |
| C1    | s9_2      | Q05@C     |     73 |        512 |       -0.0006 |       -0.0005 |      -0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.8925 |
| C1    | s9_2      | Q05@C     |    101 |        512 |       -0.0006 |       -0.0006 |      -0.0000 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.5975 |
| C1    | s9_2      | Q05@D     |    211 |       1024 |       -0.0006 |       -0.0006 |      -0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.6350 |
| C1    | s9_2      | Q05@D     |    613 |       1024 |       -0.0006 |       -0.0006 |      -0.0000 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.6300 |
| C1    | s9_2      | Q01@C     |     31 |        512 |       -0.0004 |       -0.0005 |       0.0001 |        0.0001 |        0.0001 |       0.0001 |             1.0000 |             1.0000 |            0.1500 |
| C1L   | s9        | Q05@C     |     31 |        512 |       -0.0027 |       -0.0026 |      -0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.8025 |
| C1L   | s9        | Q05@C     |     73 |        512 |       -0.0026 |       -0.0026 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4325 |
| C1L   | s9        | Q05@C     |    101 |        512 |       -0.0029 |       -0.0029 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4125 |
| C1L   | s9        | Q05@D     |    211 |       1024 |       -0.0028 |       -0.0028 |      -0.0000 |        0.0001 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.8550 |
| C1L   | s9        | Q05@D     |    613 |       1024 |       -0.0029 |       -0.0029 |      -0.0000 |        0.0001 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.7750 |
| C1L   | s9        | Q01@C     |     31 |        512 |       -0.0027 |       -0.0027 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.2100 |
| C1L   | s9_2      | Q05@C     |     31 |        512 |       -0.0026 |       -0.0025 |      -0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.8050 |
| C1L   | s9_2      | Q05@C     |     73 |        512 |       -0.0025 |       -0.0025 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4375 |
| C1L   | s9_2      | Q05@C     |    101 |        512 |       -0.0028 |       -0.0028 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4075 |
| C1L   | s9_2      | Q05@D     |    211 |       1024 |       -0.0027 |       -0.0027 |      -0.0000 |        0.0001 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.8500 |
| C1L   | s9_2      | Q05@D     |    613 |       1024 |       -0.0028 |       -0.0028 |      -0.0000 |        0.0001 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.7700 |
| C1L   | s9_2      | Q01@C     |     31 |        512 |       -0.0026 |       -0.0026 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.2075 |
| C2    | s9        | Q05@C     |     31 |        512 |       -0.0022 |       -0.0021 |      -0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9950 |
| C2    | s9        | Q05@C     |     73 |        512 |       -0.0021 |       -0.0021 |      -0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.7475 |
| C2    | s9        | Q05@C     |    101 |        512 |       -0.0024 |       -0.0023 |      -0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9825 |
| C2    | s9        | Q05@D     |    211 |       1024 |       -0.0018 |       -0.0018 |      -0.0000 |        0.0002 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.5525 |
| C2    | s9        | Q05@D     |    613 |       1024 |       -0.0017 |       -0.0018 |       0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.0675 |
| C2    | s9        | Q01@C     |     31 |        512 |       -0.0016 |       -0.0017 |       0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0000 |
| C2    | s9_2      | Q05@C     |     31 |        512 |       -0.0030 |       -0.0029 |      -0.0001 |        0.0003 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9950 |
| C2    | s9_2      | Q05@C     |     73 |        512 |       -0.0028 |       -0.0028 |      -0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.7575 |
| C2    | s9_2      | Q05@C     |    101 |        512 |       -0.0031 |       -0.0030 |      -0.0001 |        0.0003 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9850 |
| C2    | s9_2      | Q05@D     |    211 |       1024 |       -0.0026 |       -0.0026 |      -0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.5575 |
| C2    | s9_2      | Q05@D     |    613 |       1024 |       -0.0025 |       -0.0026 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0650 |
| C2    | s9_2      | Q01@C     |     31 |        512 |       -0.0024 |       -0.0024 |       0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0000 |
| C2L   | s9        | Q05@C     |     31 |        512 |       -0.0060 |       -0.0059 |      -0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.8275 |
| C2L   | s9        | Q05@C     |     73 |        512 |       -0.0062 |       -0.0062 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1700 |
| C2L   | s9        | Q05@C     |    101 |        512 |       -0.0061 |       -0.0061 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1825 |
| C2L   | s9        | Q05@D     |    211 |       1024 |       -0.0065 |       -0.0065 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0675 |
| C2L   | s9        | Q05@D     |    613 |       1024 |       -0.0065 |       -0.0065 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.1250 |
| C2L   | s9        | Q01@C     |     31 |        512 |       -0.0059 |       -0.0059 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1375 |
| C2L   | s9_2      | Q05@C     |     31 |        512 |       -0.0052 |       -0.0052 |      -0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.8375 |
| C2L   | s9_2      | Q05@C     |     73 |        512 |       -0.0055 |       -0.0055 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1700 |
| C2L   | s9_2      | Q05@C     |    101 |        512 |       -0.0054 |       -0.0054 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1825 |
| C2L   | s9_2      | Q05@D     |    211 |       1024 |       -0.0057 |       -0.0057 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0575 |
| C2L   | s9_2      | Q05@D     |    613 |       1024 |       -0.0057 |       -0.0057 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.1250 |
| C2L   | s9_2      | Q01@C     |     31 |        512 |       -0.0052 |       -0.0052 |       0.0000 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            0.1350 |
| C3    | s9        | Q05@C     |     31 |        512 |       -0.0019 |       -0.0019 |      -0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9850 |
| C3    | s9        | Q05@C     |     73 |        512 |       -0.0021 |       -0.0021 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4775 |
| C3    | s9        | Q05@C     |    101 |        512 |       -0.0018 |       -0.0018 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4000 |
| C3    | s9        | Q05@D     |    211 |       1024 |       -0.0021 |       -0.0021 |       0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.0050 |
| C3    | s9        | Q05@D     |    613 |       1024 |       -0.0020 |       -0.0020 |       0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.0275 |
| C3    | s9        | Q01@C     |     31 |        512 |       -0.0021 |       -0.0021 |       0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0100 |
| C3    | s9_2      | Q05@C     |     31 |        512 |       -0.0025 |       -0.0025 |      -0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.9825 |
| C3    | s9_2      | Q05@C     |     73 |        512 |       -0.0027 |       -0.0027 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4375 |
| C3    | s9_2      | Q05@C     |    101 |        512 |       -0.0024 |       -0.0024 |       0.0000 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.4225 |
| C3    | s9_2      | Q05@D     |    211 |       1024 |       -0.0027 |       -0.0027 |       0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.0050 |
| C3    | s9_2      | Q05@D     |    613 |       1024 |       -0.0026 |       -0.0026 |       0.0000 |        0.0001 |        0.0001 |       0.0000 |             1.0000 |             1.0000 |            0.0275 |
| C3    | s9_2      | Q01@C     |     31 |        512 |       -0.0027 |       -0.0028 |       0.0001 |        0.0002 |        0.0002 |       0.0000 |             1.0000 |             1.0000 |            0.0125 |
| C3L   | s9        | Q05@C     |     31 |        512 |       -0.0059 |       -0.0058 |      -0.0001 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.8950 |
| C3L   | s9        | Q05@C     |     73 |        512 |       -0.0054 |       -0.0054 |       0.0000 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.3500 |
| C3L   | s9        | Q05@C     |    101 |        512 |       -0.0053 |       -0.0054 |       0.0001 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.1750 |
| C3L   | s9        | Q05@D     |    211 |       1024 |       -0.0054 |       -0.0055 |       0.0001 |        0.0002 |        0.0002 |       0.0001 |             1.0000 |             1.0000 |            0.0175 |
| C3L   | s9        | Q05@D     |    613 |       1024 |       -0.0052 |       -0.0053 |       0.0001 |        0.0002 |        0.0002 |       0.0001 |             1.0000 |             1.0000 |            0.0300 |
| C3L   | s9        | Q01@C     |     31 |        512 |       -0.0053 |       -0.0054 |       0.0001 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.0950 |
| C3L   | s9_2      | Q05@C     |     31 |        512 |       -0.0062 |       -0.0061 |      -0.0001 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.8950 |
| C3L   | s9_2      | Q05@C     |     73 |        512 |       -0.0057 |       -0.0057 |       0.0000 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.3400 |
| C3L   | s9_2      | Q05@C     |    101 |        512 |       -0.0056 |       -0.0057 |       0.0001 |        0.0004 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.1750 |
| C3L   | s9_2      | Q05@D     |    211 |       1024 |       -0.0057 |       -0.0059 |       0.0001 |        0.0002 |        0.0002 |       0.0001 |             1.0000 |             1.0000 |            0.0175 |
| C3L   | s9_2      | Q05@D     |    613 |       1024 |       -0.0055 |       -0.0056 |       0.0001 |        0.0003 |        0.0002 |       0.0001 |             1.0000 |             1.0000 |            0.0300 |
| C3L   | s9_2      | Q01@C     |     31 |        512 |       -0.0056 |       -0.0057 |       0.0001 |        0.0003 |        0.0003 |       0.0001 |             1.0000 |             1.0000 |            0.0925 |
| CR3   | s9        | Q05@C     |     31 |        128 |       -0.0108 |       -0.0107 |      -0.0001 |        0.0011 |        0.0012 |       0.0001 |             1.0000 |             1.0000 |            0.7250 |
| CR3   | s9        | Q05@C     |     73 |        128 |       -0.0104 |       -0.0105 |       0.0002 |        0.0009 |        0.0010 |       0.0001 |             1.0000 |             1.0000 |            0.0125 |
| CR3   | s9        | Q05@C     |    101 |        128 |       -0.0110 |       -0.0111 |       0.0001 |        0.0010 |        0.0010 |       0.0001 |             1.0000 |             1.0000 |            0.2550 |
| CR3   | s9        | Q05@D     |    211 |        512 |       -0.0107 |       -0.0108 |       0.0001 |        0.0005 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.1225 |
| CR3   | s9        | Q05@D     |    613 |        512 |       -0.0110 |       -0.0110 |       0.0001 |        0.0005 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.1750 |
| CR3   | s9        | Q01@C     |     31 |        512 |       -0.0103 |       -0.0103 |       0.0001 |        0.0006 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0550 |
| CR3   | s9_2      | Q05@C     |     31 |        128 |       -0.0079 |       -0.0078 |      -0.0000 |        0.0012 |        0.0012 |       0.0001 |             1.0000 |             1.0000 |            0.7100 |
| CR3   | s9_2      | Q05@C     |     73 |        128 |       -0.0081 |       -0.0082 |       0.0001 |        0.0010 |        0.0010 |       0.0000 |             1.0000 |             1.0000 |            0.0075 |
| CR3   | s9_2      | Q05@C     |    101 |        128 |       -0.0082 |       -0.0082 |       0.0000 |        0.0010 |        0.0011 |       0.0000 |             1.0000 |             1.0000 |            0.4350 |
| CR3   | s9_2      | Q05@D     |    211 |        512 |       -0.0079 |       -0.0079 |       0.0000 |        0.0005 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.2025 |
| CR3   | s9_2      | Q05@D     |    613 |        512 |       -0.0081 |       -0.0081 |       0.0000 |        0.0006 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.3350 |
| CR3   | s9_2      | Q01@C     |     31 |        512 |       -0.0076 |       -0.0077 |       0.0000 |        0.0006 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.0825 |
| CR3L  | s9        | Q05@C     |     31 |        128 |       -0.0092 |       -0.0089 |      -0.0002 |        0.0014 |        0.0014 |       0.0002 |             1.0000 |             1.0000 |            0.8475 |
| CR3L  | s9        | Q05@C     |     73 |        128 |       -0.0082 |       -0.0086 |       0.0004 |        0.0014 |        0.0013 |       0.0002 |             1.0000 |             1.0000 |            0.0575 |
| CR3L  | s9        | Q05@C     |    101 |        128 |       -0.0094 |       -0.0097 |       0.0003 |        0.0018 |        0.0017 |       0.0002 |             1.0000 |             1.0000 |            0.0600 |
| CR3L  | s9        | Q05@D     |    211 |        512 |       -0.0095 |       -0.0097 |       0.0002 |        0.0008 |        0.0008 |       0.0001 |             1.0000 |             1.0000 |            0.0050 |
| CR3L  | s9        | Q05@D     |    613 |        512 |       -0.0087 |       -0.0091 |       0.0004 |        0.0008 |        0.0008 |       0.0001 |             1.0000 |             1.0000 |            0.0000 |
| CR3L  | s9        | Q01@C     |     31 |        512 |       -0.0092 |       -0.0094 |       0.0002 |        0.0009 |        0.0008 |       0.0001 |             1.0000 |             1.0000 |            0.0550 |
| CR3L  | s9_2      | Q05@C     |     31 |        128 |       -0.0085 |       -0.0083 |      -0.0002 |        0.0012 |        0.0012 |       0.0002 |             1.0000 |             1.0000 |            0.8450 |
| CR3L  | s9_2      | Q05@C     |     73 |        128 |       -0.0084 |       -0.0088 |       0.0004 |        0.0010 |        0.0010 |       0.0002 |             1.0000 |             1.0000 |            0.0450 |
| CR3L  | s9_2      | Q05@C     |    101 |        128 |       -0.0095 |       -0.0098 |       0.0003 |        0.0014 |        0.0014 |       0.0002 |             1.0000 |             1.0000 |            0.0500 |
| CR3L  | s9_2      | Q05@D     |    211 |        512 |       -0.0096 |       -0.0098 |       0.0002 |        0.0007 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.0075 |
| CR3L  | s9_2      | Q05@D     |    613 |        512 |       -0.0090 |       -0.0094 |       0.0003 |        0.0007 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.0000 |
| CR3L  | s9_2      | Q01@C     |     31 |        512 |       -0.0091 |       -0.0093 |       0.0002 |        0.0007 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.0450 |
| CR4   | s9        | Q05@C     |     31 |        512 |       -0.0115 |       -0.0113 |      -0.0002 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.9475 |
| CR4   | s9        | Q05@C     |     73 |        512 |       -0.0128 |       -0.0128 |       0.0000 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.4225 |
| CR4   | s9        | Q05@C     |    101 |        512 |       -0.0101 |       -0.0100 |      -0.0002 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.8450 |
| CR4   | s9        | Q05@D     |    211 |       1024 |       -0.0110 |       -0.0112 |       0.0002 |        0.0006 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0275 |
| CR4   | s9        | Q05@D     |    613 |       1024 |       -0.0101 |       -0.0101 |      -0.0001 |        0.0006 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.7125 |
| CR4   | s9        | Q01@C     |     31 |        512 |       -0.0131 |       -0.0132 |       0.0002 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.0975 |
| CR4   | s9_2      | Q05@C     |     31 |        512 |       -0.0062 |       -0.0062 |      -0.0001 |        0.0004 |        0.0004 |       0.0001 |             1.0000 |             1.0000 |            0.9100 |
| CR4   | s9_2      | Q05@C     |     73 |        512 |       -0.0068 |       -0.0068 |       0.0000 |        0.0004 |        0.0004 |       0.0001 |             1.0000 |             1.0000 |            0.4700 |
| CR4   | s9_2      | Q05@C     |    101 |        512 |       -0.0055 |       -0.0054 |      -0.0002 |        0.0003 |        0.0004 |       0.0001 |             1.0000 |             1.0000 |            1.0000 |
| CR4   | s9_2      | Q05@D     |    211 |       1024 |       -0.0061 |       -0.0060 |      -0.0001 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            1.0000 |
| CR4   | s9_2      | Q05@D     |    613 |       1024 |       -0.0058 |       -0.0056 |      -0.0002 |        0.0003 |        0.0003 |       0.0000 |             1.0000 |             1.0000 |            1.0000 |
| CR4   | s9_2      | Q01@C     |     31 |        512 |       -0.0068 |       -0.0069 |       0.0000 |        0.0004 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.2300 |
| CR4L  | s9        | Q05@C     |     31 |        512 |       -0.0059 |       -0.0059 |      -0.0000 |        0.0008 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.7375 |
| CR4L  | s9        | Q05@C     |     73 |        512 |       -0.0065 |       -0.0064 |      -0.0000 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.6750 |
| CR4L  | s9        | Q05@C     |    101 |        512 |       -0.0061 |       -0.0062 |       0.0001 |        0.0008 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.0500 |
| CR4L  | s9        | Q05@D     |    211 |       1024 |       -0.0058 |       -0.0059 |       0.0001 |        0.0006 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.0200 |
| CR4L  | s9        | Q05@D     |    613 |       1024 |       -0.0066 |       -0.0068 |       0.0001 |        0.0006 |        0.0006 |       0.0000 |             1.0000 |             1.0000 |            0.0000 |
| CR4L  | s9        | Q01@C     |     31 |        512 |       -0.0064 |       -0.0065 |       0.0000 |        0.0009 |        0.0009 |       0.0001 |             1.0000 |             1.0000 |            0.2375 |
| CR4L  | s9_2      | Q05@C     |     31 |        512 |       -0.0083 |       -0.0082 |      -0.0001 |        0.0008 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.8550 |
| CR4L  | s9_2      | Q05@C     |     73 |        512 |       -0.0084 |       -0.0084 |      -0.0000 |        0.0008 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.6725 |
| CR4L  | s9_2      | Q05@C     |    101 |        512 |       -0.0068 |       -0.0070 |       0.0002 |        0.0007 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.0125 |
| CR4L  | s9_2      | Q05@D     |    211 |       1024 |       -0.0070 |       -0.0071 |       0.0002 |        0.0006 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0000 |
| CR4L  | s9_2      | Q05@D     |    613 |       1024 |       -0.0072 |       -0.0073 |       0.0001 |        0.0006 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0175 |
| CR4L  | s9_2      | Q01@C     |     31 |        512 |       -0.0077 |       -0.0078 |       0.0001 |        0.0009 |        0.0008 |       0.0001 |             1.0000 |             1.0000 |            0.1500 |
| CR5   | s9        | Q05@C     |     31 |        128 |       -0.0178 |       -0.0176 |      -0.0002 |        0.0015 |        0.0014 |       0.0002 |             1.0000 |             1.0000 |            0.8225 |
| CR5   | s9        | Q05@C     |     73 |        128 |       -0.0134 |       -0.0136 |       0.0002 |        0.0015 |        0.0014 |       0.0002 |             1.0000 |             1.0000 |            0.0875 |
| CR5   | s9        | Q05@C     |    101 |        128 |       -0.0161 |       -0.0160 |      -0.0000 |        0.0015 |        0.0013 |       0.0002 |             1.0000 |             1.0000 |            0.6225 |
| CR5   | s9        | Q05@D     |    211 |        512 |       -0.0141 |       -0.0143 |       0.0002 |        0.0007 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0250 |
| CR5   | s9        | Q05@D     |    613 |        512 |       -0.0136 |       -0.0139 |       0.0002 |        0.0007 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0050 |
| CR5   | s9        | Q01@C     |     31 |        512 |       -0.0141 |       -0.0143 |       0.0002 |        0.0008 |        0.0007 |       0.0001 |             1.0000 |             1.0000 |            0.0325 |
| CR5   | s9_2      | Q05@C     |     31 |        128 |       -0.0108 |       -0.0106 |      -0.0002 |        0.0014 |        0.0012 |       0.0002 |             1.0000 |             1.0000 |            0.8500 |
| CR5   | s9_2      | Q05@C     |     73 |        128 |       -0.0071 |       -0.0073 |       0.0003 |        0.0014 |        0.0012 |       0.0002 |             1.0000 |             1.0000 |            0.0850 |
| CR5   | s9_2      | Q05@C     |    101 |        128 |       -0.0095 |       -0.0095 |      -0.0001 |        0.0014 |        0.0012 |       0.0002 |             1.0000 |             1.0000 |            0.6625 |
| CR5   | s9_2      | Q05@D     |    211 |        512 |       -0.0077 |       -0.0079 |       0.0002 |        0.0006 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0275 |
| CR5   | s9_2      | Q05@D     |    613 |        512 |       -0.0073 |       -0.0075 |       0.0003 |        0.0006 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0050 |
| CR5   | s9_2      | Q01@C     |     31 |        512 |       -0.0077 |       -0.0079 |       0.0002 |        0.0007 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0275 |
| CR5L  | s9        | Q05@C     |     31 |        128 |       -0.0018 |       -0.0016 |      -0.0002 |        0.0006 |        0.0008 |       0.0003 |             1.0000 |             0.9750 |            0.7975 |
| CR5L  | s9        | Q05@C     |     73 |        128 |       -0.0022 |       -0.0025 |       0.0002 |        0.0007 |        0.0009 |       0.0003 |             1.0000 |             0.9950 |            0.1825 |
| CR5L  | s9        | Q05@C     |    101 |        128 |       -0.0031 |       -0.0035 |       0.0004 |        0.0007 |        0.0009 |       0.0003 |             1.0000 |             1.0000 |            0.0500 |
| CR5L  | s9        | Q05@D     |    211 |        512 |       -0.0025 |       -0.0026 |       0.0001 |        0.0003 |        0.0004 |       0.0001 |             1.0000 |             1.0000 |            0.1175 |
| CR5L  | s9        | Q05@D     |    613 |        512 |       -0.0027 |       -0.0032 |       0.0005 |        0.0004 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0000 |
| CR5L  | s9        | Q01@C     |     31 |        512 |       -0.0024 |       -0.0026 |       0.0002 |        0.0004 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.0725 |
| CR5L  | s9_2      | Q05@C     |     31 |        128 |       -0.0048 |       -0.0046 |      -0.0002 |        0.0012 |        0.0010 |       0.0002 |             1.0000 |             1.0000 |            0.8450 |
| CR5L  | s9_2      | Q05@C     |     73 |        128 |       -0.0039 |       -0.0042 |       0.0003 |        0.0012 |        0.0011 |       0.0002 |             1.0000 |             1.0000 |            0.0850 |
| CR5L  | s9_2      | Q05@C     |    101 |        128 |       -0.0059 |       -0.0059 |      -0.0000 |        0.0012 |        0.0011 |       0.0002 |             1.0000 |             1.0000 |            0.5250 |
| CR5L  | s9_2      | Q05@D     |    211 |        512 |       -0.0048 |       -0.0049 |       0.0001 |        0.0006 |        0.0005 |       0.0001 |             1.0000 |             1.0000 |            0.1225 |
| CR5L  | s9_2      | Q05@D     |    613 |        512 |       -0.0044 |       -0.0046 |       0.0003 |        0.0007 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0050 |
| CR5L  | s9_2      | Q01@C     |     31 |        512 |       -0.0044 |       -0.0046 |       0.0002 |        0.0007 |        0.0006 |       0.0001 |             1.0000 |             1.0000 |            0.0250 |


## W4-03 Gap 1 under O-true (analytic, global search)

| cid              | target         | category              |   n_lower |   n_confirmed |   best_dL |   best_dsp |   ideal_MCC |   ideal_leak_max | transfer_all   | practical_adequate   |
|:-----------------|:---------------|:----------------------|----------:|--------------:|----------:|-----------:|------------:|-----------------:|:---------------|:---------------------|
| MX-B01c03@AD     | C1:L:FULL      | empirical support     |         0 |             0 |    0.0000 |     0.0000 |      1.0000 |           0.0278 | True           | True                 |
| MX-B01c03@AD     | C2:L:FULL      | empirical support     |         0 |             0 |   -0.0000 |     0.0009 |      1.0000 |           0.0278 | True           | True                 |
| MX-B01c03@AD     | C3:L:FULL      | empirical support     |         0 |             0 |   -0.0000 |     0.0010 |      1.0000 |           0.0524 | True           | True                 |
| MX-B01c03@AD     | CR3:L:MASK-ALL | global counterexample |         6 |             6 |   -0.0106 |     0.0500 |      0.9975 |           0.0687 | True           | True                 |
| MX-B01c03@AD     | CR3:L:MASK-COL | global counterexample |        24 |            24 |   -0.0136 |     0.0594 |      0.9965 |           0.0948 | True           | True                 |
| MX-B01c03@AD     | CR4:L:FULL     | empirical support     |         0 |             0 |   -0.0078 |     0.0483 |      0.9977 |           0.0851 | True           | True                 |
| MX-B01c03@AD     | CR5:L:MASK-ALL | empirical support     |         0 |             0 |   -0.0049 |     0.0372 |      0.9986 |           0.0654 | True           | True                 |
| MX-B01c03@AD     | CR5:L:MASK-COL | empirical support     |         0 |             0 |   -0.0049 |     0.0370 |      0.9986 |           0.0650 | True           | True                 |
| MX-B08c03@AD     | C1:L:FULL      | empirical support     |         0 |             0 |    0.0000 |     0.0000 |      1.0000 |           0.0278 | True           | True                 |
| MX-B08c03@AD     | C2:L:FULL      | tied non-permutation  |         0 |             0 |   -0.0000 |     0.2190 |      0.9518 |           0.4593 | False          | False                |
| MX-B08c03@AD     | C3:L:FULL      | tied non-permutation  |         0 |             0 |   -0.0000 |     0.1870 |      0.9650 |           0.4387 | False          | False                |
| MX-B08c03@AD     | CR3:L:MASK-ALL | tied non-permutation  |         0 |             0 |   -0.0087 |     0.0649 |      0.9958 |           0.0830 | True           | True                 |
| MX-B08c03@AD     | CR3:L:MASK-COL | tied non-permutation  |         0 |             0 |   -0.0082 |     0.0768 |      0.9941 |           0.1324 | True           | True                 |
| MX-B08c03@AD     | CR4:L:FULL     | global counterexample |        24 |            24 |   -0.0108 |     0.0730 |      0.9947 |           0.1017 | True           | True                 |
| MX-B08c03@AD     | CR5:L:MASK-ALL | tied non-permutation  |         0 |             0 |   -0.0089 |     0.0549 |      0.9970 |           0.0856 | True           | True                 |
| MX-B08c03@AD     | CR5:L:MASK-COL | empirical support     |         0 |             0 |   -0.0081 |     0.0440 |      0.9981 |           0.0813 | True           | True                 |
| RW-GL-l0.75-a1.0 | C1:L:FULL      | empirical support     |         0 |             0 |    0.0000 |     0.0000 |      1.0000 |           0.0278 | True           | True                 |
| RW-GL-l0.75-a1.0 | C2:L:FULL      | empirical support     |         0 |             0 |   -0.0000 |     0.0009 |      1.0000 |           0.0278 | True           | True                 |
| RW-GL-l0.75-a1.0 | C3:L:FULL      | empirical support     |         0 |             0 |   -0.0000 |     0.0010 |      1.0000 |           0.0524 | True           | True                 |
| RW-GL-l0.75-a1.0 | CR3:L:MASK-ALL | global counterexample |        24 |            24 |   -0.0071 |     0.0509 |      0.9974 |           0.0704 | True           | True                 |
| RW-GL-l0.75-a1.0 | CR3:L:MASK-COL | global counterexample |        24 |            24 |   -0.0079 |     0.0651 |      0.9958 |           0.1065 | True           | True                 |
| RW-GL-l0.75-a1.0 | CR4:L:FULL     | global counterexample |        24 |            24 |   -0.0074 |     0.0542 |      0.9970 |           0.0842 | True           | True                 |
| RW-GL-l0.75-a1.0 | CR5:L:MASK-ALL | empirical support     |         0 |             0 |   -0.0056 |     0.0389 |      0.9985 |           0.0702 | True           | True                 |
| RW-GL-l0.75-a1.0 | CR5:L:MASK-COL | empirical support     |         0 |             0 |   -0.0055 |     0.0403 |      0.9984 |           0.0656 | True           | True                 |


## W4-03 Gap 2: libraries failing both-marginal (primary mask)

| cid              | inferred   | toy   | variant   |   n_seeds |
|:-----------------|:-----------|:------|:----------|----------:|
| MX-B01c03@AD     | Q05@C      | CR3   | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR3   | s9_2      |         3 |
| MX-B01c03@AD     | Q05@C      | CR3L  | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR3L  | s9_2      |         3 |
| MX-B01c03@AD     | Q05@C      | CR4   | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR4   | s9_2      |         3 |
| MX-B01c03@AD     | Q05@C      | CR4L  | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR4L  | s9_2      |         3 |
| MX-B01c03@AD     | Q05@C      | CR5   | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR5   | s9_2      |         3 |
| MX-B01c03@AD     | Q05@C      | CR5L  | s9        |         3 |
| MX-B01c03@AD     | Q05@C      | CR5L  | s9_2      |         3 |
| MX-B01c03@AD     | Q05@D      | CR3   | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR3   | s9_2      |         2 |
| MX-B01c03@AD     | Q05@D      | CR3L  | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR3L  | s9_2      |         2 |
| MX-B01c03@AD     | Q05@D      | CR4   | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR4   | s9_2      |         2 |
| MX-B01c03@AD     | Q05@D      | CR4L  | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR4L  | s9_2      |         2 |
| MX-B01c03@AD     | Q05@D      | CR5   | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR5   | s9_2      |         2 |
| MX-B01c03@AD     | Q05@D      | CR5L  | s9        |         2 |
| MX-B01c03@AD     | Q05@D      | CR5L  | s9_2      |         2 |
| MX-B08c03@AD     | Q05@C      | C1    | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C1    | s9_2      |         3 |
| MX-B08c03@AD     | Q05@C      | C1L   | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C1L   | s9_2      |         3 |
| MX-B08c03@AD     | Q05@C      | C2    | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C2    | s9_2      |         3 |
| MX-B08c03@AD     | Q05@C      | C2L   | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C2L   | s9_2      |         3 |
| MX-B08c03@AD     | Q05@C      | C3    | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C3    | s9_2      |         3 |
| MX-B08c03@AD     | Q05@C      | C3L   | s9        |         3 |
| MX-B08c03@AD     | Q05@C      | C3L   | s9_2      |         3 |
| MX-B08c03@AD     | Q05@D      | C1    | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C1    | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | C1L   | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C1L   | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | C2    | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C2    | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | C2L   | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C2L   | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | C3    | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C3    | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | C3L   | s9        |         2 |
| MX-B08c03@AD     | Q05@D      | C3L   | s9_2      |         2 |
| MX-B08c03@AD     | Q05@D      | CR4L  | s9        |         1 |
| RW-GL-l0.75-a1.0 | Q05@C      | CR4   | s9        |         3 |
| RW-GL-l0.75-a1.0 | Q05@D      | CR4   | s9        |         2 |
| RW-GL-l0.75-a1.0 | Q05@D      | CR4   | s9_2      |         1 |


## W4-03 Gap 2: both-marginal / perfect counts over the 24 primary-mask libraries (12 toys x 2 variants)

| cid              | inferred   |   seed | variant   |   libs |   O_perfect |   I_perfect |   O_marginal |   I_marginal |   both_marginal |   pearson_med |
|:-----------------|:-----------|-------:|:----------|-------:|------------:|------------:|-------------:|-------------:|----------------:|--------------:|
| MX-B01c03@AD     | Q05@C      |     31 | s9        |     12 |          12 |           6 |           12 |            6 |               6 |         0.875 |
| MX-B01c03@AD     | Q05@C      |     31 | s9_2      |     12 |          12 |           6 |           12 |            6 |               6 |         0.829 |
| MX-B01c03@AD     | Q05@C      |     73 | s9        |     12 |          12 |           6 |           12 |            6 |               6 |         0.873 |
| MX-B01c03@AD     | Q05@C      |     73 | s9_2      |     12 |          12 |           6 |           12 |            6 |               6 |         0.825 |
| MX-B01c03@AD     | Q05@C      |    101 | s9        |     12 |          12 |           6 |           12 |            6 |               6 |         0.862 |
| MX-B01c03@AD     | Q05@C      |    101 | s9_2      |     12 |          12 |           6 |           12 |            6 |               6 |         0.814 |
| MX-B01c03@AD     | Q05@D      |    211 | s9        |     12 |          12 |           6 |           12 |            6 |               6 |         0.871 |
| MX-B01c03@AD     | Q05@D      |    211 | s9_2      |     12 |          12 |           6 |           12 |            6 |               6 |         0.822 |
| MX-B01c03@AD     | Q05@D      |    613 | s9        |     12 |          12 |           6 |           12 |            6 |               6 |         0.873 |
| MX-B01c03@AD     | Q05@D      |    613 | s9_2      |     12 |          12 |           6 |           12 |            6 |               6 |         0.822 |
| MX-B08c03@AD     | Q05@C      |     31 | s9        |     12 |           4 |           7 |            6 |           12 |               6 |         0.969 |
| MX-B08c03@AD     | Q05@C      |     31 | s9_2      |     12 |           4 |           8 |            6 |           12 |               6 |         0.974 |
| MX-B08c03@AD     | Q05@C      |     73 | s9        |     12 |           4 |           6 |            6 |           12 |               6 |         0.974 |
| MX-B08c03@AD     | Q05@C      |     73 | s9_2      |     12 |           4 |           9 |            6 |           12 |               6 |         0.979 |
| MX-B08c03@AD     | Q05@C      |    101 | s9        |     12 |           4 |           7 |            6 |           12 |               6 |         0.967 |
| MX-B08c03@AD     | Q05@C      |    101 | s9_2      |     12 |           4 |           9 |            6 |           12 |               6 |         0.976 |
| MX-B08c03@AD     | Q05@D      |    211 | s9        |     12 |           4 |           7 |            6 |           12 |               6 |         0.973 |
| MX-B08c03@AD     | Q05@D      |    211 | s9_2      |     12 |           4 |           8 |            6 |           12 |               6 |         0.977 |
| MX-B08c03@AD     | Q05@D      |    613 | s9        |     12 |           4 |           6 |            6 |           11 |               5 |         0.971 |
| MX-B08c03@AD     | Q05@D      |    613 | s9_2      |     12 |           4 |           8 |            6 |           12 |               6 |         0.979 |
| RW-GL-l0.75-a1.0 | Q05@C      |     31 | s9        |     12 |          12 |           8 |           12 |           11 |              11 |         0.959 |
| RW-GL-l0.75-a1.0 | Q05@C      |     31 | s9_2      |     12 |          12 |           6 |           12 |           12 |              12 |         0.968 |
| RW-GL-l0.75-a1.0 | Q05@C      |     73 | s9        |     12 |          12 |           7 |           12 |           11 |              11 |         0.966 |
| RW-GL-l0.75-a1.0 | Q05@C      |     73 | s9_2      |     12 |          12 |           7 |           12 |           12 |              12 |         0.973 |
| RW-GL-l0.75-a1.0 | Q05@C      |    101 | s9        |     12 |          12 |           7 |           12 |           11 |              11 |         0.960 |
| RW-GL-l0.75-a1.0 | Q05@C      |    101 | s9_2      |     12 |          12 |           6 |           12 |           12 |              12 |         0.968 |
| RW-GL-l0.75-a1.0 | Q05@D      |    211 | s9        |     12 |          12 |           7 |           12 |           11 |              11 |         0.963 |
| RW-GL-l0.75-a1.0 | Q05@D      |    211 | s9_2      |     12 |          12 |           7 |           12 |           11 |              11 |         0.971 |
| RW-GL-l0.75-a1.0 | Q05@D      |    613 | s9        |     12 |          12 |           7 |           12 |           11 |              11 |         0.965 |
| RW-GL-l0.75-a1.0 | Q05@D      |    613 | s9_2      |     12 |          12 |           6 |           12 |           12 |              12 |         0.972 |
