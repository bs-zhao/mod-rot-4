### W1-00

468 checks, 0 failed.

| method | max_abs_rel_err | ux_moment_rel_err | frac_points_crossing_x_boundary |
|---|---|---|---|
| I02.FD0.003 | 0.0001 | 0.0001 | 0.7500 |
| I02.FD0.01 | 0.0011 | 0.0006 | 0.7500 |
| I02.FD0.03 | 0.0101 | 0.0052 | 0.7500 |
| I02.FD0.1(hist) | 0.0970 | 0.0548 | 0.8125 |
| I03.GFD0.03 | 0.0110 | 0.0026 | 0.7500 |

### W1-01 class B: metrics (of 415) passing on EVERY library of the scope (Q01, pilot, seed 31)

| channel_group | mask_policy | scope | libs | analytic perfect | inferred perfect | inferred marginal | both perfect | both marginal | both strict-perfect | both perfect (legacy tol.) |
|---|---|---|---|---|---|---|---|---|---|---|
| L/C | MASK-ALL | C-family | 6 | 174 | 158 | 175 | 140 | 157 | 140 | 140 |
| L/C | MASK-ALL | CR-family | 6 | 83 | 1 | 4 | 1 | 4 | 1 | 1 |
| L/C | MASK-ALL | all | 12 | 53 | 0 | 2 | 0 | 0 | 0 | 0 |
| P | MASK-ALL | C-family | 6 | 159 | 147 | 185 | 134 | 159 | 134 | 134 |
| P | MASK-ALL | CR-family | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | MASK-ALL | all | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| L/C | MASK-COL | C-family | 6 | 174 | 158 | 175 | 140 | 157 | 140 | 140 |
| L/C | MASK-COL | CR-family | 6 | 85 | 0 | 1 | 0 | 1 | 0 | 0 |
| L/C | MASK-COL | all | 12 | 58 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | MASK-COL | C-family | 6 | 159 | 147 | 185 | 134 | 159 | 134 | 134 |
| P | MASK-COL | CR-family | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P | MASK-COL | all | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### W1-01 class C: metrics (of 415) passing on EVERY library of the scope (Q01, pilot, seed 31)

| channel_group | mask_policy | scope | libs | analytic perfect | inferred perfect | inferred marginal | both perfect | both marginal | both strict-perfect | both perfect (legacy tol.) |
|---|---|---|---|---|---|---|---|---|---|---|
| L/C | MASK-ALL | C-family | 6 | 174 | 154 | 176 | 139 | 160 | 139 | 139 |
| L/C | MASK-ALL | CR-family | 6 | 83 | 7 | 11 | 7 | 11 | 7 | 7 |
| L/C | MASK-ALL | all | 12 | 53 | 0 | 4 | 0 | 0 | 0 | 0 |
| L/C | MASK-COL | C-family | 6 | 174 | 154 | 176 | 139 | 160 | 139 | 139 |
| L/C | MASK-COL | CR-family | 6 | 85 | 1 | 8 | 1 | 8 | 1 | 1 |
| L/C | MASK-COL | all | 12 | 58 | 0 | 4 | 0 | 0 | 0 | 0 |

### W1-01 four-way table (library x metric cells, marginal flags, Q01 pilot, FULL / MASK-ALL)

| cls | fam_group | channel | cells | O_pass_I_pass | O_pass_I_fail | O_fail_I_pass | O_fail_I_fail | O_fail_constant | O_fail_wrong_min | I_fail_constant | I_fail_wrong_min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | C | YL | 2490 | 1077 | 71 | 213 | 1129 | 624 | 196 | 276 | 924 |
| B | C | YP | 2490 | 1174 | 40 | 103 | 1173 | 296 | 980 | 294 | 919 |
| B | CR | YL | 2490 | 94 | 1040 | 9 | 1347 | 136 | 1220 | 166 | 2221 |
| B | CR | YP | 2490 | 64 | 184 | 11 | 2231 | 234 | 2008 | 216 | 2199 |
| C | C | YC | 2490 | 1085 | 63 | 208 | 1134 | 624 | 196 | 280 | 917 |
| C | CR | YC | 2490 | 141 | 993 | 6 | 1350 | 136 | 1220 | 160 | 2183 |

### W1-01 evaluation 1 (median pooled Pearson analytic vs decoder over metrics; matrix errors averaged over rotations; S9 and S9_2 pooled)

| toy | channel | pearson_all | pearson_priority | UX_rel_err | UU_rel_err | oracle_zero_false_energy |
|---|---|---|---|---|---|---|
| C1 | YL | 0.999 | 0.997 | 0.028 |  | 0.024 |
| C1 | YP | 0.999 | 0.999 | 0.042 | 0.237 | 0.000 |
| C2 | YL | 0.993 | 0.986 | 0.061 |  | 0.035 |
| C2 | YP | 0.994 | 0.991 | 0.034 | 0.211 | 0.000 |
| C3 | YL | 0.985 | 0.965 | 0.068 |  | 0.050 |
| C3 | YP | 0.997 | 0.998 | 0.029 | 0.152 | 0.000 |
| CR3 | YL | 0.623 | 0.502 | 0.082 | 0.801 | 0.007 |
| CR3 | YP | 0.766 | 0.780 | 0.107 | 0.540 | 0.000 |
| CR4 | YL | 0.356 | -0.065 | 0.101 | 0.747 | 0.011 |
| CR4 | YP | 0.762 | 0.721 | 0.111 | 0.515 | 0.000 |
| CR5 | YL | 0.552 | 0.171 | 0.070 | 0.860 | 0.007 |
| CR5 | YP | 0.822 | 0.821 | 0.112 | 0.555 | 0.000 |
| C1L | YC | 0.999 | 0.999 | 0.010 |  | 0.006 |
| C2L | YC | 0.998 | 0.995 | 0.025 |  | 0.004 |
| C3L | YC | 0.995 | 0.988 | 0.021 |  | 0.006 |
| CR3L | YC | 0.644 | 0.483 | 0.069 | 0.689 | 0.006 |
| CR4L | YC | 0.475 | 0.030 | 0.077 | 0.764 | 0.013 |
| CR5L | YC | 0.430 | 0.039 | 0.069 | 0.925 | 0.009 |

### W1-02 measure sensitivity: metrics both-marginal on every library of the scope (pilot, L/C, MASK-ALL)

| cls | scope | Q01 | Q02 | Q03 | Q04 | Q05 |
|---|---|---|---|---|---|---|
| B | C-family | 157 | 151 | 159 |  | 157 |
| B | CR-family | 4 | 5 | 4 | 9 | 5 |
| C | C-family | 160 | 161 | 159 |  | 161 |
| C | CR-family | 11 | 11 | 11 | 13 | 11 |

### W1-02 seed / fidelity stability of the priority panel (cells = library x measure x channel x mask x metric)

| fidelity | cls | cells | frac_same_status_3_seeds | frac_both_marginal_all_3 |
|---|---|---|---|---|
| C | B | 13680 | 0.971 | 0.242 |
| C | C | 6840 | 0.981 | 0.326 |
| D | B | 13392 | 0.985 | 0.254 |
| D | C | 6696 | 0.990 | 0.335 |
| C+D | B | 13392 | 0.968 |  |
| C+D | C | 6696 | 0.979 |  |

### W1-03 representation (TEST subjects; evaluation only)

| toy | variant | supervised_MCC | supervised_min | supervised_leak_max | unrotated_MCC | unrotated_leak_max | linear_R2_min | cov_condition | map_condition |
|---|---|---|---|---|---|---|---|---|---|
| C1 | s9 | 0.973 | 0.972 | 0.040 | 0.848 | 0.534 | 0.944 | 1.104 | 1.061 |
| C1 | s9_2 | 0.973 | 0.972 | 0.040 | 0.848 | 0.534 | 0.944 | 1.092 | 1.056 |
| C2 | s9 | 0.977 | 0.966 | 0.030 | 0.802 | 0.512 | 0.934 | 1.259 | 1.118 |
| C2 | s9_2 | 0.977 | 0.966 | 0.043 | 0.802 | 0.512 | 0.934 | 1.254 | 1.114 |
| C3 | s9 | 0.971 | 0.958 | 0.051 | 0.791 | 0.531 | 0.920 | 1.187 | 1.110 |
| C3 | s9_2 | 0.971 | 0.957 | 0.050 | 0.791 | 0.531 | 0.920 | 1.156 | 1.101 |
| CR3 | s9 | 0.871 | 0.835 | 0.124 | 0.613 | 0.630 | 0.696 | 1.519 | 1.093 |
| CR3 | s9_2 | 0.869 | 0.831 | 0.079 | 0.613 | 0.630 | 0.696 | 1.135 | 1.217 |
| CR4 | s9 | 0.850 | 0.807 | 0.153 | 0.633 | 0.613 | 0.652 | 1.625 | 1.051 |
| CR4 | s9_2 | 0.847 | 0.806 | 0.080 | 0.633 | 0.613 | 0.652 | 1.102 | 1.278 |
| CR5 | s9 | 0.885 | 0.849 | 0.080 | 0.668 | 0.624 | 0.719 | 1.361 | 1.087 |
| CR5 | s9_2 | 0.884 | 0.846 | 0.050 | 0.668 | 0.624 | 0.719 | 1.109 | 1.112 |
| C1L | s9 | 0.998 | 0.998 | 0.029 | 0.880 | 0.551 | 0.995 | 1.073 | 1.040 |
| C1L | s9_2 | 0.998 | 0.998 | 0.029 | 0.880 | 0.551 | 0.995 | 1.070 | 1.038 |
| C2L | s9 | 0.997 | 0.994 | 0.033 | 0.774 | 0.661 | 0.990 | 1.143 | 1.075 |
| C2L | s9_2 | 0.997 | 0.994 | 0.032 | 0.774 | 0.661 | 0.990 | 1.094 | 1.057 |
| C3L | s9 | 0.995 | 0.992 | 0.052 | 0.883 | 0.602 | 0.985 | 1.138 | 1.077 |
| C3L | s9_2 | 0.995 | 0.992 | 0.051 | 0.883 | 0.602 | 0.985 | 1.112 | 1.069 |
| CR3L | s9 | 0.964 | 0.946 | 0.086 | 0.696 | 0.610 | 0.900 | 1.482 | 1.183 |
| CR3L | s9_2 | 0.965 | 0.949 | 0.051 | 0.696 | 0.610 | 0.900 | 1.172 | 1.093 |
| CR4L | s9 | 0.917 | 0.886 | 0.145 | 0.609 | 0.690 | 0.790 | 1.821 | 1.214 |
| CR4L | s9_2 | 0.919 | 0.889 | 0.061 | 0.609 | 0.690 | 0.790 | 1.112 | 1.134 |
| CR5L | s9 | 0.955 | 0.927 | 0.099 | 0.700 | 0.615 | 0.865 | 1.632 | 1.249 |
| CR5L | s9_2 | 0.958 | 0.930 | 0.019 | 0.700 | 0.615 | 0.865 | 1.075 | 1.065 |

### W1-03 bridge: fraction of the analytic passes (priority panel, marginal) kept by each surface, same points

| toy | variant | channel | analytic_passes | frac_kept_BO | frac_kept_BA | frac_kept_D | frac_kept_S | coverage | pattern |
|---|---|---|---|---|---|---|---|---|---|
| C1 | s9 | YL | 70 | 1.00 | 1.00 | 0.99 | 0.59 | 1.00 | decoder keeps most analytic passes |
| C1 | s9 | YP | 73 | 1.00 | 0.95 | 0.97 | 0.95 | 1.00 | decoder keeps most analytic passes |
| C1 | s9_2 | YL | 70 | 1.00 | 1.00 | 0.99 | 0.57 | 1.00 | decoder keeps most analytic passes |
| C1 | s9_2 | YP | 73 | 0.97 | 0.99 | 0.97 | 0.97 | 1.00 | decoder keeps most analytic passes |
| C1L | s9 | YC | 70 | 1.00 | 1.00 | 0.97 | 0.93 | 1.00 | decoder keeps most analytic passes |
| C1L | s9_2 | YC | 70 | 1.00 | 1.00 | 0.97 | 0.93 | 1.00 | decoder keeps most analytic passes |
| C2 | s9 | YL | 68 | 1.00 | 1.00 | 0.93 | 0.63 | 1.00 | decoder keeps most analytic passes |
| C2 | s9 | YP | 52 | 0.98 | 0.98 | 1.00 | 0.98 | 1.00 | decoder keeps most analytic passes |
| C2 | s9_2 | YL | 68 | 1.00 | 0.97 | 0.93 | 0.66 | 1.00 | decoder keeps most analytic passes |
| C2 | s9_2 | YP | 52 | 0.98 | 0.98 | 1.00 | 0.98 | 1.00 | decoder keeps most analytic passes |
| C2L | s9 | YC | 68 | 1.00 | 0.91 | 0.99 | 0.88 | 1.00 | decoder keeps most analytic passes |
| C2L | s9_2 | YC | 68 | 1.00 | 0.91 | 0.97 | 0.91 | 1.00 | decoder keeps most analytic passes |
| C3 | s9 | YL | 65 | 1.00 | 1.00 | 1.00 | 0.49 | 1.00 | decoder keeps most analytic passes |
| C3 | s9 | YP | 48 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | decoder keeps most analytic passes |
| C3 | s9_2 | YL | 65 | 1.00 | 1.00 | 1.00 | 0.49 | 1.00 | decoder keeps most analytic passes |
| C3 | s9_2 | YP | 48 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | decoder keeps most analytic passes |
| C3L | s9 | YC | 65 | 1.00 | 1.00 | 0.97 | 0.92 | 1.00 | decoder keeps most analytic passes |
| C3L | s9_2 | YC | 65 | 1.00 | 1.00 | 0.97 | 0.92 | 1.00 | decoder keeps most analytic passes |
| CR3 | s9 | YL | 73 | 0.99 | 0.95 | 0.15 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR3 | s9 | YP | 0 |  |  |  |  | 1.00 | no analytic passes on this library (nothing to keep) |
| CR3 | s9_2 | YL | 73 | 1.00 | 0.82 | 0.14 | 0.00 | 0.99 | O good, bridges good, D bad -> decoder / conditional surface |
| CR3 | s9_2 | YP | 0 |  |  |  |  | 0.99 | no analytic passes on this library (nothing to keep) |
| CR3L | s9 | YC | 73 | 0.99 | 0.88 | 0.22 | 0.00 | 0.99 | O good, bridges good, D bad -> decoder / conditional surface |
| CR3L | s9_2 | YC | 73 | 0.99 | 0.93 | 0.21 | 0.00 | 0.99 | O good, bridges good, D bad -> decoder / conditional surface |
| CR4 | s9 | YL | 83 | 0.99 | 0.94 | 0.06 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR4 | s9 | YP | 3 | 0.67 | 1.00 | 0.33 | 0.00 | 1.00 | O good, B-orth bad -> alignment / geometry |
| CR4 | s9_2 | YL | 83 | 1.00 | 0.29 | 0.07 | 0.02 | 1.00 | O good, B-aff bad -> affine map / residual / support |
| CR4 | s9_2 | YP | 3 | 1.00 | 1.00 | 0.67 | 0.00 | 1.00 | decoder keeps most analytic passes |
| CR4L | s9 | YC | 83 | 0.99 | 0.82 | 0.14 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR4L | s9_2 | YC | 83 | 0.99 | 0.60 | 0.17 | 0.00 | 0.99 | O good, B-aff bad -> affine map / residual / support |
| CR5 | s9 | YL | 82 | 0.99 | 0.96 | 0.16 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR5 | s9 | YP | 3 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR5 | s9_2 | YL | 82 | 1.00 | 0.99 | 0.10 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR5 | s9_2 | YP | 3 | 1.00 | 1.00 | 0.00 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR5L | s9 | YC | 82 | 0.99 | 0.83 | 0.15 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |
| CR5L | s9_2 | YC | 82 | 1.00 | 1.00 | 0.15 | 0.00 | 1.00 | O good, bridges good, D bad -> decoder / conditional surface |

### W1-03 interpolation (1-a) bridge + a decoder: fraction of analytic passes kept

| bridge | cls | family_group | 0.0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|---|---|
| BA | B | C | 0.99 | 0.98 | 0.98 | 0.98 | 0.98 |
| BA | B | CR | 0.83 | 0.53 | 0.35 | 0.25 | 0.11 |
| BA | C | C | 0.97 | 0.97 | 0.98 | 0.98 | 0.97 |
| BA | C | CR | 0.84 | 0.64 | 0.37 | 0.27 | 0.17 |
| BO | B | C | 0.99 | 0.98 | 0.98 | 0.98 | 0.98 |
| BO | B | CR | 0.99 | 0.57 | 0.38 | 0.26 | 0.11 |
| BO | C | C | 1.00 | 0.98 | 0.98 | 0.98 | 0.97 |
| BO | C | CR | 0.99 | 0.49 | 0.37 | 0.26 | 0.17 |

### W1-03 T-free surrogate S (selection on VALIDATION loss)

| toy | selected_degree | selected_lambda | half_refit_UX_rel_diff | S_vs_D_eta_corr_Q01 | D_vs_O_eta_corr_Q01(eval-only) | S_vs_O_eta_corr_Q01(eval-only) |
|---|---|---|---|---|---|---|
| C1 | 2 | 0.000 | 0.044 | 0.998 | 0.971 | 0.970 |
| C2 | 3 | 0.000 | 0.068 | 0.993 | 0.980 | 0.976 |
| C3 | 3 | 0.000 | 0.104 | 0.989 | 0.975 | 0.970 |
| CR3 | 3 | 0.000 | 0.137 | 0.987 | 0.985 | 0.973 |
| CR4 | 3 | 0.000 | 0.115 | 0.996 | 0.992 | 0.991 |
| CR5 | 3 | 0.000 | 0.113 | 0.992 | 0.989 | 0.982 |
| C1L | 3 | 0.010 | 0.006 | 0.999 | 0.997 | 0.998 |
| C2L | 3 | 0.000 | 0.056 | 0.999 | 0.997 | 0.997 |
| C3L | 3 | 0.000 | 0.021 | 0.998 | 0.996 | 0.996 |
| CR3L | 3 | 0.000 | 0.036 | 0.993 | 0.998 | 0.993 |
| CR4L | 3 | 0.000 | 0.030 | 0.999 | 0.999 | 0.998 |
| CR5L | 3 | 0.000 | 0.046 | 0.995 | 0.998 | 0.996 |

### W1-04 global analytic categories per target (priority-panel metrics; distance threshold 1e-3)

| family | channel | mask | empirical global support | global counterexample | inconclusive | inconclusive search | nonunique minimum |
|---|---|---|---|---|---|---|---|
| C1 | L | FULL | 83 | 0 | 7 | 1 | 2 |
| C1 | P | FULL | 6 | 15 | 1 | 0 | 71 |
| C2 | L | FULL | 73 | 0 | 9 | 2 | 9 |
| C2 | P | FULL | 0 | 23 | 1 | 0 | 69 |
| C3 | L | FULL | 63 | 11 | 7 | 2 | 10 |
| C3 | P | FULL | 1 | 30 | 1 | 0 | 61 |
| CR3 | L | MASK-ALL | 1 | 72 | 3 | 6 | 11 |
| CR3 | L | MASK-COL | 8 | 73 | 1 | 1 | 10 |
| CR3 | P | MASK-ALL | 0 | 92 | 1 | 0 | 0 |
| CR3 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |
| CR4 | L | FULL | 20 | 53 | 2 | 0 | 18 |
| CR4 | P | FULL | 0 | 91 | 1 | 0 | 1 |
| CR5 | L | MASK-ALL | 22 | 45 | 1 | 0 | 25 |
| CR5 | L | MASK-COL | 23 | 50 | 3 | 0 | 17 |
| CR5 | P | MASK-ALL | 0 | 91 | 1 | 0 | 1 |
| CR5 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |

### W1-04 global analytic categories per target (priority-panel metrics; distance threshold 0.05)

| family | channel | mask | empirical global support | global counterexample | inconclusive | inconclusive search | nonunique minimum |
|---|---|---|---|---|---|---|---|
| C1 | L | FULL | 83 | 0 | 5 | 1 | 2 |
| C1 | P | FULL | 74 | 15 | 1 | 0 | 3 |
| C2 | L | FULL | 77 | 0 | 7 | 2 | 5 |
| C2 | P | FULL | 50 | 23 | 1 | 0 | 19 |
| C3 | L | FULL | 64 | 11 | 5 | 2 | 9 |
| C3 | P | FULL | 44 | 30 | 1 | 0 | 18 |
| CR3 | L | MASK-ALL | 7 | 42 | 2 | 38 | 4 |
| CR3 | L | MASK-COL | 43 | 46 | 1 | 2 | 1 |
| CR3 | P | MASK-ALL | 0 | 92 | 1 | 0 | 0 |
| CR3 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |
| CR4 | L | FULL | 64 | 24 | 2 | 2 | 1 |
| CR4 | P | FULL | 0 | 91 | 1 | 0 | 1 |
| CR5 | L | MASK-ALL | 71 | 17 | 1 | 3 | 1 |
| CR5 | L | MASK-COL | 75 | 13 | 1 | 4 | 0 |
| CR5 | P | MASK-ALL | 0 | 91 | 1 | 0 | 1 |
| CR5 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |

### W1-04 global analytic categories per target (priority-panel metrics; distance threshold 0.15)

| family | channel | mask | empirical global support | global counterexample | inconclusive | inconclusive search | nonunique minimum |
|---|---|---|---|---|---|---|---|
| C1 | L | FULL | 83 | 0 | 5 | 1 | 2 |
| C1 | P | FULL | 74 | 15 | 1 | 0 | 3 |
| C2 | L | FULL | 77 | 0 | 7 | 2 | 5 |
| C2 | P | FULL | 53 | 23 | 1 | 0 | 16 |
| C3 | L | FULL | 66 | 9 | 5 | 2 | 9 |
| C3 | P | FULL | 44 | 30 | 1 | 0 | 18 |
| CR3 | L | MASK-ALL | 18 | 20 | 2 | 50 | 3 |
| CR3 | L | MASK-COL | 63 | 16 | 1 | 13 | 0 |
| CR3 | P | MASK-ALL | 0 | 92 | 1 | 0 | 0 |
| CR3 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |
| CR4 | L | FULL | 72 | 12 | 2 | 7 | 0 |
| CR4 | P | FULL | 0 | 91 | 1 | 0 | 1 |
| CR5 | L | MASK-ALL | 76 | 12 | 1 | 4 | 0 |
| CR5 | L | MASK-COL | 77 | 11 | 1 | 4 | 0 |
| CR5 | P | MASK-ALL | 0 | 91 | 1 | 0 | 1 |
| CR5 | P | MASK-COL | 0 | 92 | 1 | 0 | 0 |

Metrics with empirical global support (d_SP > 0.05) on every logit/continuous target: `L03|UX.lq.1`

### W1-05 lanes

| lane | entity | n |
|---|---|---|
| A1 | metric | 33 |
| A2 | metric | 211 |
| B | metric | 38 |
| C | metric | 24 |
| D | library | 6 |
| E | library | 5 |
| E | metric | 42 |
