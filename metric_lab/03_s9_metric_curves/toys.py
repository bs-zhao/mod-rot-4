"""The 12 test beds (docs/00_project_flow/02_test_beds.md) and the one NN run per toy used by this experiment.

Class B (binary choice): the S3 run at beta = 5e-4, repeat 1 (the reference run of every choice model).
Class C (continuous noisy logit, <MODEL>L): the S3 run at the T-free selected beta (largest beta whose validation R^2
is within 0.005 of the small-beta reference and with no collapsed latent dimension).
Each toy is evaluated on its S9 sweep (raw Z) and its S9_2 sweep (globally z-scored Z): 24 databases.
Other saved runs (repeat 2, CR3 at beta = 5e-5) are extra databases; run_all.sh also processes them.
"""
TOYS = {
    # model: (class, run folder relative to outputs/nn)
    "C1":   ("B", "C1/beta0p0005_lambda0_ctx0p25to0p8_L4_a73b15c147/repeat01"),
    "C2":   ("B", "C2/beta0p0005_lambda0_ctx0p25to0p8_L4_a73b15c147/repeat01"),
    "C3":   ("B", "C3/beta0p0005_lambda0_ctx0p25to0p8_L6_99df1d7a92/repeat01"),
    "CR3":  ("B", "CR3/beta0p0005_lambda0_ctx0p25to0p8_L4_a73b15c147/repeat01"),
    "CR4":  ("B", "CR4/beta0p0005_lambda0_ctx0p25to0p8_L3_09fe72d2fe/repeat01"),
    "CR5":  ("B", "CR5/beta0p0005_lambda0_ctx0p25to0p8_L3_09fe72d2fe/repeat01"),
    "C1L":  ("C", "C1L/beta0p018_lambda0_ctx0p25to0p8_L4_694ebc0d12/repeat01"),
    "C2L":  ("C", "C2L/beta0p018_lambda0_ctx0p25to0p8_L4_694ebc0d12/repeat01"),
    "C3L":  ("C", "C3L/beta0p006_lambda0_ctx0p25to0p8_L6_4ade5ca369/repeat01"),
    "CR3L": ("C", "CR3L/beta0p003_lambda0_ctx0p25to0p8_L4_f7b516956e/repeat01"),
    "CR4L": ("C", "CR4L/beta0p001_lambda0_ctx0p25to0p8_L3_d0bd42a54e/repeat01"),
    "CR5L": ("C", "CR5L/beta0p006_lambda0_ctx0p25to0p8_L3_db03bec16e/repeat01"),
}


def toy_of(model, condition, repeat):
    """The toy name if (model, condition, repeat) is that toy's designated run, else None."""
    entry = TOYS.get(model)
    return model if entry and entry[1] == f"{model}/{condition}/{repeat}" else None
