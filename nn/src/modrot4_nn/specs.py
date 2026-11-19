"""Model labels for diagnostic plots; NN training does not access true T."""
NAMES = {
    "C1": ("b1","b2","b3","b4"),
    "C2": ("b1","b2","b3","b4"),
    "C3": ("b1","b2","b3","b4","b5","b6"),
    "CR3": ("alpha","lambda_loss","gamma","tau"),
    "CR4": ("lambda_loss","gamma","tau"),
    "CR5": ("lambda_loss","gamma","tau"),
}


class _ModelNames:
    def __init__(self, model_id):
        self.parameter_names = NAMES[model_id]


def get_model(model_id):
    return _ModelNames(model_id)
