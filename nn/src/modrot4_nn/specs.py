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


# "<MODEL>L": same true parameters as <MODEL>, continuous noisy-logit response instead of a choice.
LOGIT_MODELS = tuple(m + "L" for m in NAMES)


def base_model(model_id):
    return model_id[:-1] if model_id in LOGIT_MODELS else model_id


def is_logit_model(model_id):
    return model_id in LOGIT_MODELS


def get_model(model_id):
    return _ModelNames(base_model(model_id))
