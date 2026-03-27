from .external_baseline import ExternalBaselineAdapter


class JanusTPAdapter(ExternalBaselineAdapter):
    model_family = "janus"
    prompt_mode = "tp"
