from .external_baseline import ExternalBaselineAdapter


class JanusIPAdapter(ExternalBaselineAdapter):
    model_family = "janus"
    prompt_mode = "ip"
