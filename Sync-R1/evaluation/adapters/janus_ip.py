from .janus_base import JanusAdapterBase


class JanusIPAdapter(JanusAdapterBase):
    model_family = "janus"
    prompt_mode = "ip"
