from .janus_base import JanusAdapterBase


class JanusTPAdapter(JanusAdapterBase):
    model_family = "janus"
    prompt_mode = "tp"
