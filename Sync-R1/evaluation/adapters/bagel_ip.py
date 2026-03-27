from .bagel_base import BagelAdapterBase


class BagelIPAdapter(BagelAdapterBase):
    model_family = "bagel"
    prompt_mode = "ip"
