from .bagel_base import BagelAdapterBase


class BagelTPAdapter(BagelAdapterBase):
    model_family = "bagel"
    prompt_mode = "tp"
