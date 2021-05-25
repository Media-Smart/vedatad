from vedacore.misc import build_from_cfg, registry


def build_enhance_module(cfg, default_args=None):
    """Build enhance module.

    Args:
        cfg (dict): The enhance module config, which should contain:
            - type (str): Layer type.
            - module args: Args needed to instantiate an enhance module.
        default_args (dict | None): The default config for enhance module.
            Default: None

    Returns:
        nn.Module: Created enhance module.
    """
    return build_from_cfg(cfg, registry, 'enhance_module', default_args)
