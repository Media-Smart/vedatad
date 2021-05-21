from vedacore.misc import build_from_cfg, registry, singleton_arg


def build_assigner(cfg, **default_args):
    """Builder of segment assigner."""
    return build_from_cfg(cfg, registry, 'segment_assigner', default_args)


def build_sampler(cfg, **default_args):
    """Builder of segment sampler."""
    return build_from_cfg(cfg, registry, 'segment_sampler', default_args)


@singleton_arg
def build_segment_coder(cfg, **default_args):
    segment_coder = build_from_cfg(cfg, registry, 'segment_coder',
                                   default_args)
    return segment_coder
