from .detr import build_detr


def build_model(args):
    return build_detr(args)