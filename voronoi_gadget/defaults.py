import yaml
from pathlib import Path


def get_style_config(style='default'):
    """
    Read style_config.yaml
    """
    config_path = Path(__file__).parent / "config/style_config.yaml"
    with open(config_path, "r") as ymlfile:
        content = yaml.load(ymlfile, Loader=yaml.FullLoader)
        cfg = dict()
        default_cfg = content["default"]
        new_cfg = content[style]
        for key in default_cfg.keys():
            if key in new_cfg:
                cfg[key] = new_cfg[key]
            else:
                cfg[key] = default_cfg[key]
    return cfg


def get_plot_config(qty, statsmode=None, select=False, npanels=4):
    """
    For a given quantity, returns default subplot titles, colormap, FF
    colormap limit type (symmetric or not).
    """
    config_path = Path(__file__).parent / "config/plot_config.yaml"
    with open(config_path, "r") as ymlfile:
        content = yaml.load(ymlfile, Loader=yaml.FullLoader)
        cfg = dict()
        default_cfg = content["default"]
        qty_cfg = content[qty]
        for key in default_cfg.keys():
            if key in qty_cfg:
                cfg[key] = qty_cfg[key]
            else:
                cfg[key] = default_cfg[key]
        cfg["titles"] = [r"$%s$" % title for title in cfg["titles"]]
    return cfg["titles"], cfg["cmap"], cfg["cmaplimits"], cfg["statsmode"], cfg["addlambdar"], cfg["centeriszero"]
