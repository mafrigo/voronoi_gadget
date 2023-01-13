def get_style_config(style='default'):
    """
    Read style_config.yaml
    """

    import yaml
    from pathlib import Path
    config_path = Path(__file__).parent / "../config/style_config.yaml"
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


def getdefaultplotparams(qty, statsmode=None, select=False, npanels=4):
    """
    For a given quantity, returns default subplot titles, colormap, FF
    colormap limit type (symmetric or not).
    """

    if qty == 'vel':
        titles = [r'$V_{avg} (\rm km/s)$', r'$\sigma (\rm km/s)$', r'$h_3$',
                  r'$h_4$']
        cmap = 'sauron'
        cmaplimits = ['symmetric', 'minmax', 'symmetric', 'symmetric']
    elif qty == 'age':
        if not select:
            titles = [r'$\rm Stellar \, age \, (\rm Gyr)$',
                      r'$\sigma_{\rm age} \, (\rm Gyr)$', r'$h_3$', r'$h_4$']
        else:
            titles = [r'$\rm age \, (\rm Gyr)$',
                      r'$\sigma_{\rm age} \, (\rm Gyr)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'logage':
        titles = [r'$\rm log_{10} \, \rm age \, (\rm Gyr)$',
                  r'$\sigma_{\rmlog \, age} (\rm Gyr)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'ZH':
        titles = [r'$\rm log(Z/Z_\odot)$', r'$\sigma_{Z}$', r'$h_3$', r'$h_4$']
        # titles=[r'$[Z/H]$', r'$\sigma_{[Z/H]}$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'alphafe':
        titles = [r'$[\alpha/{\rm Fe}]$', r'$\sigma_{[\alpha/{\rm Fe}]}$',
                  r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'OFe':
        titles = ['      ' + r'$[{\rm O}/{\rm Fe}]$',
                  r'$\sigma_{[{\rm O}/{\rm Fe}]}$',
                  r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'afmhot'
    elif qty == 'r':
        titles = [r'$r (\rm kpc)$', r'$\sigma_{\rm r} \, (\rm kpc)$', r'$h_3$',
                  r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'hot'
    elif qty == 'temp':
        titles = [r'$T\, (K)$', r'$\sigma_{\rm T} \, (K)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'
    elif qty == 'lambdar':
        titles = [r'$\lambda_R$', r'$\sigma (km/s)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'Spectral_r'
    elif qty == 'h3par':
        titles = [r'$K$', r'$\sigma (km/s)$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'bwr'
    else:
        titles = [r'$' + qty + '$', r'$\sigma_{' + qty + '}$', r'$h_3$', r'$h_4$']
        cmaplimits = ['minmax', 'minmax', 'symmetric', 'symmetric']
        cmap = 'viridis'

    if statsmode == '2gauss':
        if qty in ['vel']:
            cmaplimits = ['symmetric', 'minmax', 'symmetric', 'minmax', [0., 1.]]
        else:
            cmaplimits = ['minmax', 'minmax', 'minmax', 'minmax', [0., 1.]]
        titles = [r'$\mu_1$', r'$\sigma_1$', r'$\mu_2$', r'$\sigma_2$',
                  'fraction']

    if npanels > len(cmaplimits):
        cmaplimits = cmaplimits + ["minmax"] * (npanels - len(cmaplimits))

    return titles, cmap, cmaplimits
