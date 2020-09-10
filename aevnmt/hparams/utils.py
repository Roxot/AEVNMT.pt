import json
from .hparams import Hyperparameters


def convert_config(fn, translation_dict):
    """
    Convert old non-nested json config to nested YAML config.

    This only works with json configs that are compatible with
    the previous parser (all arguments have to be in hparam_translation_dict.yaml)

    :param fn: [description]
    :type fn: function
    :param translation_dict: [description]
    :type translation_dict: [type]
    """
    correct_config = True
    with open(fn, 'r') as f:
        cfg = json.load(f)
        
    new_cfg = dict()
    for k, v in cfg.items():
        if k in translation_dict:
            new_cfg[translation_dict[k]] = v
        else:
            correct_config = False
    if not correct_config:
        print(f'{fn} contains an incorrect config, and is not converted.')
        return

    parser = Hyperparameters(check_required=False)._parser
    new_cfg = parser.parse_object(new_cfg)
    
    new_fn = Path(fn).with_suffix('.yaml')
    parser.save(new_cfg, str(new_fn), format='yaml', overwrite=True)