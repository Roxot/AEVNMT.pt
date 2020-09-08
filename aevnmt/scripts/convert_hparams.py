import re
import yaml, json
from aevnmt.hparams.args import all_args
from aevnmt.hparams import Hyperparameters
import glob


def search_prefixes():
    """
    search if any '{prefix}.{arg}' exists in the code, where prefix != hparams

    result: does not exist, only hparams.* needs to be replaced.
    """

    r_searcher = r"(\w*)\.{}(?![_\w\d])"
    all_hparams = all_args.keys()

    files = glob.glob('../**/*.py', recursive=True)
    print(list(files))

    all_prefixes = set()
    for fn in files:
        if "/hparams/" in fn:
            continue
        prefixes = set()
        with open(fn, 'r') as f:
            txt = f.read()
            for param in all_hparams:
                result = re.findall(r_searcher.format(param), txt)
                if result:
                    prefixes |= set(result)
        if 'self' in prefixes:
            prefixes.remove('self')
        if len(prefixes):
            print(fn)
            print(prefixes)
            all_prefixes |= prefixes
    print(all_prefixes)


def replace_in_file(fn, translation_dict):
    regex = r"hparams\.{}(?![_\w\d])"
    total = 0
    with open(fn, 'r') as f:
        content = f.read()
        for k, v in translation_dict.items():
            # Skip hparams without changes
            if k != v:
                # search all hparam occurences, change, add count to total
                content, num = re.subn(regex.format(k), f"hparams.{v}", content)
                total += num
    with open(fn, 'w') as f:
        f.write(content)
    if total > 0:
        print(fn, "--", total, "changes")

def get_replace_files():
    filenames = glob.glob('../**/*.py', recursive=True)
    # Exclude hparam and script folder.
    filenames = [f for f in filenames if '/hparams/' not in f]
    filenames = [f for f in filenames if '/scripts/' not in f]
    return filenames

def replace_all():
    with open('../hparams/hparam_translation_dict.yaml', 'r') as f:
        translation_dict = yaml.load(f)

    filenames = get_replace_files()
    for fn in filenames:
        replace_in_file(fn, translation_dict)

def replace_single(fn):
    with open('../hparams/hparam_translation_dict.yaml', 'r') as f:
        translation_dict = yaml.load(f)
    print(translation_dict['model_checkpoint'])
    replace_in_file(fn, translation_dict)

def convert_config(fn, translation_dict):
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
        print(f'{fn} has outdated config')
        return

    parser = Hyperparameters(check_required=False)._parser
    new_cfg = parser.parse_object(new_cfg)
    
    new_fn = fn.replace('.json', '.yaml')
    parser.save(new_cfg, new_fn, format='yaml', overwrite=True)

def convert_demo_configs():
    with open('../hparams/hparam_translation_dict.yaml', 'r') as f:
        translation_dict = yaml.load(f)

    filenames = glob.glob('../../demo/hparams/*.json')
    for fn in filenames:
        convert_config(fn, translation_dict) 

if __name__ == "__main__":
    # replace_all()
    convert_demo_configs()

    """
    ../aevnmt_helper.py -- 91 changes
    ../translate.py -- 17 changes
    ../train_utils.py -- 43 changes
    ../train.py -- 10 changes
    ../create_vocab.py -- 2 changes
    ../opt_utils.py -- 18 changes
    ../nmt_helper.py -- 6 changes
    """
    