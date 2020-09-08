import jsonargparse
from jsonargparse import ArgumentParser, ActionConfigFile
from .utils import rgetattr, rsetattr, str_to_bool
from .args import arg_groups


class Hyperparameters:
    def __init__(self, arg_groups=arg_groups, check_required=True):
        self.arg_groups = arg_groups
        self._parser = make_grouped_parser(arg_groups, check_required)
        self._args = self._parser.parse_args()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._args, attr)

    def update_from_file(self, config_file, override=False):
        try:
            new_args = self._parser.parse_path(config_file, defaults=True)
        except jsonargparse.ParserError:
            print(f"ParserError: Incorrect config file: {config_file}")
            return

        if override:
            merged = self._parser.merge_config(new_args, self._args)
        else:
            merged = self._parser.merge_config(self._args, new_args)
        self._args = merged
            
    def print_values(self):
        """
        TODO Deprecated, use __str__ method instead.
        """
        print(self)

    def save(self, filename):
        self._parser.save(self._args, path=str(filename), skip_none=False, overwrite=True)

    def __str__(self):
        """
        TODO Print grouped by arg group.
        Not natively supported by argparse, see:
        https://stackoverflow.com/questions/38884513/python-argparse-how-can-i-get-namespace-objects-for-argument-groups-separately
        """
        return self._parser.dump(self._args)


def make_grouped_parser(arg_groups, check_required=True):
    """
    Builds a grouped argument parser.

    :param arg_groups: a dict of dicts {group_name: args}. See ./args for format.
    :param check_required: Check for required arguments, defaults to True
    :return: argument parser containing the args from arg_groups.
    """
    parser = ArgumentParser()
    parser.add_argument('--hparams_file', action=ActionConfigFile)
    for group_name, arg_group in arg_groups.items():
        add_arg_group(parser, group_name, arg_group, check_required=check_required)
    return parser


def add_arg_group(parser, group_name, arg_group, check_required=True):
    group = parser.add_argument_group(group_name)
    for arg_name, arg_val in arg_group.items():
        if len(arg_val) == 6:
            arg_val = arg_val[1:] #TODO fix args.py and remove
        arg_type, default, required, description, _ = arg_val
        arg_type = str_to_bool if arg_type == bool else arg_type
        required = required if check_required else False
        group.add_argument(f"--{arg_name}", type=arg_type, help=description, default=default, required=required)
