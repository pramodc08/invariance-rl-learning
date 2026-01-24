import os
import toml
import argparse
import sys
import warnings
import torch
import numpy as np
from pathlib import Path

current_file_dir = Path(__file__).resolve().parent

warnings.filterwarnings("ignore")

def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool)):
        # If the object is a string, number, or boolean, it's already JSON serializable
        return obj
    elif isinstance(obj, np.ndarray):
        # If the object is a NumPy array, convert it to a list
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        # If the object is a list or tuple, process each element in the collection
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        # If the object is a dictionary, process each key-value pair
        return {key: make_json_serializable(value) for key, value in obj.items()}
    else:
        # For any other type of object, return a string representation
        return str(obj)
    

def type_mapping(value_type, value):
    """
    Return type to be used for typecasting as per the string value and typeccasted value
    :param value_type: string
    :param value: any
    :return: tuple of data_type, type-casted value
    """
    type_ = {
        "int": int,
        "bool": bool,
        "float": float,
        "str": str,
        "list": list,
    }.get(value_type, None)

    value_ = type_(value)

    return type_, value_


def argparse_generator(config_file_path=None, config_parse_path=None):
    """
    Processes the args from config folder to generate argparse
    If config_file_path is provided, then values from config will override the default value
    :param config_file_path: string [Relative path for the config to be replaced with default]
    :return: dictionary of the argument to be used for the project
    """
    if config_parse_path is None:
        argparse_file = os.path.join(current_file_dir, "config/run.argument.toml")
    else:
        argparse_file = config_parse_path

    new_config = {}
    if config_file_path is not None:
        config_file = os.path.join(current_file_dir, config_file_path)
        new_config = toml.load(config_file)

    argparse_config = toml.load(argparse_file)
    parser = argparse.ArgumentParser(description='Invariance reinforcement learning in Lunarlander env',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key in argparse_config.keys():
        group = parser.add_argument_group(key)
        argparse_dict = argparse_config.get(key)
        for attr, value in argparse_dict.items():
            type_, value_ = type_mapping(value.get("type", None), value.get("default", None))
            group.add_argument(value.get("key"), help=value.get("help", ""), default=value_,
                                type=type_, nargs='?', metavar="")


        new_config_dict = new_config.get(key, {})
        try:
            group.set_defaults(**new_config_dict)
        except TypeError:
            warnings.warn("TypeError found in argparse_generator with config to replace default")

    args = vars(parser.parse_args())
    new_args = {}
    for key in argparse_config.keys():
        new_inner_dict = {}
        argparse_dict = argparse_config.get(key)
        for inner_key in argparse_dict.keys():
            new_inner_dict[inner_key] = args.get(inner_key)
        new_args[key] = new_inner_dict

    return new_args

def set_seed(seed, num_threads=4):
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    torch.set_num_threads(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)