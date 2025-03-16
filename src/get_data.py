# used to configure the build process for Python extensions, checking system settings and dependencies.
from distutils.command.config import config
import os 
import requests
import shutil
# shutil is a Python module that provides a higher-level interface for file 
# operations, such as copying, moving, and deleting files and directories. It is part of Python's standard '
# 'library and is commonly used for tasks that involve manipulating files and directories.
import random
import argparse
import pandas as pd
import numpy as np
import yaml

def get_data(config_path):
    config = read_params(config_path)
    if not isinstance(config, dict):
        raise ValueError("Config file is not properly loaded as a dictionary.")
    # print(f"DEBUG: Type of config = {type(config)}")  # Should be <class 'dict'>
    # print(f"DEBUG: Config content = {config}")  # See the content
    return config


def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config    
if __name__=="__main__":
    arg=argparse.ArgumentParser()# Create an ArgumentParser object
    arg.add_argument("--config",default="params.yaml")# Add a command-line argument named "--config" with a default value of "params.yaml"

    passed_args=arg.parse_args() # Parse the command-line arguments and store the result in passed_args
    a=get_data(config_path=passed_args.config)