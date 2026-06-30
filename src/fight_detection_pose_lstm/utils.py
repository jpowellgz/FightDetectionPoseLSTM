import os
import json

def check_path(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

def load_config(path: str):
    config = {}
    with open(path, 'r') as file_obj:
        config = json.load(file_obj)
    return config