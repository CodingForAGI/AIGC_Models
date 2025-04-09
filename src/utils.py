import os
import sys
import yaml
import torch

def get_repo_root():
    cur_file = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(cur_file, "../"))
    return repo_root


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)

def load_yaml_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"The file {file_path} is not a valid YAML file.")
        sys.exit(1)
    return None

def get_dataset_root():
    repo_root = get_repo_root()
    print(f"Repo root: {repo_root}")
    yaml_data = load_yaml_config(file_path="project_cfg.yaml")
    print("Load project_cfg.yaml successfully.")
    dataset_root = os.path.join(repo_root, yaml_data["DATASET"]["ROOT"])
    print(f"Dataset root: {dataset_root}")

    return dataset_root