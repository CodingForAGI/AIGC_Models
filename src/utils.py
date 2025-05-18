import os
import sys
import yaml
import torch
from datetime import datetime


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
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"The file {file_path} is not a valid YAML file.")
        sys.exit(1)
    return None


def save_to_yaml(data, file_path):
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"Error saving to YAML file: {e}")
        sys.exit(1)


def parse_project_cfg():
    repo_root = get_repo_root()
    print(f"Repo root: {repo_root}")
    yaml_data = load_yaml_config(file_path="project_cfg.yaml")
    print("Load project_cfg.yaml successfully.")
    dataset_root = os.path.join(repo_root, yaml_data["DATASET"]["ROOT"])
    model_save_root = os.path.join(repo_root, yaml_data["MODEL"]["SAVE_DIR"])
    log_root = os.path.join(repo_root, yaml_data["LOG"]["ROOT"])

    return {
        "dataset_root": dataset_root,
        "model_save_root": model_save_root,
        "log_root": log_root,
    }


PROJECT_CFG = parse_project_cfg()


def get_format_time():
    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_time


def get_log_file_path(task_name):
    log_root = PROJECT_CFG["log_root"]
    current_time = get_format_time()
    log_file_path = os.path.join(log_root, f"{task_name}_{current_time}.log")
    print(f"Save training log to: {log_file_path}")
    return log_file_path


def load_weights_from_training_status(model, weights_path, device=torch.device("cpu")):
    repo_root = get_repo_root()
    if not weights_path:
        raise ValueError(f"`weights_path` must be provided, but got: {weights_path}")
    weights_path = os.path.normpath(os.path.join(repo_root, weights_path))
    if os.path.exists(weights_path):
        print(f"weights_path: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        print(f"Successfully load weights from {weights_path}.")


def save_training_status(epoch, model, optimizer, loss, checkpoint_path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
