from src.utils import get_repo_root
from .classification_datasets import create_image_classification_dataloader
import os

DATASET_ROOT = os.path.join(get_repo_root(), "datasets")