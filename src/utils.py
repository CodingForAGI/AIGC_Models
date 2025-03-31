import os


def get_repo_root():
    cur_file = os.path.abspath(__file__)
    repo_root = os.path.normpath(os.path.join(cur_file, "../"))
    return repo_root