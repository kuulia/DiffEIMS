""" train_utils.py"""
import os
from typing import NoReturn


def make_result_dirs(dirs_to_create: list[str]) -> NoReturn:

    for path in dirs_to_create:
        os.makedirs(path, exist_ok=True)