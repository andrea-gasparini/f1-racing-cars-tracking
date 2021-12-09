import os

from typing import List

def join_dirs(base_dir: str, subdirs: List[str]) -> List[str]:
    return [os.path.join(base_dir, subdir) for subdir in subdirs]
