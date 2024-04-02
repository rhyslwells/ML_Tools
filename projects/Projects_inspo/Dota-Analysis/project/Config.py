import os

from pathlib import Path


def get_project_root():
    """Returns root of the project.

        Parameters:
        -----------
        None

        Returns:
        -----------
        root: pathlib.Path
            Root to the project saved withing pathlib.Path object.
    """
    root = Path(__file__).parent.parent
    return root


DATA_DIR = os.path.join(str(get_project_root()), "data")
IMAGE_DIR = os.path.join(str(get_project_root()), "image")
