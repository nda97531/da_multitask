import os
from loguru import logger


def read_all_text(path: str) -> str:
    """
    Read a whole text file.

    Args:
        path: path to file

    Returns:
        a str containing the whole file content
    """
    with open(path, 'r') as F:
        content = F.read()
    return content


def write_text_file(content: str, path: str, overwrite: bool = False) -> bool:
    """
    Write a string into a text file.

    Args:
        content: content to write
        path: path to save file
        overwrite: overwrite if file already exists

    Returns:
        True if successfully; False otherwise
    """
    if (not overwrite) and os.path.isfile(path):
        logger.info(f'Not writing {path} because it already exists.')
        return False

    os.makedirs(os.path.split(path)[0], exist_ok=True)

    with open(path, 'w') as F:
        F.write(content)
    logger.info(f'File {path} saved.')
    return True


def read_last_line(path: str) -> str:
    """
    Read only that last line of a text file

    Args:
        path: path to file

    Returns:
        the last line of file
    """
    with open(path, 'rb') as f:
        try:  # catch OSError in case of a one line file
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    return last_line
