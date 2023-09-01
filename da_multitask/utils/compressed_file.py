import zipfile
import os
from typing import Union


def unzip_file(zip_path: str, destination_folder: str = None, extract_to_name: bool = True,
               del_zip: bool = False) -> Union[str, None]:
    """
    Extract a zip file.
    Args:
        zip_path: path to zip file
        destination_folder: folder to save output, default: same folder as input
        extract_to_name: whether to create a new folder with the same name as the zip file and extract into it,
        if False, save directly to destination_folder
        del_zip: whether to delete zip file after extraction

    Returns:
        path to extracted folder if successful, None otherwise
    """
    assert zip_path.endswith('.zip')
    input_folder, input_filename = os.path.split(zip_path)

    if not destination_folder:
        destination_folder = input_folder
    # create destination folder
    if extract_to_name:
        destination_folder = os.sep.join([destination_folder, input_filename[:-4]])
        os.makedirs(destination_folder, exist_ok=True)
    # extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    # delete zip file
    if del_zip:
        os.remove(zip_path)

    return destination_folder
