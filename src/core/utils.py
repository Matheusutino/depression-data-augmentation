import yaml
import json
import os
from typing import Any
from collections import defaultdict

def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a Python dictionary.

    :param file_path: Path to the YAML file.
    :return: Dictionary containing the YAML data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML file. {e}")

def read_json(file_path: str) -> Any:
    """Reads a JSON file and returns its content.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(file_path: str, data: Any) -> None:
    """Saves data as a JSON file.

    Args:
        data (Any): Data to be saved.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def create_directory(path: str) -> None:
    """Creates a directory if it does not exist.
    
    Args:
        path (str): The directory path to be created.
    """
    os.makedirs(path, exist_ok=True)

def check_directory_exists(directory_path: str) -> None:
    """Checks if a directory exists and raises an error if it does.

    Args:
        directory_path (str): Path to the directory.

    Raises:
        FileExistsError: If the directory already exists.
    """
    if os.path.exists(directory_path):
        raise FileExistsError(f"Directory '{directory_path}' already exists.")

def get_subdirectories(path):
    """
    Returns a list of all subdirectories within the given path.

    Args:
        path (str): The path to the directory to search for subdirectories.

    Returns:
        list: A list of paths of all subdirectories inside the given path.

    Raises:
        FileNotFoundError: If the specified path does not exist.
        NotADirectoryError: If the specified path is not a directory.
    """
    if not os.path.isdir(path):
        raise NotADirectoryError(f"The path {path} is not a directory or does not exist.")
    
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def str_to_bool(value: str) -> bool:
    """Converts a string representing a boolean value to a boolean.

    Args:
        value (str): String representation of a boolean ('True' or 'False').

    Returns:
        bool: Converted boolean value.

    Raises:
        ValueError: If the string is not 'True' or 'False'.
    """
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError("Invalid boolean string. Expected 'True' or 'False'.")
    
def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries, concatenating list values.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary with concatenated lists.
    """
    merged_dict = defaultdict(list)

    for key in set(dict1) | set(dict2):
        merged_dict[key] = dict1.get(key, []) + dict2.get(key, [])

    return dict(merged_dict)

def get_output_path(train_dataset_path: str, **kwargs) -> str:
    """Generates the output path based on the input training dataset path.
    
    Args:
        train_dataset_path (str): Path to the training dataset JSON file.
        **kwargs: Additional key-value pairs to append as subdirectories in the output path.
    
    Returns:
        str: The corresponding output path in the 'results' directory.
    """
    # Split the path
    parts = train_dataset_path.split("/")
    
    # Extract required elements
    dataset_name = parts[1]  # e.g., 'instagram'
    augmentation_type = parts[2]  # e.g., 'copy'
    if augmentation_type == "llm":
        model = parts[3]
        prompt_name = parts[4]
        n = parts[5] + "/" 
    elif augmentation_type == "contextual":
        model = parts[3]
        n = parts[4] + "/"
    else:
        n = parts[3] + "/"
    filename = os.path.basename(train_dataset_path)  # e.g., 'train_original.json'
    
    # Extract the part after 'train_'
    suffix = filename.replace("train_", "").replace(".json", "")

    if suffix == "original":
        augmentation_type = "no_augmentation"
        # Skip adding {n} if suffix is "original"
        n = ""
    
    # Append kwargs values as additional subdirectories
    kwargs_path = "/".join(str(value) for value in kwargs.values())
    
    # Construct the output path
    if kwargs_path:
        if augmentation_type == "llm":
            return f"results/{dataset_name}/{augmentation_type}/{suffix}/{model}/{prompt_name}/{n}{kwargs_path}"
        elif augmentation_type == "contextual":
            return f"results/{dataset_name}/{augmentation_type}/{suffix}/{model}/{n}{kwargs_path}"
        else:
            return f"results/{dataset_name}/{augmentation_type}/{suffix}/{n}{kwargs_path}"
        
    return f"results/{dataset_name}/{augmentation_type}/{n}"

