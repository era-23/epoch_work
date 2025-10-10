import os
from pathlib import Path
import shutil
import argparse
import fnmatch

def modify_line_in_matching_files(src_dir, pattern, search_text, replace_text):
    """
    Modify a specific line in files matching the given pattern.

    :param src_dir: Source directory to search for files.
    :param pattern: Pattern to match files (e.g., '*.txt').
    :param search_text: Text to search for in the file.
    :param replace_text: Text to replace the matching line with.
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                with open(file_path, 'w') as f:
                    for line in lines:
                        if search_text in line:
                            print(f"Replacing {search_text} with {replace_text} in {f.name}....")
                            f.write(line.replace(search_text, replace_text))
                        else:
                            f.write(line)

def copy_files_matching_pattern(src_dir, dest_dir, pattern):
    """
    Recursively copy files from src_dir to dest_dir if they match the given pattern.

    :param src_dir: Source directory to search for files.
    :param dest_dir: Destination directory to copy files to.
    :param pattern: Pattern to match files (e.g., '*.txt').
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, _, files in os.walk(src_dir):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                src_file = os.path.join(root, file)
                relative_path = os.path.relpath(root, src_dir)
                dest_subdir = os.path.join(dest_dir, relative_path)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                shutil.copy2(src_file, os.path.join(dest_subdir, file))

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser("parser")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files matching the pattern from source to destination."
    )
    parser.add_argument(
        "--modify",
        action="store_true",
        help="Modify a specific line in files matching the pattern."
    )
    parser.add_argument(
        "--source",
        action="store",
        help="Source directory.",
        required = True,
        type=Path
    )
    parser.add_argument(
        "--destination",
        action="store",
        help="Destination directory.",
        required = False,
        type=Path
    )
    parser.add_argument(
        "--file_pattern",
        action="store",
        help="File pattern to match (e.g., '*.txt').",
        required = True,
        type=str
    )
    parser.add_argument(
        "--search_text",
        action="store",
        help="Text to search for in the files.",
        required = False,
        type=str
    )
    parser.add_argument(
        "--replace_text",
        action="store",
        help="Text to replace the matching line with.",
        required = False,
        type=str
    )

    args = parser.parse_args()
    source_directory = args.source
    file_pattern = args.file_pattern

    if args.copy:
        destination_directory = args.destination
        print(f"Copying files from {source_directory} to {destination_directory} matching pattern '{file_pattern}'")
        if not source_directory.is_dir():
            raise ValueError(f"Source directory {source_directory} does not exist or is not a directory.")
        if not destination_directory.is_dir():
            raise ValueError(f"Destination directory {destination_directory} does not exist or is not a directory.")
        if not file_pattern:
            raise ValueError("File pattern must be specified.")
        
        copy_files_matching_pattern(source_directory, destination_directory, file_pattern)

    if args.modify:
        search_text = args.search_text
        replace_text = args.replace_text
        print(f"Modifying files in {source_directory} matching pattern '{file_pattern}'")
        if not source_directory.is_dir():
            raise ValueError(f"Source directory {source_directory} does not exist or is not a directory.")
        if not file_pattern:
            raise ValueError("File pattern must be specified.")
        
        modify_line_in_matching_files(source_directory, file_pattern, search_text, replace_text)