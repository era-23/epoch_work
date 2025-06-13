import os
from pathlib import Path
import shutil
import argparse
import fnmatch

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
        required = True,
        type=Path
    )
    parser.add_argument(
        "--pattern",
        action="store",
        help="File pattern to match (e.g., '*.txt').",
        required = False,
        type=str,
        default="input.deck"  # Default pattern can be changed as needed
    )

    args = parser.parse_args()
    source_directory = args.source
    
    destination_directory = args.destination
    file_pattern = args.pattern
    print(f"Copying files from {source_directory} to {destination_directory} matching pattern '{file_pattern}'")
    if not source_directory.is_dir():
        raise ValueError(f"Source directory {source_directory} does not exist or is not a directory.")
    if not destination_directory.is_dir():
        raise ValueError(f"Destination directory {destination_directory} does not exist or is not a directory.")
    if not file_pattern:
        raise ValueError("File pattern must be specified.")
    
    copy_files_matching_pattern(source_directory, destination_directory, file_pattern)