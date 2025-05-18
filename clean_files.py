import os
import sys
from pathlib import Path

# Configuration for cleaning rules
CLEAN_RULES = [
    {
        'directory': './experiments',
        'file_pattern': '*.yaml',
        'description': '.yaml files'
    },
    {
        'directory': './log',
        'file_pattern': '*.log',
        'description': '.log files'
    },
    {
        'directory': './log',
        'file_pattern': 'events.out.tfevents*',
        'description': 'tensorboard log files'
    },
    {
        'directory': './output',
        'file_pattern': '*.pth',
        'description': '.pth files'
    }
]

def confirm_action():
    """Confirm with the user before performing the cleaning operation"""
    print("\n⚠️ WARNING: This operation will permanently delete the following files:")
    for rule in CLEAN_RULES:
        print(f"  - All {rule['description']} in the directory {os.path.abspath(rule['directory'])}")
    
    while True:
        response = input("\nContinue? (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            print("Operation cancelled.")
            return False

def clean_files():
    """Execute the file cleaning operation"""
    if not confirm_action():
        return
    
    total_deleted = 0
    failed_files = []
    
    for rule in CLEAN_RULES:
        dir_path = Path(rule['directory'])
        file_pattern = rule['file_pattern']
        description = rule['description']
        
        print(f"\nCleaning {description} in {dir_path.resolve()}...")
        
        # Check if the directory exists
        if not dir_path.exists():
            print(f"  Warning: Directory '{dir_path}' does not exist, skipping.")
            continue
        
        if not dir_path.is_dir():
            print(f"  Error: '{dir_path}' is not a directory, skipping.")
            continue
        
        # Get all files matching the pattern
        files_to_delete = list(dir_path.glob(file_pattern))
        
        if not files_to_delete:
            print(f"  No {description} found.")
            continue
        
        print(f"  Found {len(files_to_delete)} {description}, deleting...")
        
        # Delete files
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                total_deleted += 1
                print(f"  - Deleted: {file_path}")
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"  ! Failed to delete: {file_path} - {e}")
    
    # Print summary
    print("\n===== Cleaning Summary =====")
    print(f"Total files deleted: {total_deleted}")
    
    if failed_files:
        print(f"Files failed to delete: {len(failed_files)}")
        print("\nList of failed deletions:")
        for file_path, error in failed_files:
            print(f"  - {file_path}: {error}")
    else:
        print("All files deleted successfully.")

if __name__ == "__main__":
    clean_files()    