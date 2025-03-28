import os

# List of directory names to skip entirely
skip_dirs = {'.venv', 'external libraries', 'scratches', 'consoles', 'bin', 'lib', '.git'}

for root, dirs, files in os.walk('.'):
    # Remove any directories that are in the skip list
    dirs[:] = [d for d in dirs if d not in skip_dirs]

    # At the top-level, include the data folder but avoid traversing it
    if root == '.' and 'data' in dirs:
        print("Current Directory:", root)
        print("Subdirectories:", dirs)
        print("Files:", files)
        print("-" * 40)
        # Remove 'data' to skip its internal contents
        dirs.remove('data')
    else:
        print("Current Directory:", root)
        print("Subdirectories:", dirs)
        print("Files:", files)
        print("-" * 40)
