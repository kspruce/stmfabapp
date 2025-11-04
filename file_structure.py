import os

def print_directory_tree(path, prefix="", max_depth=3, current_depth=0):
    if current_depth > max_depth:
        return
    
    items = sorted(os.listdir(path))
    for i, item in enumerate(items):
        if item.startswith('.'):  # Skip hidden files
            continue
            
        item_path = os.path.join(path, item)
        is_last = i == len(items) - 1
        
        print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
        
        if os.path.isdir(item_path) and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            print_directory_tree(item_path, prefix + extension, max_depth, current_depth + 1)

# Run this in your project root
print_directory_tree(".")
