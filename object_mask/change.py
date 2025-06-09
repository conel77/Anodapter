import os

root_dir = "/home/work/smk/anodapter_final/Object_mask"

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == "000.png":
            old_path = os.path.join(subdir, file)
            new_path = os.path.join(subdir, "0001.png")
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")