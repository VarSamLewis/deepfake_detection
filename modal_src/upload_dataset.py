import modal

app = modal.App("upload-dataset")

# Create volume
volume = modal.Volume.from_name("deepfake-dataset", create_if_missing=True)

@app.function(
    volumes={"/vol": volume},
    timeout=3600,
)
def upload_files(files_dict, split_name):
    """Upload dataset files to Modal volume"""
    import os

    # Write files to volume
    for remote_path, content in files_dict.items():
        filepath = f"/vol{remote_path}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(content)

    volume.commit()
    print(f"Uploaded {len(files_dict)} files for {split_name} split")


@app.local_entrypoint()
def main():
    """Upload dataset from local to Modal volume"""
    from pathlib import Path

    # Get the local dataset directory - works from modal_src or project root
    script_dir = Path(__file__).parent
    if script_dir.name == "modal_src":
        project_root = script_dir.parent
    else:
        project_root = script_dir

    local_dataset_dir = project_root / "data" / "Dataset"

    if not local_dataset_dir.exists():
        print(f"Error: Dataset directory not found at {local_dataset_dir}")
        return

    print(f"Uploading dataset from: {local_dataset_dir}")

    # Upload Test folder
    test_dir = local_dataset_dir / "Test"
    if test_dir.exists():
        print(f"\nPreparing Test dataset from: {test_dir}")
        files_dict = {}
        for file_path in test_dir.rglob("*.jpg"):
            relative_path = file_path.relative_to(test_dir)
            remote_path = f"/Dataset/Test/{relative_path}"
            with open(file_path, 'rb') as f:
                files_dict[remote_path] = f.read()

        print(f"Uploading {len(files_dict)} test images...")
        upload_files.remote(files_dict, "Test")
        print("✓ Test dataset uploaded")

    print("\nDataset uploaded to Modal volume successfully!")
    print("Volume name: deepfake-dataset")
    print("Dataset path in volume: /Dataset/")
