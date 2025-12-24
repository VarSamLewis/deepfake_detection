import modal

app = modal.App("upload-model")

# Create volume
volume = modal.Volume.from_name("deepfake-model", create_if_missing=True)

@app.function(
    volumes={"/vol": volume},
    timeout=1800,
)
def upload_files(files_dict):
    """Upload model files to Modal volume"""
    import os

    # Write files to volume
    for remote_path, content in files_dict.items():
        filepath = f"/vol{remote_path}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(content)

    volume.commit()
    print(f"Uploaded {len(files_dict)} files to volume")


@app.local_entrypoint()
def main():
    """Upload model from local to Modal volume"""
    from pathlib import Path

    # Get the local model directory - works from modal_src or project root
    script_dir = Path(__file__).parent
    if script_dir.name == "modal_src":
        project_root = script_dir.parent
    else:
        project_root = script_dir

    local_model_dir = project_root / "model" / "models" / "deepfake-detector-v2"

    if not local_model_dir.exists():
        print(f"Error: Model directory not found at {local_model_dir}")
        return

    print(f"Uploading model from: {local_model_dir}")

    # Read all files from local directory
    files_dict = {}
    for file_path in local_model_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_model_dir)
            remote_path = f"/deepfake-detector-v2/{relative_path}"
            with open(file_path, 'rb') as f:
                files_dict[remote_path] = f.read()
            print(f"  Preparing: {relative_path}")

    print(f"\nUploading {len(files_dict)} files to Modal volume...")
    upload_files.remote(files_dict)

    print("Model uploaded to Modal volume successfully!")
    print("Volume name: deepfake-model")
    print("Model path in volume: /deepfake-detector-v2")
