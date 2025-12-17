import torch
from transformers import ViTForImageClassification, ViTImageProcessor
import os

def download_model():
    model_name = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    save_dir = "./models/deepfake-detector-v2"

    print(f"Downloading model: {model_name}")

    # Download model and processor
    model = ViTForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    processor = ViTImageProcessor.from_pretrained(model_name, torch_dtype=torch.float16)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save locally
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"✓ Model downloaded and saved to {save_dir}")
    print(f"✓ Model size: {sum(p.numel() for p in model.parameters())} parameters")

if __name__ == "__main__":
    download_model()
