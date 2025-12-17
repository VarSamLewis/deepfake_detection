import os
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed


print("All imports successful!")

# Model and dataset paths (relative to project root)
project_root = Path(__file__).parent.parent
model_path = project_root / "model" / "models" / "deepfake-detector-v2"
dataset_path = project_root / "data" / "Dataset" / "Test"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

print(f"Loading model from: {model_path}")
model = ViTForImageClassification.from_pretrained(str(model_path), torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print(f"✓ Model loaded on: {device}")
print(f"✓ Model: {model_path}")
print(f"✓ Labels: {model.config.id2label}")
print(f"✓ Dtype: {next(model.parameters()).dtype}")

def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image)
        #input_batch = input_tensor.unsqueeze(0).to(device)
        input_batch = input_tensor.unsqueeze(0).to(device).half()

        with torch.no_grad():
            outputs = model(input_batch)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probs[0][predicted_class].item()

        label = model.config.id2label[predicted_class]
        swapped_label = 'Deepfake' if label == 'Realism' else 'Realism'

        return {
            'label': swapped_label,
            'score': confidence
        }
    except Exception as e:
        return {
            'label': 'Error',
            'score': 0.0,
            'error': str(e)
        }
results = []
MAX_WORKERS = os.cpu_count()

img_files = list(Path(dataset_path).rglob("*.jpg"))

print(f"\nProcessing {len(img_files)} images in parallel with {MAX_WORKERS} workers...")

print(f"Found {len(img_files)} images in test set")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_img = {executor.submit(predict_image, str(img_path)): img_path for img_path in img_files}
    
    for future in tqdm(as_completed(future_to_img), total=len(img_files), desc="Testing (Parallel)"):
        img_path = future_to_img[future]
        
        try:
            
            result = future.result() 
            parent_folder = img_path.parent.name.lower()
            if 'fake' in parent_folder or 'deepfake' in parent_folder:
                true_label = 'Deepfake'
            elif 'real' in parent_folder:
                true_label = 'Realism'
            else:
                true_label = 'Unknown'

            results.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'true_label': true_label,
                'predicted_label': result['label'],
                'confidence': result['score']
            })

        except Exception as exc:
            print(f'{img_path} generated an exception: {exc}')
            
            results.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'true_label': 'Unknown',
                'predicted_label': 'Error',
                'confidence': 0.0,
                'error': str(exc)
            })

print("\nParallel processing complete.")
