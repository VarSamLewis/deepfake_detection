import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("deepfake-detection")

# Define Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.57.3",
        "torch==2.9.1",
        "torchvision==0.24.1",
        "pandas==2.2.3",
        "numpy==2.2.0",
        "Pillow==11.0.0",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
        "scikit-learn==1.8.0",
        "tqdm==4.67.1",
    )
)

# Create persistent volumes for model and dataset
model_volume = modal.Volume.from_name("deepfake-model", create_if_missing=True)
dataset_volume = modal.Volume.from_name("deepfake-dataset", create_if_missing=True)

# Volume mount paths
MODEL_DIR = "/model"
DATASET_DIR = "/dataset"
GPU = "B200"

@app.function(
    image=image,
    gpu=GPU,
    volumes={
        MODEL_DIR: model_volume,
        DATASET_DIR: dataset_volume,
    },
    timeout=7200,
)
def run_inference():
    """Run deepfake detection inference on the test dataset using L4 GPU"""
    import torch
    import torchvision.transforms as transforms
    from transformers import ViTForImageClassification
    from PIL import Image
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Verify GPU availability
    assert torch.cuda.is_available(), "GPU not available!"
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")

    # Setup paths
    model_path = Path(f"{MODEL_DIR}/deepfake-detector-v2")
    dataset_path = Path(f"{DATASET_DIR}/Dataset/Test")

    print(f"Loading model from: {model_path}")
    print(f"Dataset path: {dataset_path}")

    # Load model
    model = ViTForImageClassification.from_pretrained(str(model_path), torch_dtype=torch.float16)
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded on: {device}")
    print(f"✓ Labels: {model.config.id2label}")
    print(f"✓ Dtype: {next(model.parameters()).dtype}")

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def predict_image(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(device).half()

            with torch.no_grad():
                outputs = model(input_batch)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(logits, dim=1).item()
                confidence = probs[0][predicted_class].item()

            label = model.config.id2label[predicted_class]
            # Swap labels as in original code
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

    # Get all test images
    img_files = list(dataset_path.rglob("*.jpg"))
    print(f"\nFound {len(img_files)} images in test set")

    results = []
    MAX_WORKERS = os.cpu_count()

    print(f"Processing {len(img_files)} images in parallel with {MAX_WORKERS} workers...")

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

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate metrics
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

    valid_results = df[df['predicted_label'] != 'Error']

    if len(valid_results) > 0:
        accuracy = accuracy_score(valid_results['true_label'], valid_results['predicted_label'])
        f1 = f1_score(valid_results['true_label'], valid_results['predicted_label'], average='weighted')
        precision = precision_score(valid_results['true_label'], valid_results['predicted_label'], average='weighted')
        recall = recall_score(valid_results['true_label'], valid_results['predicted_label'], average='weighted')

        print(f"\n{'='*50}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*50}")
        print(f"Total images processed: {len(df)}")
        print(f"Successful predictions: {len(valid_results)}")
        print(f"Errors: {len(df) - len(valid_results)}")
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(valid_results['true_label'], valid_results['predicted_label']))

        return {
            "status": "success",
            "total_images": len(df),
            "successful_predictions": len(valid_results),
            "errors": len(df) - len(valid_results),
            "metrics": {
                "accuracy": float(accuracy),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            },
            "results_preview": results[:10]
        }
    else:
        print("No valid predictions made!")
        return {
            "status": "error",
            "message": "No valid predictions",
            "total_images": len(df)
        }


@app.local_entrypoint()
def main():
    """Run inference on uploaded model and dataset"""
    print("\n" + "="*60)
    print(f"Running Inference on {GPU} GPU")
    print("="*60)
    result = run_inference.remote()
    print(f"\nInference result: {result}")
