"""
Inference Script (Make Predictions)
=====================================
After training, use this script to:
1. Recognize a face from a single image file
2. Compare two faces to check if they're the same person (verification)
3. Run on a batch of images and save results
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.dirname(__file__))
from models.cnn_model import build_model
from utils.data_loader import get_transforms, IMG_SIZE


def load_model(checkpoint_path):
    """Load a trained model from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at '{checkpoint_path}'. "
            "Please run train.py first!"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_name = checkpoint.get("model_name", "cnn")
    img_size = checkpoint.get("img_size", 100 if model_name == "cnn" else IMG_SIZE)
    grayscale = model_name == "cnn"

    model = build_model(model_name, checkpoint["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, test_transforms = get_transforms(img_size, grayscale=grayscale)

    print(f"Model loaded from epoch {checkpoint['epoch']} "
          f"(val acc: {checkpoint['val_acc']:.1f}%)")
    print(f"   Architecture: {model_name} | Classes: {checkpoint['class_names']}")

    return model, checkpoint["class_names"], test_transforms


def preprocess_image(image_path, transform):
    """Load and preprocess a single image for prediction."""
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0)


def predict(model, class_names, transform, image_path, top_k=3):
    """Predict the identity of a face in an image."""
    tensor = preprocess_image(image_path, transform)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(class_names)))

    print(f"\nPredictions for: {os.path.basename(image_path)}")
    print("-" * 40)
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        name = class_names[idx.item()]
        score = prob.item() * 100
        bar = "#" * int(score / 5)
        print(f"  {i+1}. {name:<25} {score:5.1f}%  {bar}")

    best_name = class_names[top_indices[0].item()]
    best_conf = top_probs[0].item() * 100
    return best_name, best_conf


def verify_faces(model, transform, image_path_1, image_path_2, threshold=0.7):
    """Face verification using cosine similarity between embeddings."""
    tensor1 = preprocess_image(image_path_1, transform)
    tensor2 = preprocess_image(image_path_2, transform)

    with torch.no_grad():
        emb1 = model.get_embedding(tensor1)
        emb2 = model.get_embedding(tensor2)

    similarity = F.cosine_similarity(emb1, emb2).item()

    print(f"\nFace Verification")
    print(f"   Image 1: {os.path.basename(image_path_1)}")
    print(f"   Image 2: {os.path.basename(image_path_2)}")
    print(f"   Similarity: {similarity:.4f} (threshold: {threshold})")

    if similarity > threshold:
        print(f"   SAME PERSON  (confidence: {similarity*100:.1f}%)")
    else:
        print(f"   DIFFERENT PEOPLE  (similarity too low)")

    return similarity > threshold, similarity


def batch_predict(model, class_names, transform, image_dir, output_file="predictions.csv"):
    """Run predictions on all images in a directory and save results to CSV."""
    import csv

    results = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    image_files = [
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]

    print(f"\nRunning batch prediction on {len(image_files)} images...")

    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        try:
            name, conf = predict(model, class_names, transform, fpath, top_k=1)
            results.append({"file": fname, "predicted": name, "confidence": f"{conf:.1f}%"})
        except Exception as e:
            results.append({"file": fname, "predicted": "ERROR", "confidence": str(e)})

    with open(output_file, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=["file", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition Inference")
    parser.add_argument("--model", default="checkpoints/best_model.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--image", default=None,
                        help="Single image to predict")
    parser.add_argument("--verify", nargs=2, metavar=("IMG1", "IMG2"),
                        help="Two images to compare (face verification)")
    parser.add_argument("--batch", default=None,
                        help="Directory of images for batch prediction")
    args = parser.parse_args()

    model, class_names, transform = load_model(args.model)

    if args.image:
        predict(model, class_names, transform, args.image)

    elif args.verify:
        verify_faces(model, transform, args.verify[0], args.verify[1])

    elif args.batch:
        batch_predict(model, class_names, transform, args.batch)

    else:
        print("No action specified. Use --image, --verify, or --batch.")
        print("Example: python predict.py --image my_face.jpg")
