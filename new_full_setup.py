"""Lab 5 Setup code"""

# All setup should be ready to go with this code, just needs the paths to the images and ground truth boxes

import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import libraries
import torch
import torch_snippets as ts
import torchvision.transforms as transforms
from PIL import Image
from torch.backends import mps
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.ops import nms

# Check for cuda
if torch.cuda.is_available():
    device = torch.device("cuda")
elif mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# # Suppress non-critical warnings for clarity
# import warnings

# warnings.filterwarnings("ignore")

"""FUNCTIONS FOR LOADING/PROCESSING IMAGES"""


def load_and_preprocess_custom_images(image_folder):
    """
    Loads and preprocesses custom images for inference.

    Args:
        image_folder (str): Path to the folder containing custom images.

    Returns:
        List[torch.Tensor]: List of preprocessed images as tensors.
    """
    # Define resizing and normalization transformations
    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),  # Resize to standard input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalization for ImageNet-trained models
        ]
    )

    # Load and transform each image
    images = []
    for image_path in glob.glob(f"{image_folder}/*.jpg"):
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        image = transform(image)  # Apply transformations
        images.append(image)

    return images


def load_and_preprocess_bus_dataset(bus_dataset_folder):
    """
    Loads and preprocesses bus dataset images for inference.

    Args:
        bus_dataset_folder (str): Path to the folder containing bus dataset images.

    Returns:
        List[torch.Tensor]: List of preprocessed images from the bus dataset.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    images = []
    for i, image_path in enumerate(glob.glob(f"{bus_dataset_folder}/*.jpg")):
        if i >= 10:  # Stop after 10 images
            break
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)

    return images


# For Debugging
def verify_image_properties(images):
    """
    Prints image tensor properties for verification.

    Args:
        images (List[torch.Tensor]): List of image tensors.
    """
    for i, img in enumerate(images):
        print(f"Image {i+1}: Shape={img.shape}, Type={img.dtype}")


"""FUNCTION FOR LOADING MODELS"""


def load_models():
    """
    Loads and initializes both Faster R-CNN and YOLOv5 models for object detection.

    Returns:
        Tuple[torch.nn.Module, torch.nn.Module]:
            A tuple containing initialized Faster R-CNN and YOLOv5 models.
    """
    # Load and initialize Faster R-CNN with ResNet backbone
    faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
    faster_rcnn.eval()  # Set to evaluation mode

    # Load and initialize YOLOv5 from PyTorch Hub
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    yolo.eval()  # Set to evaluation mode

    return faster_rcnn, yolo


"""FUNCTION FOR RUNNING INFERENCE"""


def run_inference_and_visualize(
    models, images, save_folder="inference_results", iou_threshold=0.5
):
    """
    Runs inference on a set of images using both Faster R-CNN and YOLOv5 models,
    applies NMS to Faster R-CNN predictions, then visualizes and saves the results with class labels.

    Args:
        models (tuple): Tuple containing the Faster R-CNN and YOLOv5 models.
        images (list): List of preprocessed images as tensors.
        save_folder (str): Path to save the annotated images.
        iou_threshold (float): IoU threshold for NMS.
    """
    faster_rcnn, yolo = models
    faster_rcnn_classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta[
        "categories"
    ]
    yolo_classes = yolo.names

    os.makedirs(save_folder, exist_ok=True)

    for i, image in enumerate(images):
        # Denormalize and prepare image for visualization
        img_display = image.permute(1, 2, 0).cpu()
        img_display = img_display * torch.tensor(
            [0.229, 0.224, 0.225]
        ) + torch.tensor([0.485, 0.456, 0.406])
        img_display = torch.clamp(img_display, 0, 1).numpy()

        # Faster R-CNN inference
        with torch.no_grad():
            faster_rcnn_prediction = faster_rcnn([image])[0]
            keep = nms(
                faster_rcnn_prediction["boxes"],
                faster_rcnn_prediction["scores"],
                iou_threshold,
            )
            faster_rcnn_prediction["boxes"] = faster_rcnn_prediction["boxes"][
                keep
            ]
            faster_rcnn_prediction["labels"] = faster_rcnn_prediction[
                "labels"
            ][keep]

        # Plot Faster R-CNN results
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img_display)

        for box, label in zip(
            faster_rcnn_prediction["boxes"], faster_rcnn_prediction["labels"]
        ):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                color="red",
                linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(
                x1,
                y1 - 5,
                faster_rcnn_classes[label],
                color="red",
                fontsize=12,
            )

        plt.title(f"Faster R-CNN - Image {i+1}")
        plt.savefig(f"{save_folder}/faster_rcnn_image_{i+1}.png")
        plt.close()

        # YOLOv5 inference
        with torch.no_grad():
            # Convert image to format expected by YOLOv5
            img_yolo = (img_display * 255).astype(np.uint8)  # Convert to uint8
            results = yolo(img_yolo)

            # Plot YOLOv5 results
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(img_display)

            # Get detections
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det.cpu().numpy()
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color="blue",
                    linewidth=2,
                )
                ax.add_patch(rect)
                ax.text(
                    x1,
                    y1 - 5,
                    f"{yolo_classes[int(cls)]}: {conf:.2f}",
                    color="blue",
                    fontsize=12,
                )

        plt.title(f"YOLOv5 - Image {i+1}")
        plt.savefig(f"{save_folder}/yolo_image_{i+1}.png")
        plt.close()


"""FUNCTIONS FOR EVALUATING MODELS"""

# All computation/analysis functions


def compute_iou(boxA, boxB):
    """
    Computes Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): Coordinates [x1, y1, x2, y2] of the first bounding box.
        boxB (list): Coordinates [x1, y1, x2, y2] of the second bounding box.

    Returns:
        float: IoU value between 0 and 1.
    """
    # Convert and reshape boxes to ensure correct dimensionality
    boxA = (
        torch.tensor(boxA, dtype=torch.float32).reshape(-1)
        if not isinstance(boxA, torch.Tensor)
        else boxA.reshape(-1)
    )
    boxB = (
        torch.tensor(boxB, dtype=torch.float32).reshape(-1)
        if not isinstance(boxB, torch.Tensor)
        else boxB.reshape(-1)
    )

    # Find intersecting box coordinates using tensor operations
    xA = torch.maximum(boxA[0], boxB[0])
    yA = torch.maximum(boxA[1], boxB[1])
    xB = torch.minimum(boxA[2], boxB[2])
    yB = torch.minimum(boxA[3], boxB[3])

    # Calculate intersection area using tensor operations
    interArea = torch.maximum(torch.tensor(0.0), xB - xA + 1) * torch.maximum(
        torch.tensor(0.0), yB - yA + 1
    )

    # Calculate box areas using tensor operations
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Calculate IoU
    iou = interArea / (boxAArea + boxBArea - interArea)

    return torch.mean(iou).item()


def calculate_accuracy(
    predictions, ground_truths, iou_threshold=0.5, confidence_threshold=0.5
):
    """
    Calculates the accuracy of model predictions based on IoU and confidence thresholds.

    Args:
        predictions (list): List of dictionaries containing 'boxes' and 'scores' for predictions.
        ground_truths (list): List of ground truth bounding boxes.
        iou_threshold (float): Minimum IoU for a prediction to be considered correct.
        confidence_threshold (float): Minimum confidence score for a prediction to be considered.

    Returns:
        float: Accuracy as the percentage of correct detections.
    """
    correct_detections = 0
    for pred in predictions:
        for gt in ground_truths:
            iou = compute_iou(pred["box"], gt)
            if iou >= iou_threshold and pred["score"] >= confidence_threshold:
                correct_detections += 1
                break  # Only count one match per ground truth to avoid multiple counting

    accuracy = correct_detections / len(predictions) if predictions else 0
    return accuracy * 100  # Return as a percentage


def measure_inference_time(model, images):
    """
    Measures the average inference time per image for a given model.

    Args:
        model (torch.nn.Module): The model to test.
        images (list): List of images to run inference on.

    Returns:
        float: Average inference time per image in seconds.
    """
    start_time = time.time()
    with torch.no_grad():
        for image in images:
            model([image])
    end_time = time.time()

    # Calculate average time per image
    average_time = (end_time - start_time) / len(images)
    return average_time


# Run all above functions
def evaluate_models(models, images, ground_truths):
    """
    Evaluates both Faster R-CNN and YOLOv5 models on a dataset and computes IoU, accuracy,
    and average inference time.

    Args:
        models (tuple): Tuple containing Faster R-CNN and YOLOv5 models.
        images (list): List of preprocessed images.
        ground_truths (list): List of ground truth bounding boxes for each image.

    Returns:
        dict: Dictionary containing evaluation metrics for each model.
    """
    faster_rcnn, yolo = models

    # Initialize results dictionary
    results = {
        "Faster R-CNN": {"IoU": [], "Accuracy": None, "Inference Time": None},
        "YOLOv5": {"IoU": [], "Accuracy": None, "Inference Time": None},
    }

    # Measure Faster R-CNN
    frcnn_predictions = [faster_rcnn([img])[0] for img in images]
    results["Faster R-CNN"]["IoU"] = [
        compute_iou(pred["boxes"], gt)
        for pred, gt in zip(frcnn_predictions, ground_truths)
    ]
    results["Faster R-CNN"]["Accuracy"] = calculate_accuracy(
        frcnn_predictions, ground_truths
    )
    results["Faster R-CNN"]["Inference Time"] = measure_inference_time(
        faster_rcnn, images
    )

    # Measure YOLOv5
    yolo_predictions = [yolo([img])[0] for img in images]
    results["YOLOv5"]["IoU"] = [
        compute_iou(pred["boxes"], gt)
        for pred, gt in zip(yolo_predictions, ground_truths)
    ]
    results["YOLOv5"]["Accuracy"] = calculate_accuracy(
        yolo_predictions, ground_truths
    )
    results["YOLOv5"]["Inference Time"] = measure_inference_time(yolo, images)

    return results


"""WRAPPER FUNCTION FOR ALL STEPS"""


def run(custom_images_folder, bus_dataset_folder, ground_truths):
    """
    Executes the complete lab pipeline: loads images, runs inference on both models,
    visualizes and saves results, and computes quantitative metrics.

    Args:
        custom_images_folder (str): Path to the folder with custom images.
        bus_dataset_folder (str): Path to the bus dataset folder.
        ground_truths (list): List of ground truth bounding boxes for the bus dataset.
    """
    # Step 1: Load and preprocess images
    # custom_images = load_and_preprocess_custom_images(custom_images_folder)
    bus_images = load_and_preprocess_bus_dataset(bus_dataset_folder)

    print("my custom images", custom_images)

    # Debugging: Verify image properties
    # verify_image_properties(custom_images)
    verify_image_properties(bus_images)

    print("my verify images ran")

    # Step 2: Load models
    models = load_models()

    print("my models", models)

    # Step 3: Run inference and visualize results on custom images
    # print("Running inference and visualizing results on custom images...")
    # run_inference_and_visualize(
    #     models, custom_images, save_folder="custom_inference_results"
    # )

    # Step 4: Evaluate models quantitatively on bus dataset images
    print("Evaluating models on the bus dataset...")
    results = evaluate_models(models, bus_images, ground_truths)

    # Step 5: Display quantitative results
    print("Quantitative Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Metrics:")
        print(f"Mean IoU: {np.mean(metrics['IoU']):.2f}")
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")
        print(
            f"Average Inference Time: {metrics['Inference Time']:.4f} seconds"
        )


"""MAIN FUNCTION"""


def main():
    custom_images_folder = "./images"
    bus_dataset_folder = (
        "./sixhky/open-images-bus-trucks/versions/1/images/images"
    )

    # Load ground truths from CSV
    df = pd.read_csv("./sixhky/open-images-bus-trucks/versions/1/df.csv")
    df = df.head(10)  # Take only first 10 entries

    # Format ground truths into list of bounding boxes
    ground_truths = []
    for _, row in df.iterrows():
        box = [
            row["XMin"] * 640,
            row["YMin"] * 640,
            row["XMax"] * 640,
            row["YMax"] * 640,
        ]
        ground_truths.append(box)

    run(custom_images_folder, bus_dataset_folder, ground_truths)


main()
