"""Lab 5 Setup code"""

# All setup should be ready to go with this code, just needs the paths to the images and ground truth boxes

# Import libraries
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.ops import nms
import glob

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

import os
import torch_snippets as ts
import matplotlib.pyplot as plt

import time
import numpy as np

# Check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Suppress non-critical warnings for clarity
import warnings
warnings.filterwarnings("ignore")

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
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to standard input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet-trained models
    ])

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
    # Reuse the same transform pipeline for consistency
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to standard input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ImageNet-trained models
    ])

    # Load and transform each image in the bus dataset folder
    images = []
    for image_path in glob.glob(f"{bus_dataset_folder}/*.jpg"):
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
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo.eval()  # Set to evaluation mode

    return faster_rcnn, yolo


"""FUNCTION FOR RUNNING INFERENCE"""

def run_inference_and_visualize(models, images, save_folder="inference_results", iou_threshold=0.5):
    """
    Runs inference on a set of images using both Faster R-CNN and YOLOv5 models, 
    applies NMS to Faster R-CNN predictions, then visualizes and saves the results with class labels.

    Args:
        models (tuple): Tuple containing the Faster R-CNN and YOLOv5 models.
        images (list): List of preprocessed images as tensors.
        save_folder (str): Path to save the annotated images.
        iou_threshold (float): IoU threshold for NMS.
    """
    # Unpack models
    faster_rcnn, yolo = models

    # Load class labels for visualization
    faster_rcnn_classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    yolo_classes = yolo.names  # YOLO class names

    # Ensure the save directory exists
    os.makedirs(save_folder, exist_ok=True)
    
    # Iterate through images and perform inference with both models
    for i, image in enumerate(images):
        # Run inference with Faster R-CNN
        with torch.no_grad():
            faster_rcnn_prediction = faster_rcnn([image])[0]

            # Apply NMS to Faster R-CNN predictions
            keep = nms(faster_rcnn_prediction['boxes'], faster_rcnn_prediction['scores'], iou_threshold)
            faster_rcnn_prediction['boxes'] = faster_rcnn_prediction['boxes'][keep]
            faster_rcnn_prediction['scores'] = faster_rcnn_prediction['scores'][keep]
            faster_rcnn_prediction['labels'] = faster_rcnn_prediction['labels'][keep]

        # Display results for Faster R-CNN with class names
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ts.show(image, boxes=faster_rcnn_prediction['boxes'], 
                labels=[faster_rcnn_classes[label] for label in faster_rcnn_prediction['labels']], ax=ax)
        plt.title(f"Faster R-CNN - Image {i+1}")
        plt.savefig(f"{save_folder}/faster_rcnn_image_{i+1}.png")
        plt.show()
        
        # Run inference with YOLOv5
        with torch.no_grad():
            yolo_prediction = yolo([image])[0]

        # Display results for YOLOv5 with class names
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ts.show(image, boxes=yolo_prediction['boxes'], 
                labels=[yolo_classes[label] for label in yolo_prediction['labels']], ax=ax)
        plt.title(f"YOLOv5 - Image {i+1}")
        plt.savefig(f"{save_folder}/yolo_image_{i+1}.png")
        plt.show()




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
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def calculate_accuracy(predictions, ground_truths, iou_threshold=0.5, confidence_threshold=0.5):
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
            iou = compute_iou(pred['box'], gt)
            if iou >= iou_threshold and pred['score'] >= confidence_threshold:
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
        "YOLOv5": {"IoU": [], "Accuracy": None, "Inference Time": None}
    }
    
    # Measure Faster R-CNN
    frcnn_predictions = [faster_rcnn([img])[0] for img in images]
    results["Faster R-CNN"]["IoU"] = [compute_iou(pred["boxes"], gt) for pred, gt in zip(frcnn_predictions, ground_truths)]
    results["Faster R-CNN"]["Accuracy"] = calculate_accuracy(frcnn_predictions, ground_truths)
    results["Faster R-CNN"]["Inference Time"] = measure_inference_time(faster_rcnn, images)
    
    # Measure YOLOv5
    yolo_predictions = [yolo([img])[0] for img in images]
    results["YOLOv5"]["IoU"] = [compute_iou(pred["boxes"], gt) for pred, gt in zip(yolo_predictions, ground_truths)]
    results["YOLOv5"]["Accuracy"] = calculate_accuracy(yolo_predictions, ground_truths)
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
    custom_images = load_and_preprocess_custom_images(custom_images_folder)
    bus_images = load_and_preprocess_bus_dataset(bus_dataset_folder)

    # Debugging: Verify image properties
    verify_image_properties(custom_images)
    verify_image_properties(bus_images)

    # Step 2: Load models
    models = load_models()

    # Step 3: Run inference and visualize results on custom images
    print("Running inference and visualizing results on custom images...")
    run_inference_and_visualize(models, custom_images, save_folder="custom_inference_results")

    # Step 4: Evaluate models quantitatively on bus dataset images
    print("Evaluating models on the bus dataset...")
    results = evaluate_models(models, bus_images, ground_truths)

    # Step 5: Display quantitative results
    print("Quantitative Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name} Metrics:")
        print(f"Mean IoU: {np.mean(metrics['IoU']):.2f}")
        print(f"Accuracy: {metrics['Accuracy']:.2f}%")
        print(f"Average Inference Time: {metrics['Inference Time']:.4f} seconds")


"""MAIN FUNCTION"""
def main():
    custom_images_folder = "ADD PATH HERE"
    bus_dataset_folder = "ADD PATH HERE"
    ground_truths = [...]  # Placeholder

    # Run the lab pipeline
    run(custom_images_folder, bus_dataset_folder, ground_truths)