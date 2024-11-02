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

    images = {}
    i = 0
    for _, image_path in enumerate(glob.glob(f"{bus_dataset_folder}/*.jpg")):
        if i == 3:
            break
        i += 1
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images[image_path.split("/")[-1]] = image

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
        Tuple[torch.nn.Module, torch.nn.Module, list]: 
            A tuple containing initialized Faster R-CNN and YOLOv5 models and the Faster R-CNN classes.
    """
    # Load and initialize Faster R-CNN with ResNet backbone
    faster_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    faster_rcnn.eval()  # Set to evaluation mode
    faster_rcnn_classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    # Load and initialize YOLOv5 from PyTorch Hub
    yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    yolo.eval()  # Set to evaluation mode

    return faster_rcnn, yolo, faster_rcnn_classes


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
    faster_rcnn, yolo, faster_rcnn_classes = models
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

def compute_iou(pred_boxes, gt_box):
    """
    Computes the maximum IoU between a list of prediction boxes and a single ground truth box.

    Args:
        pred_boxes (list): List of dictionaries containing 'box' and 'score' for predictions.
        gt_box (list): Single ground truth box coordinates [x1, y1, x2, y2].

    Returns:
        float: Maximum IoU value for all matching boxes.
    """
    max_iou = 0.0
    gt_box = torch.as_tensor(gt_box, dtype=torch.float32)

    for pred in pred_boxes:
        pred_box = torch.as_tensor(pred["box"], dtype=torch.float32)
        # Compute intersection
        xA = torch.maximum(pred_box[0], gt_box[0])
        yA = torch.maximum(pred_box[1], gt_box[1])
        xB = torch.minimum(pred_box[2], gt_box[2])
        yB = torch.minimum(pred_box[3], gt_box[3])
        inter_area = torch.maximum(torch.tensor(0.0), xB - xA + 1) * torch.maximum(torch.tensor(0.0), yB - yA + 1)

        # Compute union area
        pred_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
        gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
        union_area = pred_area + gt_area - inter_area

        # Compute IoU
        iou = inter_area / union_area
        max_iou = max(max_iou, iou.item())

    return max_iou


def calculate_accuracy(predictions, ground_truths, iou_threshold=0.5, confidence_threshold=0.5):
    """
    Calculates the accuracy of model predictions based on IoU and confidence thresholds.

    Args:
        predictions (dict): Dictionary containing predicted boxes for each image.
        ground_truths (dict): Dictionary of ground truth bounding boxes for each image.
        iou_threshold (float): Minimum IoU for a prediction to be considered correct.
        confidence_threshold (float): Minimum confidence score for a prediction to be considered.

    Returns:
        float: Accuracy as the percentage of correct detections.
    """
    correct_detections = 0
    total_ground_truths = len(ground_truths)
    
    for img_name, gt_info in ground_truths.items():
        gt_box = gt_info['box']
        gt_class = gt_info['class']

        # Filter predictions by matching class and confidence threshold
        matching_preds = [pred for pred in predictions[img_name] 
                          if pred["class"] == gt_class and pred["score"] >= confidence_threshold]
        
        if not matching_preds:
            continue  # No predictions match this ground truth class
        
        # Find the best IoU for matching predictions with the ground truth box
        best_iou = compute_iou(matching_preds, gt_box)
        
        if best_iou >= iou_threshold:
            correct_detections += 1

    accuracy = (correct_detections / total_ground_truths) * 100 if total_ground_truths > 0 else 0
    return accuracy


def measure_inference_time(model, name, images):
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
            image = images[image]
            if name == "YOLOv5":  # Check if it's YOLOv5
                # Convert tensor to numpy, then back to proper format for YOLO
                img_np = image.cpu().numpy()
                img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
                model(img_np)
            else:
                model([image])
    end_time = time.time()

    # Calculate average time per image
    average_time = (end_time - start_time) / len(images)
    return average_time


def evaluate_models(models, images, ground_truths, faster_rcnn_classes):
    """
    Evaluates both Faster R-CNN and YOLOv5 models on a dataset and computes IoU, accuracy,
    and average inference time.

    Args:
        models (tuple): Tuple containing Faster R-CNN and YOLOv5 models.
        images (list): List of preprocessed images.
        ground_truths (dict): Dictionary of ground truth bounding boxes for each image.

    Returns:
        dict: Dictionary containing evaluation metrics for each model.
    """
    faster_rcnn, yolo = models
    results = {
        "Faster R-CNN": {"IoU": [], "Accuracy": None, "Inference Time": None},
        "YOLOv5": {"IoU": [], "Accuracy": None, "Inference Time": None},
    }

    # Faster R-CNN predictions
    frcnn_predictions = {}
    with torch.no_grad():
        for name in images:
            img = images[name]
            pred = faster_rcnn([img])[0]
            if len(pred["boxes"]) > 0:
                scores, indices = torch.topk(pred["scores"], len(pred["scores"]))
                frcnn_predictions[name] = []
                for idx in indices:
                    frcnn_predictions[name].append(
                        {
                            "box": pred["boxes"][idx].cpu().numpy(),
                            "score": pred["scores"][idx].item(),
                            "class": faster_rcnn_classes[pred["labels"][idx].item()]
                        }
                    )
            else:
                frcnn_predictions[name] = []

    # YOLOv5 predictions
    yolo_predictions = {}
    with torch.no_grad():
        for name in images:
            img = images[name]
            img_np = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
            img_np = ((img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]) * 255  # Denormalize
            pred = yolo(img_np.astype(np.uint8))
            
            yolo_predictions[name] = []
            if len(pred.xyxy[0]) > 0:
                scores, indices = torch.topk(pred.xyxy[0][:, 4], len(pred.xyxy[0]))
                for idx in indices:
                    yolo_predictions[name].append(
                        {
                            "box": pred.xyxy[0][idx][:4].cpu().numpy(),
                            "score": pred.xyxy[0][idx][4].item(),
                            "class": yolo.names[int(pred.xyxy[0][idx][5].item())]
                        }
                    )
    
    # Calculate metrics
    results["Faster R-CNN"]["IoU"] = [
        compute_iou(frcnn_predictions[name], ground_truths[name]['box'])
        for name in ground_truths
    ]
    results["YOLOv5"]["IoU"] = [
        compute_iou(yolo_predictions[name], ground_truths[name]['box'])
        for name in ground_truths
    ]
    
    results["Faster R-CNN"]["Accuracy"] = calculate_accuracy(frcnn_predictions, ground_truths)
    results["YOLOv5"]["Accuracy"] = calculate_accuracy(yolo_predictions, ground_truths)
    
    results["Faster R-CNN"]["Inference Time"] = measure_inference_time(faster_rcnn, "Faster R-CNN", images)
    results["YOLOv5"]["Inference Time"] = measure_inference_time(yolo, "YOLOv5", images)
    
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
    # Load and preprocess images
    # custom_images = load_and_preprocess_custom_images(custom_images_folder)
    bus_images = load_and_preprocess_bus_dataset(bus_dataset_folder)

    # Format ground truths with fixed scaling to match our resized images
    resized_ground_truths = {}
    for _, row in ground_truths.iterrows():
        if f"{row['ImageID']}.jpg" in bus_images:
            box = [
                row["XMin"] * 640,  # Scale to match resized image width
                row["YMin"] * 640,  # Scale to match resized image height
                row["XMax"] * 640,
                row["YMax"] * 640,
            ]
            resized_ground_truths[f"{row['ImageID']}.jpg"] = {"box": box, "class": row["ClassName"]}

    # Load models
    faster_rcnn, yolo, faster_rcnn_classes = load_models()

    # Run inference on custom images
    print("Running inference and visualizing results on custom images...")
    # run_inference_and_visualize(
    #     models, custom_images, save_folder="custom_inference_results"
    # )

    # Evaluate models on bus dataset
    print("Evaluating models on the bus dataset...")
    results = evaluate_models((faster_rcnn, yolo), bus_images, resized_ground_truths, faster_rcnn_classes)

    # Display results
    print("\nFaster R-CNN Metrics:")
    print(f"Mean IoU: {np.mean(results['Faster R-CNN']['IoU']):.2f}")
    print(f"Accuracy: {results['Faster R-CNN']['Accuracy']:.2f}%")
    print(
        f"Average Inference Time: {results['Faster R-CNN']['Inference Time']:.4f} seconds"
    )

    print("\nYOLOv5 Metrics:")
    print(f"Mean IoU: {np.mean(results['YOLOv5']['IoU']):.2f}")
    print(f"Accuracy: {results['YOLOv5']['Accuracy']:.2f}%")
    print(
        f"Average Inference Time: {results['YOLOv5']['Inference Time']:.4f} seconds"
    )


"""MAIN FUNCTION"""

def main():
    custom_images_folder = "./images"
    bus_dataset_folder = (
        "./sixhky/open-images-bus-trucks/versions/1/images/less_images"
    )

    # Load ground truths from CSV
    df = pd.read_csv("./sixhky/open-images-bus-trucks/versions/1/df.csv")

    run(custom_images_folder, bus_dataset_folder, df)


main()
