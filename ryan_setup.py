

# Import required libraries
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from torchvision import ops
#import requests

# Check for cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Suppress non-critical warnings for clarity
import warnings
warnings.filterwarnings("ignore")


def load_models():
    """
    Loads the Faster R-CNN and YOLOv5 models with pre-trained weights, setting them to evaluation mode.
    Returns both models for further inference.
    """
    # Load Faster R-CNN with ResNet50 backbone
    model_rcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model_rcnn.eval()

    # Load YOLOv5 from PyTorch Hub
    model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
    model_yolo.eval()
    
    return model_rcnn, model_yolo

# Initialize models
model_rcnn, model_yolo = load_models()


# Dataset classes and helper functions
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class BusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f"{self.root_dir}/{i}.jpg" for i in range(1, 101)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def prepare_dataset(custom_image_paths, bus_dataset_path):
    """
    Prepares the custom and bus datasets with necessary transformations.
    Returns DataLoaders for both datasets.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare custom dataset and loader
    custom_dataset = CustomImageDataset(custom_image_paths, transform=transform)
    custom_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False)
    
    # Prepare bus dataset and loader
    bus_dataset = BusDataset(bus_dataset_path, transform=transform)
    bus_loader = DataLoader(bus_dataset, batch_size=4, shuffle=False)
    
    return custom_loader, bus_loader


def run_inference(model, dataloader):
    """
    Runs inference using the specified model on the given dataloader.
    Returns the detection results.
    """
    results = []
    for images in dataloader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
        results.extend(outputs)
    return results

# Functions for extracting results
def compute_IoU(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate union area
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def apply_NMS(boxes, scores, iou_threshold=0.5):
    """
    Applies Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
    """
    indices = ops.nms(boxes, scores, iou_threshold)
    return indices


def compute_metrics(predictions, ground_truth, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Computes detection metrics including accuracy, mean IoU, and inference time.
    Returns a dictionary of metric values.
    """
    correct_detections = 0
    total_detections = 0
    iou_sum = 0.0
    num_iou_matches = 0
    inference_times = []

    for pred, gt in zip(predictions, ground_truth):
        start_time = time.time()
        
        # Filter predictions by confidence threshold
        boxes_pred = [box for box, score in zip(pred['boxes'], pred['scores']) if score >= confidence_threshold]
        scores_pred = [score for score in pred['scores'] if score >= confidence_threshold]
        labels_pred = [label for label, score in zip(pred['labels'], pred['scores']) if score >= confidence_threshold]
        
        boxes_gt = gt['boxes']
        labels_gt = gt['labels']
        
        total_detections += len(labels_gt)

        for i, box_gt in enumerate(boxes_gt):
            matched = False
            for j, box_pred in enumerate(boxes_pred):
                iou = compute_IoU(box_gt, box_pred)
                if iou >= iou_threshold and labels_pred[j] == labels_gt[i]:
                    correct_detections += 1
                    iou_sum += iou
                    num_iou_matches += 1
                    matched = True
                    break
            
            if not matched:
                correct_detections += 0

        end_time = time.time()
        inference_times.append(end_time - start_time)

    accuracy = correct_detections / total_detections if total_detections > 0 else 0
    mean_iou = iou_sum / num_iou_matches if num_iou_matches > 0 else 0
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0

    return {
        "accuracy": accuracy,
        "mean_iou": mean_iou,
        "average_inference_time": avg_inference_time
    }


def plot_results(image, boxes, labels, scores):
    """
    Plots the detection results on the image with bounding boxes and labels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{label}: {score:.2f}", color='red', fontsize=12)
    plt.show()


# Execution commands
# Specify paths to your custom images and the bus dataset directory
custom_image_paths = []
bus_dataset_path = ""

# Prepare datasets and loaders
custom_loader, bus_loader = prepare_dataset(custom_image_paths, bus_dataset_path)

# Run inference on custom dataset for qualitative evaluation
custom_results_rcnn = run_inference(model_rcnn, custom_loader)
custom_results_yolo = run_inference(model_yolo, custom_loader)

# Run inference on bus dataset for quantitative evaluation
bus_results_rcnn = run_inference(model_rcnn, bus_loader)
bus_results_yolo = run_inference(model_yolo, bus_loader)

# Ground truth for bus dataset (example structure)
ground_truth_bus = [
    {'boxes': [[10, 20, 50, 60], [30, 40, 70, 80]], 'labels': [1, 2]}, 
    {'boxes': [[15, 25, 55, 65]], 'labels': [3]}
    # Extend this as per actual ground truth for all bus images
]

# Compute metrics
metrics_rcnn = compute_metrics(bus_results_rcnn, ground_truth_bus)
metrics_yolo = compute_metrics(bus_results_yolo, ground_truth_bus)
print("Faster R-CNN Metrics:", metrics_rcnn)
print("YOLOv5 Metrics:", metrics_yolo)

# Visualize results for custom images
for idx, (image, result_rcnn, result_yolo) in enumerate(zip(custom_loader, custom_results_rcnn, custom_results_yolo)):
    print(f"Image {idx + 1} - Faster R-CNN Results")
    plot_results(image, result_rcnn['boxes'], result_rcnn['labels'], result_rcnn['scores'])
    
    print(f"Image {idx + 1} - YOLOv5 Results")
    plot_results(image, result_yolo['boxes'], result_yolo['labels'], result_yolo['scores'])

