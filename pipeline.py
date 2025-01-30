import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

class EdgeDetectionTransform:
    """
    A custom transform to perform edge detection on images using the Canny edge detection algorithm.

    Methods:
        __call__(img): Converts an input image into edge-detected format.
    """
    def __call__(self, img):
        """
        Converts the input image to grayscale, applies Canny edge detection, and converts the result back to an image.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: Edge-detected image with 3 color channels.
        """
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edges_3c = np.stack([edges] * 3, axis=-1)
        img_edges = Image.fromarray(edges_3c)
        return img_edges

class ShellRegressionModel(nn.Module):
    """
    A neural network model for predicting shell parameters using ResNet18 as a backbone.

    Attributes:
        model (torch.nn.Module): The modified ResNet18 model for regression.

    Methods:
        forward(x): Performs a forward pass of the model.
    """
    def __init__(self, num_outputs):
        """
        Initializes the ShellRegressionModel with the specified number of output parameters.

        Args:
            num_outputs (int): Number of output parameters to predict.
        """
        super(ShellRegressionModel, self).__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_outputs)
                
    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted output.
        """
        x = self.model(x)
        return x

def load_regression_models(shell_types, device):
    """
    Loads regression models for all shell types specified in the configuration.

    Args:
        shell_types (dict): Dictionary containing shell type metadata.
        device (torch.device): Device on which the models will be loaded.

    Returns:
        dict: A dictionary mapping shell types to their respective regression models.
    """
    regression_models = {}
    for shell_type, config in shell_types.items():
        model = ShellRegressionModel(num_outputs=config['num_outputs']).to(device)
        standardized_shell_type = shell_type.lower().replace(' ', '_')
        model_file_name = f'models\\best_model_{standardized_shell_type}.pth'
        if not os.path.isfile(model_file_name):
            print(f"Error: Model file '{model_file_name}' not found.")
            continue
        checkpoint = torch.load(model_file_name, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.parameter_columns = config['parameter_columns']
        regression_models[standardized_shell_type] = model
    return regression_models

def classify_and_predict_image(image, device, classification_model, regression_models, idx_to_class, shell_types, classification_transform, regression_transform):
    """
    Classifies the input image and predicts shell parameters.

    Args:
        image (PIL.Image.Image): The input image.
        device (torch.device): Device for computation.
        classification_model (torch.nn.Module): Trained classification model.
        regression_models (dict): Dictionary of regression models for shell types.
        idx_to_class (dict): Mapping from class indices to class names.
        shell_types (dict): Metadata for shell types.
        classification_transform (torchvision.transforms.Compose): Transform for classification.
        regression_transform (torchvision.transforms.Compose): Transform for regression.

    Returns:
        tuple: Predicted class, parameter dictionary, and closest match information.
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    edges_pil = Image.fromarray(edges)
    image_classification = classification_transform(edges_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = classification_model(image_classification)
        _, predicted = torch.max(outputs, 1)
        predicted_class_idx = predicted.item()
        predicted_class = idx_to_class[predicted_class_idx]
    
    standardized_predicted_class = predicted_class.lower().replace(' ', '_')
    
    if standardized_predicted_class not in regression_models:
        return f"Error: No regression model found for shell type '{predicted_class}'.", None, None
    
    regression_model = regression_models[standardized_predicted_class]
    image_regression = regression_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        parameters = regression_model(image_regression)
        parameters = parameters.cpu().numpy().flatten()
    
    parameter_columns = shell_types[standardized_predicted_class]['parameter_columns']
    parameter_dict = dict(zip(parameter_columns, parameters))
    
    closest_image_path, closest_data, closest_distance = find_closest_match(parameters, standardized_predicted_class, shell_types)
    
    return predicted_class, parameter_dict, (closest_image_path, closest_data, closest_distance)

def find_and_display_closest_match(input_image_path, predicted_parameters, shell_type, shell_types):
    """
    Finds and displays the closest matching image from the dataset.

    Args:
        input_image_path (str): Path to the input image.
        predicted_parameters (list): Predicted parameters for the shell.
        shell_type (str): Shell type.
        shell_types (dict): Metadata for shell types.
    """
    csv_file = shell_types[shell_type]['csv_file']
    image_dir = shell_types[shell_type]['image_dir']
    parameter_columns = shell_types[shell_type]['parameter_columns']
    
    data = pd.read_csv(csv_file)
    labels = data[parameter_columns].values
    distances = pairwise_distances([predicted_parameters], labels, metric='euclidean').flatten()
    closest_idx = np.argmin(distances)
    closest_distance = distances[closest_idx]
    closest_data = data.iloc[closest_idx]
    closest_image_name = closest_data['img']
    closest_image_path = os.path.join(image_dir, closest_image_name)
    
    plot_images(input_image_path, closest_image_path)
    
    print("\nClosest match in the dataset:")
    print(closest_data[['img'] + parameter_columns])
    print(f"\nDistance: {closest_distance:.4f}")

def find_closest_match(predicted_parameters, shell_type, shell_types):
    """
    Finds the closest matching shell image from the dataset based on predicted parameters.

    Args:
        predicted_parameters (list): Predicted shell parameters.
        shell_type (str): Shell type name.
        shell_types (dict): Dictionary containing metadata for shell types.

    Returns:
        tuple: (closest image path, closest data row, closest distance)
    """
    csv_file = shell_types[shell_type]['csv_file']
    image_dir = "shell_images"  # Folder in GitHub Repo
    parameter_columns = shell_types[shell_type]['parameter_columns']

    # Load dataset
    data = pd.read_csv(csv_file)
    labels = data[parameter_columns].values  # Extract feature columns
    distances = pairwise_distances([predicted_parameters], labels, metric='euclidean').flatten()

    # Find closest match
    closest_idx = distances.argmin()
    closest_distance = distances[closest_idx]
    closest_data = data.iloc[closest_idx]
    closest_image_name = closest_data['img']  # Image filename from CSV
    closest_image_path = os.path.join(image_dir, closest_image_name)

    return closest_image_path, closest_data, closest_distance

def modify_resnet18(num_classes):
    """
    Modifies ResNet18 for grayscale input and a specified number of output classes.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Modified ResNet18 model.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_classification_model(model_path, num_classes, device):
    """
    Loads a classification model from a saved checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        num_classes (int): Number of output classes.
        device (torch.device): Device to load the model on.

    Returns:
        tuple: Loaded model and a mapping of indices to class names.
    """
    model = modify_resnet18(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    class_to_idx = checkpoint['class_to_idx']
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return model, idx_to_class

def plot_images(original_image_path, closest_image_path):
    """
    Plots the input image and its closest matching image side by side.

    Args:
        original_image_path (str): Path to the original image.
        closest_image_path (str): Path to the closest matching image.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    original_image = Image.open(original_image_path)
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    closest_image = Image.open(closest_image_path)
    ax[1].imshow(closest_image)
    ax[1].set_title('Closest Matching Training Image')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()
