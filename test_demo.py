import torch
from torchvision import transforms
from PIL import Image
import timm
import os
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to load the model
def load_model(model_path, num_classes):
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to preprocess an image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to predict the class of an image
def predict_image(model, image_tensor, class_names):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, predicted_class = torch.max(probabilities, dim=1)

    return class_names[predicted_class.item()], max_prob.item()

# Function to display images side by side

def show_images_side_by_side(test_image_path, predicted_class, dataset_path, output_folder="predictions"):
    # Load the test image
    test_image = Image.open(test_image_path).convert("RGB")
    
    # Find a sample image from the predicted class folder
    predicted_class_folder = os.path.join(dataset_path, predicted_class)
    if not os.path.exists(predicted_class_folder):
        print(f"Predicted class folder not found: {predicted_class_folder}")
        return
    
    sample_image_path = os.path.join(predicted_class_folder, os.listdir(predicted_class_folder)[0])
    sample_image = Image.open(sample_image_path).convert("RGB")
    
    # Create the predictions folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(test_image)
    axes[0].set_title("Test Image")
    axes[0].axis("off")

    axes[1].imshow(sample_image)
    axes[1].set_title(f"Sample from Predicted Class: {predicted_class}")
    axes[1].axis("off")
    
    plt.tight_layout()
    
    # Save the figure
    output_file = os.path.join(output_folder, f"comparison_{predicted_class}.png")
    # plt.savefig(output_file)
    # plt.close(fig)  # Close the figure to free memory
    plt.show()
    print(f"Comparison image saved at: {output_file}")

if __name__ == "__main__":
    # Path to the model and dataset
    model_path = "model.pth"
    dataset_path = "CUB_200_2011/CUB_200_2011/images"
    test_images_path = "test_demo"  # Folder containing multiple test images

    # Load class names from the dataset
    class_names = sorted(os.listdir(dataset_path))

    # Load the model
    print("Loading the model...")
    model = load_model(model_path, num_classes=len(class_names))
    print("Model loaded successfully.")

    # Iterate through all images in the test folder
    print(f"Processing images in folder: {test_images_path}")
    for image_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, image_name)

        try:
            # Preprocess the image
            image_tensor = preprocess_image(image_path)

            # Predict the class
            predicted_class, probability = predict_image(model, image_tensor, class_names)
            print(f"Image: {image_name} | Predicted Class: {predicted_class} | Probability: {probability:.4f}")

            # Display the test image alongside a sample from the predicted class
            show_images_side_by_side(image_path, predicted_class, dataset_path)

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
 