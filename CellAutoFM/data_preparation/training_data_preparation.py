import os
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, DatasetDict


def generate_labels(image_name):
    """
    Generate structured labels from the image filename.
    Args:
        image_name (str): The name of the image file.
    Returns:
        dict: A dictionary containing image metadata (date, condition, well ID).
    """
    try:
        parts = image_name.split('_')
        if len(parts) < 4:
            raise ValueError(f"Invalid image filename format: {image_name}")
        
        timepoint_id = parts[1]
        experiment_date = parts[2].split('-')[0]
        well_id = parts[3]
        
        # Define the feeding conditions based on the experiment date
        if experiment_date == "220518":
            fed_wells = ["C", "D", "E", "F", "G", "H"]
            starved_wells = ["I", "J", "K", "L", "M", "N"]
        elif experiment_date == "220524":
            fed_wells = ["C", "D", "E"]
            starved_wells = ["F", "G", "H", "I", "J", "K", "L", "M"]
        else:
            raise ValueError(f"Unknown experiment date: {experiment_date}")
        
        # Determine condition based on well ID's first character
        well_condition = "Fed" if well_id[0] in fed_wells else "Starved"
        
        # Create the structured label output
        labels = {
            "image_name": image_name,
            "timepoint_id": timepoint_id,
            "experiment_date": experiment_date,
            "well_id": well_id,
            "condition": well_condition
        }
        return labels
    
    except (IndexError, ValueError) as e:
        print(f"Error processing image {image_name}: {e}")
        return None


def load_and_resize_image(image_path, size=(512, 512)):
    """
    Load an image, convert it to RGB, and resize it.
    Args:
        image_path (str): Path to the image file.
        size (tuple): Target size for the resized image.
    Returns:
        Image: A resized PIL image, or None if loading fails.
    """
    try:
        return Image.open(image_path).convert("RGB").resize(size)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def train_data_preparation(folder_path, save_path, image_size=(512, 512), test_size=0.25):
    """
    Data preparation function to create a dataset, split it into training and validation sets, and save it to disk.
    Args:
        folder_path (str): Path to the folder containing the images.
        save_path (str): Path to save the processed dataset.
        image_size (tuple): Target size for resizing the images.
        test_size (float): Proportion of the dataset to be used for validation.
    """
    # Validate input parameters
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path does not exist: {folder_path}")
    
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    data_dict = {}
    
    # Get all .tif files
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
    
    if not tif_files:
        raise ValueError(f"No .tif files found in {folder_path}")
    
    print(f"Found {len(tif_files)} .tif files")
    
    # Process each file and generate labels
    for file in tif_files:
        labels = generate_labels(file)
        if labels is not None:  # Only process if labels were generated successfully
            image_path = os.path.join(folder_path, file)
            # Create combined label string
            combined_label = f"{labels['timepoint_id']} {labels['condition']}"
            data_dict[image_path] = combined_label
    
    if not data_dict:
        raise ValueError("No valid images found to process")
    
    print(f"Processing {len(data_dict)} valid images...")
    
    # Create dataset from images and labels
    images = []
    labels = []
    
    for image_path, label in tqdm(data_dict.items(), desc="Loading and processing images"):
        image = load_and_resize_image(image_path, size=image_size)
        if image is not None:
            images.append(image)
            labels.append(label)
    
    if not images:
        raise ValueError("No images were successfully loaded")
    
    # Create the dataset
    data = {
        "image": images,
        "label": labels
    }
    
    full_dataset = Dataset.from_dict(data)
    print(f"Created dataset with {len(full_dataset)} samples")
    
    # Split the dataset into training and validation sets
    train_test_split = full_dataset.train_test_split(test_size=test_size, seed=42)
    
    # Organize the split datasets into a DatasetDict
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"]
    })
    
    print(f"Train set: {len(dataset['train'])} samples")
    print(f"Validation set: {len(dataset['validation'])} samples")
    
    # Save the dataset to the specified path
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to: {save_path}")
    
    return dataset


if __name__ == "__main__":
    # Define parameters
    FOLDER_PATH = '/path/to/your/downloaded/image/files/'  # Replace with your actual image folder path
    SAVE_PATH = "./data/pretain_autophagy_datasets_tif"
    IMAGE_SIZE = (512, 512)
    TEST_SIZE = 0.25
    
    try:
        # Run the main function with specified parameters
        dataset = train_data_preparation(FOLDER_PATH, SAVE_PATH, IMAGE_SIZE, TEST_SIZE)
        print("Dataset preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise