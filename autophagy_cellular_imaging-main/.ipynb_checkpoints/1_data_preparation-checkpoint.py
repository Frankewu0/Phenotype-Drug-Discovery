import os
from PIL import Image
from tqdm.notebook import tqdm
from datasets import Dataset, DatasetDict

# Function to generate labels from image names
def generate_labels(image_name):
    """
    Generate structured labels from the image filename.
    Args:
        image_name (str): The name of the image file.
    Returns:
        dict: A dictionary containing image metadata (date, condition, well ID).
    """
    experiment_date = image_name.split('_')[2].split('-')[0]
    well_id = image_name.split('_')[3]
    timepoint_id = image_name.split('_')[1]

    # Define the feeding conditions based on the experiment date
    if experiment_date == "220518":
        fed_wells = ["C", "D", "E", "F", "G", "H"]
        starved_wells = ["I", "J", "K", "L", "M", "N"]
    elif experiment_date == "220524":
        fed_wells = ["C", "D", "E"]
        starved_wells = ["F", "G", "H", "I", "J", "K", "L", "M"]
    else:
        return "Unknown experiment date"

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

# Function to load and resize an image
def load_and_resize_image(image_path, size=(512, 512)):
    """
    Load an image, convert it to RGB, and resize it.
    Args:
        image_path (str): Path to the image file.
        size (tuple): Target size for the resized image.
    Returns:
        Image: A resized PIL image.
    """
    return Image.open(image_path).convert("RGB").resize(size)

# Main function to create, split, and save the dataset
def main(folder_path, save_path, image_size=(512, 512), test_size=0.25):
    """
    Main function to create a dataset, split it into training and validation sets, and save it to disk.
    Args:
        folder_path (str): Path to the folder containing the images.
        save_path (str): Path to save the processed dataset.
        image_size (tuple): Target size for resizing the images.
        test_size (float): Proportion of the dataset to be used for validation.
    """
    data_dict = {}
    # Iterate over the files in the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith('.tif'):
            # Generate labels for each image
            tmp = generate_labels(file)
            data_dict[os.path.join(folder_path, tmp['image_name'])] = generate_labels(file)['timepoint_id'] + ' ' + generate_labels(file)['condition']

    # Create dataset from images and labels
    data = {
        "image": [load_and_resize_image(i, size=image_size) for i in tqdm(data_dict.keys())],
        "label": list(data_dict.values())
    }
    full_dataset = Dataset.from_dict(data)

    # Split the dataset into training and validation sets
    train_test_split = full_dataset.train_test_split(test_size=test_size)

    # Organize the split datasets into a DatasetDict
    dataset = DatasetDict({
        "train": train_test_split["train"],
        "validation": train_test_split["test"]
    })


    # Save the dataset to the specified path
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    # Define parameters
    FOLDER_PATH = '/media/NAS05/xianglu/data/images/'
    SAVE_PATH = "./data/pretain_autophagy_datasets_tif"
    IMAGE_SIZE = (512, 512)
    TEST_SIZE = 0.25

    # Run the main function with specified parameters
    main(FOLDER_PATH, SAVE_PATH, IMAGE_SIZE, TEST_SIZE)
