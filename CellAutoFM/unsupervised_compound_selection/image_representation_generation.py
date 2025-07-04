#!/usr/bin/env python3
"""
Image Embedding Generation Script
Converts experimental images to embeddings using CellAutoFM.
"""

import os
import json
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor
from peft import PeftModel
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


class AutoModelForImageEmbedding(nn.Module):
    """Custom wrapper for image embedding generation with normalization."""
    
    def __init__(self, model_name, normalize=True):
        super(AutoModelForImageEmbedding, self).__init__()
        
        # Load a pre-trained image classification model (e.g., a Vision Transformer or similar)
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize

    def forward(self, images):
        # Forward pass through the model
        model_output = self.model(images)
        pooler_output = model_output['pooler_output']
        
        if self.normalize:
            pooler_output = torch.nn.functional.normalize(pooler_output, p=2, dim=1)

        return pooler_output

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


def load_model(pretrain_model_name, model_name):
    """Load the base model and apply PEFT configuration."""
    print("Loading base model...")
    model = AutoModelForImageEmbedding(pretrain_model_name)
    
    print("Loading PEFT model...")
    model = PeftModel.from_pretrained(model, model_name)
    
    print("Model loaded successfully!")
    return model


def setup_image_processor(model_name):
    """Setup image processor and transforms."""
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    
    val_transforms = Compose([
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ])
    
    return image_processor, val_transforms


def get_image_paths(experimental_data_folder):
    """Get all image file paths from the experimental data folder."""
    file_path_list = []
    file_path_list2 = []
    file_path_list3 = []
    
    for x in os.listdir(experimental_data_folder):
        if 'D' not in x:
            file_path_list.append(os.path.join(experimental_data_folder, x, f'{x}_1.tif'))
            file_path_list2.append(os.path.join(experimental_data_folder, x, f'{x}_2.tif'))
            file_path_list3.append(os.path.join(experimental_data_folder, x, f'{x}_Merge.tif'))
    
    return file_path_list, file_path_list2, file_path_list3


def generate_embeddings(model, file_paths, val_transforms, output_file):
    """Generate embeddings for a list of image files."""
    embedding_dict = {}
    
    print(f"Processing {len(file_paths)} images...")
    for file_path in tqdm(file_paths):
        try:
            image = Image.open(file_path)
            encoding = val_transforms(image.convert("RGB"))
            
            with torch.no_grad():
                outputs = model(encoding.unsqueeze(0))
            
            # Extract the sample name from the file path
            sample_name = file_path.split('/')[-1].split('_')[0]
            embedding_dict[sample_name] = outputs.numpy().flatten().tolist()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save embeddings to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(embedding_dict, json_file)
    
    print(f"Embeddings saved to {output_file}")
    return embedding_dict


def unsupervised_image_representation_generation(pretrain_model_name, model_name, experimental_data_folder):
    """
    Generate unsupervised image representations using a fine-tuned Vision Transformer model.
    
    Args:
        pretrain_model_name (str): Name of the pre-trained model to use as base
        model_name (str): Path to the fine-tuned PEFT model
        experimental_data_folder (str): Path to the experimental data folder containing images
    
    Returns:
        dict: Dictionary containing the generated embeddings for each image type
    """
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load model and setup processor
    model = load_model(pretrain_model_name, model_name)
    image_processor, val_transforms = setup_image_processor(model_name)
    
    # Get image paths
    file_path_list, file_path_list2, file_path_list3 = get_image_paths(experimental_data_folder)
    
    # Generate embeddings for each image type
    print("\nGenerating embeddings for _1.tif images...")
    embeddings_1 = generate_embeddings(model, file_path_list, val_transforms, 'data/experiment_embedding_data_tif.json')
    
    print("\nGenerating embeddings for _2.tif images...")
    embeddings_2 = generate_embeddings(model, file_path_list2, val_transforms, 'data/experiment_embedding_data_tif2.json')
    
    print("\nGenerating embeddings for _Merge.tif images...")
    embeddings_merge = generate_embeddings(model, file_path_list3, val_transforms, 'data/experiment_embedding_data_tif3.json')
    
    print("\nAll embeddings generated successfully!")
    
    return {
        'embeddings_1': embeddings_1,
        'embeddings_2': embeddings_2,
        'embeddings_merge': embeddings_merge
    }

if __name__ == "__main__":
    # Default configuration
    pretrain_model_name = 'google/vit-large-patch16-224'
    model_name = 'model_tif_3/google/vit-large-patch16-224'
    experimental_data_folder = 'data/experimental_data/'
    
    # Run unsupervised image representation generation
    results = unsupervised_image_representation_generation(
        pretrain_model_name=pretrain_model_name,
        model_name=model_name,
        experimental_data_folder=experimental_data_folder
    )