from data_preparation.training_data_preparation import train_data_preparation
from model_training.lora_training_ViT import train_model
from unsupervised_compound_selection.image_representation_generation import unsupervised_image_representation_generation


def main():
    # Define parameters
    FOLDER_PATH = '/media/NAS05/xianglu/data/images/'
    SAVE_PATH = "./data/pretrain_autophagy_datasets_tif"
    OUTPUT_DIR = "./models/vit-large-autophagy-lora"
    EXPERIMENTAL_DATA_FOLDER = './data/experimental_images/'
    PRETRAIN_MODEL_NAME = 'google/vit-large-patch16-224'
    
    IMAGE_SIZE = (512, 512)
    TEST_SIZE = 0.25
    
    try:
        # Run dataset preparation
        dataset = train_data_preparation(FOLDER_PATH, SAVE_PATH, IMAGE_SIZE, TEST_SIZE)
        print("Dataset preparation completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        raise
    
    # Train model
    train_model(
        pretrain_model_name=PRETRAIN_MODEL_NAME,
        dataset_saved_path=SAVE_PATH,
        output_dir=OUTPUT_DIR,
        batch_size=32,
        learning_rate=5e-4,
        num_train_epochs=20,
        cuda_device="3",
        lora_r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        gradient_accumulation_steps=4,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
        warmup_steps=100,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
        seed=42,
    )
    
    # Generate image representations
    results = unsupervised_image_representation_generation(
        pretrain_model_name=PRETRAIN_MODEL_NAME,
        model_name=OUTPUT_DIR,
        experimental_data_folder=EXPERIMENTAL_DATA_FOLDER
    )


if __name__ == "__main__":
    main()