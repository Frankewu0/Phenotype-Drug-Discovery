import os
import torch
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from peft import LoraConfig, get_peft_model
import evaluate
import logging
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(
    pretrain_model_name: str,
    dataset_saved_path: str,
    output_dir: str,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    num_train_epochs: int = 10,
    cuda_device: Optional[str] = None,
    lora_r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 10,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    dataloader_num_workers: int = 4,
    seed: int = 42,
) -> None:
    """
    Train an image classification model using PEFT (LoRA) fine-tuning.
    
    Args:
        pretrain_model_name: Name of the pre-trained model to use
        dataset_saved_path: Path to the saved dataset
        output_dir: Directory to save the trained model
        batch_size: Training batch size
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        cuda_device: CUDA device to use (e.g., "0", "1", etc.)
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        gradient_accumulation_steps: Number of gradient accumulation steps
        fp16: Whether to use mixed precision training
        save_steps: Number of steps between model saves (only used with 'steps' strategy)
        eval_steps: Number of steps between evaluations (only used with 'steps' strategy)
        logging_steps: Number of steps between logging
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        dataloader_num_workers: Number of workers for data loading
        seed: Random seed for reproducibility
    """
    try:
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Set CUDA device if specified
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
            logger.info(f"Using CUDA device: {cuda_device}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Load dataset
        logger.info(f"Loading dataset from: {dataset_saved_path}")
        ds = load_dataset(dataset_saved_path)
        
        # Validate dataset structure
        if 'train' not in ds:
            raise ValueError("Dataset must contain a 'train' split")
        
        # Create ClassLabel and update dataset labels
        unique_labels = ds['train'].unique('label')
        if len(unique_labels) < 2:
            raise ValueError("Dataset must contain at least 2 unique labels")
        
        logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
        
        class_label = ClassLabel(names=unique_labels)
        ds = ds.cast_column('label', class_label)
        
        # Create label mappings
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for i, label in enumerate(unique_labels)}
        
        # Load image processor
        logger.info(f"Loading image processor for: {pretrain_model_name}")
        image_processor = AutoImageProcessor.from_pretrained(pretrain_model_name)
        
        # Define data transformations
        normalize = Normalize(
            mean=image_processor.image_mean, 
            std=image_processor.image_std
        )
        
        train_transforms = Compose([
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        
        val_transforms = Compose([
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ])
        
        # Prepare datasets
        train_ds = ds["train"]
        val_ds = ds.get("validation", ds.get("test"))
        
        if val_ds is None:
            logger.warning("No validation/test set found. Using train set for validation.")
            val_ds = train_ds
        
        # Set dataset transformations
        train_ds.set_transform(lambda batch: preprocess(batch, train_transforms))
        val_ds.set_transform(lambda batch: preprocess(batch, val_transforms))
        
        logger.info(f"Training samples: {len(train_ds)}")
        logger.info(f"Validation samples: {len(val_ds)}")
        
        # Load pre-trained model
        logger.info(f"Loading pre-trained model: {pretrain_model_name}")
        model = AutoModelForImageClassification.from_pretrained(
            pretrain_model_name,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        )
        
        # Apply PEFT (Parameter-Efficient Fine-Tuning)
        logger.info("Applying LoRA configuration...")
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        
        # Define training arguments
        # Ensure evaluation and save strategies match for load_best_model_at_end
        strategy = "epoch" if num_train_epochs <= 10 else "steps"
        
        args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            eval_strategy=strategy,
            eval_steps=eval_steps if strategy == "steps" else None,
            save_strategy=strategy,
            save_steps=save_steps if strategy == "steps" else None,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=fp16,
            num_train_epochs=num_train_epochs,
            logging_steps=logging_steps,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            label_names=["labels"],
            dataloader_num_workers=dataloader_num_workers,
            report_to=None,  # Disable wandb/tensorboard logging by default
            seed=seed,
        )
        
        # Load evaluation metric
        metric = evaluate.load("accuracy")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=image_processor,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric),
            data_collator=collate_fn,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info(f"Saving final model to: {output_dir}")
        trainer.save_model(output_dir)
        
        # Evaluate the model
        logger.info("Evaluating final model...")
        eval_results = trainer.evaluate()
        logger.info(f"Final evaluation results: {eval_results}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


def preprocess(example_batch: Dict[str, Any], transforms: Compose) -> Dict[str, Any]:
    """
    Applies data transformations to a batch of examples.
    
    Args:
        example_batch: Batch of examples containing images
        transforms: Composed transformations for preprocessing the images
        
    Returns:
        Batch of examples with transformed images
    """
    try:
        example_batch["pixel_values"] = [
            transforms(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise


def collate_fn(examples: list) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of examples into a batch for model training or evaluation.
    
    Args:
        examples: List of examples containing pixel values and labels
        
    Returns:
        Dictionary containing stacked pixel values and labels
    """
    try:
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    except Exception as e:
        logger.error(f"Error in collate_fn: {str(e)}")
        raise


def compute_metrics(eval_pred, metric) -> Dict[str, float]:
    """
    Computes accuracy on a batch of predictions.
    
    Args:
        eval_pred: Evaluation predictions
        metric: Metric to compute accuracy
        
    Returns:
        Dictionary containing computed accuracy
    """
    try:
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)
    except Exception as e:
        logger.error(f"Error in compute_metrics: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage with improved parameters
    train_model(
        pretrain_model_name='google/vit-large-patch16-224',
        dataset_saved_path="./data/pretain_autophagy_datasets_tif",
        output_dir="./models/vit-large-autophagy-lora",
        batch_size=32,  # Reduced for better stability
        learning_rate=5e-4,  # Reduced learning rate
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