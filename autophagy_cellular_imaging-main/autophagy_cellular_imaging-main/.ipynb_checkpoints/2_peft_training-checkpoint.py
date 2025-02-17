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

def main(pretrain_model_name, dataset_saved_path, batch_size, learning_rate, num_train_epochs, cuda_device, peft_model_id):
    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    # Load dataset in streaming mode
    ds = load_dataset(dataset_saved_path)

    # Create ClassLabel and update dataset labels
    unique_labels = ds['train'].unique('label')
    class_label = ClassLabel(names=unique_labels)
    ds = ds.cast_column('label', class_label)

    # Create label mappings
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}

    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(pretrain_model_name)

    # Define data transformations
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    train_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    val_transforms = Compose(
        [
            Resize(image_processor.size["height"]),
            CenterCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    # Set dataset transformations
    train_ds = ds["train"]
    val_ds = ds["validation"]
    train_ds.set_transform(lambda batch: preprocess(batch, train_transforms))
    val_ds.set_transform(lambda batch: preprocess(batch, val_transforms))

    # Load pre-trained model
    model = AutoModelForImageClassification.from_pretrained(
        pretrain_model_name,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    # Apply PEFT (Parameter-Efficient Fine-Tuning)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Define training arguments
    args = TrainingArguments(
        output_dir=peft_model_id,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=batch_size,
        fp16=True,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        label_names=["labels"],
    )

    # Load evaluation metric
    metric = evaluate.load("accuracy")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Initialize and train the model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=image_processor,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metric),
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(peft_model_id)

def preprocess(example_batch, transforms):
    """
    Applies data transformations to a batch of examples.
    Args:
        example_batch (dict): Batch of examples containing images.
        transforms (Compose): Composed transformations for preprocessing the images.
    Returns:
        dict: Batch of examples with transformed images.
    """
    example_batch["pixel_values"] = [transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def collate_fn(examples):
    """
    Collates a batch of examples into a batch for model training or evaluation.
    Args:
        examples (list): List of examples containing pixel values and labels.
    Returns:
        dict: Dictionary containing stacked pixel values and labels.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred, metric):
    """
    Computes accuracy on a batch of predictions.
    Args:
        eval_pred (EvalPrediction): Evaluation predictions.
        metric (Metric): Metric to compute accuracy.
    Returns:
        dict: Computed accuracy.
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


if __name__ == "__main__":
    main(
        pretrain_model_name='google/vit-large-patch16-224',
        dataset_saved_path="./data/pretain_autophagy_datasets_tif",
        batch_size=128,
        learning_rate=5e-3,
        num_train_epochs=20,
        cuda_device="3",
        peft_model_id=f"model_tif_3/google/vit-large-patch16-224"
    )
