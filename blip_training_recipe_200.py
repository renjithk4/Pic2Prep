import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from PIL import Image
from datasets import Dataset
import os
import shutil
import json

# Load the BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the annotations
with open('/work/kaippilr/food_ingredient_detection/dataset_for_blip_recipe_200/annotations.json', 'r') as f:
    annotations = json.load(f)

# Prepare the dataset dictionary
data = {
    "image_path": [],
    "ingredients": []
}

# Convert annotations to dataset format
for item in annotations:
    image_path = os.path.join('/work/kaippilr/food_ingredient_detection/dataset_for_blip_recipe_200/images', item['image'])
    ingredients = item['ingredients']
    data["image_path"].append(image_path)
    data["ingredients"].append(ingredients)

# Convert to HuggingFace Dataset
dataset = Dataset.from_dict(data)

def tokenize_function(examples):
    # Load and preprocess images
    images = [Image.open(image_path).convert("RGB") for image_path in examples['image_path']]

    # Process images and text using BLIP processor
    #inputs = processor(images=images, text=examples['ingredients'], return_tensors='pt', padding=True)


    inputs = processor(
    images=images, 
    text=examples['ingredients'], 
    return_tensors='pt', 
    padding="max_length",  # Pad sequences to max length
    truncation=True,       # Truncate sequences that are longer than max_length
    max_length=64)         # Set a reasonable max length (adjust as needed)

    # Setting labels for the decoder, which is the tokenized ingredient text
    inputs['labels'] = inputs.input_ids.clone()

    return inputs

# Apply the tokenize function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
tokenized_dataset = tokenized_dataset.remove_columns(["image_path", "ingredients"])

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the trained model and processor
model_save_path = "/work/kaippilr/food_ingredient_detection/model_blip_recipe_200"
model.save_pretrained(model_save_path)
processor.save_pretrained(model_save_path)
