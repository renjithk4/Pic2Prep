# Repository for "Pic2Prep: A multimodal conversational  agent for cooking assistance" Paper
This repository contains curated dataset, implementation of methods experimented and introduced in the paper titled "Pic2Prep: A multimodal conversational  agent for cooking assistance".

# 1. Data Generation #
This folder includes the steps for image generation.

The generated image dataset available at: https://drive.google.com/drive/folders/1VdIsSouurlVAFRLaZnPuemsDXLyKRN-2?usp=drive_link
To train the model for image generation run, -> py 1. Data Generation/image_generation_recipe_.py


# 2. Inference #
This folder is including the codes for Inference.


# 3. Mapper #
This folder includes the codes for the custom ingredient-to-action mapper, trained using  Mistral model. It is fine-tuned such that ingredients are mapped to its corresponding cooking actions. 


# 4. Demo #
This folder includes the codes for the user interface.


# 5. Acknowledgements #
We appreciate the following github repos for their valuable code bases:

https://github.com/revathyramanan/cooking-action-generation