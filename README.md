# Repository for "Pic2Prep: A multimodal conversational  agent for cooking assistance" Paper
Pic2Prep is a multimodal AI system that generates detailed cooking instructions, ingredient lists, and cooking actions from food images and text inputs, offering an interactive and personalized cooking experience
<br>
Demo Link: https://drive.google.com/file/d/1V0ev_8zfygh0pjWERhMLlj-2aU-YR01n/view?usp=drive_link
<br>

![Pic2Prep User Interface](https://github.com/renjithk4/Pic2Prep/blob/main/image.png "Pic2Prep User Interface")

This repository contains curated dataset, implementation of methods experimented and introduced in the paper titled "Pic2Prep: A multimodal conversational  agent for cooking assistance".

# 1. Data Generation #
For image generation run, -> py 1. Data Generation/image_generation_recipe_.py 
The datasets are generated using Stable Diffusion for ingredients and instruction generation.
<br>
The generated image dataset available at: https://drive.google.com/drive/folders/1hSIuYzthg7idGKAQbc5WnC-OpdhVX8GH
<br>
Dataset used to train BLIP model for Ingredient Generation : https://drive.google.com/drive/folders/1XOcfV1jg6LoQ-XGuadVzC212X1Luy5Cy
<br>
Dataset used to train BLIP model for Instruction Generation : https://drive.google.com/drive/folders/1Zl96GohDDvc_PZEn04kRavXqdjm25oKj

# 2. Inference #
This folder is including the codes for Inference.
The fine-tuned BLIP models for ingredient and instruction generation(''model_blip_instructions'') and ingredient generation(''model_blip_ingredients'') are available here: https://drive.google.com/drive/folders/12NbE6VkDeXBjNaIKi1nPDcbWjuxHzGSZ


# 3. Mapper #
Several other research works are under progress are that uses Mistral Mapper. The model will be open sourced up on acceptance of other papers as it is a novel component.

# 4. Pic2Prep User Interface #
This folder includes the codes for the user interface and steps to execute it.


# 5. Acknowledgements #
We appreciate the following GitHub repos for their valuable code bases:

https://github.com/revathyramanan/cooking-action-generation

https://github.com/CompVis/stable-diffusion