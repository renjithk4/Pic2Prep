# Repository for "Pic2Prep: A multimodal conversational  agent for cooking assistance" Paper
Pic2Prep is a multimodal AI system that generates detailed cooking instructions, ingredient lists, and cooking actions from food images and text inputs, offering an interactive and personalized cooking experience

![Pic2Prep User Interface](https://github.com/renjithk4/Pic2Prep/blob/main/image.png "Pic2Prep User Interface")

This repository contains curated dataset, implementation of methods experimented and introduced in the paper titled "Pic2Prep: A multimodal conversational  agent for cooking assistance".

# 1. Data Generation #
For image generation run, -> py 1. Data Generation/image_generation_recipe_.py 
The datasets are generated using Stable Diffusion for ingredients and instruction generation.
<br>
The generated image dataset available at: https://drive.google.com/drive/folders/1BvQHLX4oP6kGucFkLPMINlTApuComKyx
<br>
Dataset used to train BLIP model for Ingredient Generation : https://drive.google.com/drive/folders/12LsSxNyps3Y1V4Ijo4bUbiZo_4ABVwfI?usp=drive_link
<br>
Dataset used to train BLIP model for Instruction Generation :https://drive.google.com/drive/folders/13s6zZWb7PVNmUjrAvNzAzrFzz2nNz2Ft?usp=drive_link

# 2. Inference #
This folder is including the codes for Inference.
The fine-tuned BLIP models for ingredient and instruction generation(''model_blip_instructions'') and ingredient generation(''model_blip_ingredients'') are available here: https://drive.google.com/drive/folders/1QhzWJjiqty8VdWuO1_NypniFmtwsqHoN 


# 3. Mapper #
This folder includes the codes for the custom ingredient-to-action mapper, trained using  Mistral model. It is fine-tuned such that ingredients are mapped to its corresponding cooking actions. 
This fine-tuned model available at: https://drive.google.com/drive/folders/17LIhanMEzivm78BvZy2hUMVyK6GGNZlA


# 4. Pic2Prep User Interface #
This folder includes the codes for the user interface and steps to execute it.


# 5. Acknowledgements #
We appreciate the following GitHub repos for their valuable code bases:

https://github.com/revathyramanan/cooking-action-generation

https://github.com/CompVis/stable-diffusion