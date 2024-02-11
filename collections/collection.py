import os
import gradio as gr
import torch
import numpy as np
import pandas as pd
from diffusers import DiffusionPipeline
from transformers import pipeline
from PIL import Image
from tqdm import tqdm
import random
pipe = pipeline('text-generation', model='daspartho/prompt-extend')

def extend_prompt(prompt):
    return pipe(prompt+',', num_return_sequences=1)[0]["generated_text"]

def text_it(inputs):
    return extend_prompt(inputs)

custom_cache_dir = "./.cache/stabilityai/sdxl-turbo"

def load_pipeline(use_cuda):
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.max_memory_allocated(device=device)
        torch.cuda.empty_cache()
        pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir=custom_cache_dir)
        pipe.enable_xformers_memory_efficient_attention()
        pipe = pipe.to(device)
        torch.cuda.empty_cache()
    else:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo", use_safetensors=True, cache_dir=custom_cache_dir)
        pipe = pipe.to(device)
    return pipe

def genie(prompt="sexy woman", steps=2, seed=0, use_cuda=False):
    pipe = load_pipeline(use_cuda)
    generator = np.random.seed(0) if seed == 0 else torch.manual_seed(seed)
    extended_prompt = extend_prompt(prompt)
    int_image = pipe(prompt=extended_prompt, generator=generator, num_inference_steps=steps, guidance_scale=0.0).images[0]
    return int_image, extended_prompt

from tqdm import tqdm

def save_images(prompt="sexy woman", steps=2, use_cuda=False, num_images=1000, folder="collection", input_seed=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    data = []
    columns = ['Image','Prompt', 'Extended_Prompt', 'Steps', 'Seed','Path']
    df = pd.DataFrame(columns=columns)
    df.to_csv('collection.csv', index=False)

    for i in tqdm(range(num_images), desc="Generating Images"):
        seed = random.randint(1, 999999999999999999) if input_seed == 0 else input_seed
        int_image, extended_prompt = genie(prompt=prompt, steps=steps, seed=seed, use_cuda=use_cuda)
        image_filename = f"image_{i+1}.png"
        file_path=f"{folder}/{image_filename}"
        int_image.save(file_path)
        new_data = [image_filename,prompt, extended_prompt, steps, seed,file_path]
        data.append(new_data)
        # Update the DataFrame and save it to the CSV file
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('collection.csv', index=False)

# Call the save_images function to generate and save 1000 images
prompt='''sexy woman,  flowing blonde hair, white dress with a red floral pattern, has red lipstick , serious expression. 
'''
save_images(prompt=prompt, steps=2, use_cuda=False, num_images=1000, folder="collection")
