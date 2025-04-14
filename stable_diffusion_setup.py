#!/usr/bin/env python
# stable_diffusion_setup.py - Set up Stable Diffusion for game asset generation

import torch
import os
import logging
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

def setup_logging():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("stable_diffusion.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_stable_diffusion(model_id="runwayml/stable-diffusion-v1-5"):
    """Set up Stable Diffusion pipeline"""
    logger = setup_logging()
    logger.info(f"Setting up Stable Diffusion with model: {model_id}")
    
    # Determine device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load model with appropriate settings
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None  # For efficiency, can be re-enabled
        )
        pipe = pipe.to(device)
        
        # Optimize for inference
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        # Use efficient scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        logger.info(f"Successfully set up Stable Diffusion on {device}")
        return pipe
    
    except Exception as e:
        logger.error(f"Error setting up Stable Diffusion: {e}")
        raise

def generate_asset(pipe, prompt, negative_prompt="", output_path=None, num_inference_steps=30, guidance_scale=7.5):
    """Generate an image asset"""
    logger = setup_logging()
    logger.info(f"Generating asset with prompt: {prompt}")
    
    try:
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Save image if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            logger.info(f"Saved asset to {output_path}")
        
        return image
    
    except Exception as e:
        logger.error(f"Error generating asset: {e}")
        raise

def load_lora_adapter(pipe, lora_path, adapter_name="game_assets"):
    """Load LoRA weights into pipeline"""
    logger = setup_logging()
    logger.info(f"Loading LoRA adapter from {lora_path}")
    
    try:
        from peft import PeftModel
        
        # Load LoRA weights
        pipe.unet = PeftModel.from_pretrained(
            pipe.unet, 
            lora_path, 
            adapter_name=adapter_name
        )
        
        # Check if text encoder weights are available
        if os.path.exists(os.path.join(lora_path, "text_encoder")):
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder,
                lora_path,
                adapter_name=adapter_name
            )
        
        logger.info(f"Successfully loaded LoRA adapter: {adapter_name}")
        return pipe
    
    except Exception as e:
        logger.error(f"Error loading LoRA adapter: {e}")
        logger.warning("Continuing with base model")
        return pipe

if __name__ == "__main__":
    # Set up Stable Diffusion
    pipe = setup_stable_diffusion()
    
    # Test with a simple image generation
    test_prompt = "A beautiful fantasy landscape with mountains and forests, game asset, digital art"
    test_image = generate_asset(
        pipe,
        prompt=test_prompt,
        output_path="outputs/test_sd.png"
    )
    
    print("Stable Diffusion setup and test complete!")