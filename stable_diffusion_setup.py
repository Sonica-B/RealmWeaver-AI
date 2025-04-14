import torch
import os
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import logging

def setup_stable_diffusion(model_id="runwayml/stable-diffusion-v1-5"):
    """Set up Stable Diffusion pipeline"""
    logger = logging.getLogger(__name__)
    logger.info(f"Setting up Stable Diffusion with model: {model_id}")
    
    # Determine device and precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load model
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None  # For efficiency, can be re-enabled
        )
        pipe = pipe.to(device)
        
        # Use efficient scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable memory optimization if on GPU
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        logger.info("Stable Diffusion pipeline set up successfully")
        return pipe
    
    except Exception as e:
        logger.error(f"Error setting up Stable Diffusion: {e}")
        raise

def generate_game_asset(pipe, prompt, output_path=None, negative_prompt="", 
                       num_images=1, guidance_scale=7.5, steps=30):
    """Generate a game asset with Stable Diffusion"""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating asset with prompt: {prompt}")
    
    try:
        # Generate image
        outputs = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )
        
        images = outputs.images
        
        # Save images if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if num_images == 1:
                images[0].save(output_path)
                logger.info(f"Saved image to {output_path}")
            else:
                base_path = os.path.splitext(output_path)[0]
                ext = os.path.splitext(output_path)[1]
                
                for i, image in enumerate(images):
                    img_path = f"{base_path}_{i+1}{ext}"
                    image.save(img_path)
                    logger.info(f"Saved image to {img_path}")
        
        return images
    
    except Exception as e:
        logger.error(f"Error generating asset: {e}")
        raise