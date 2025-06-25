# #!/usr/bin/env python
# # stable_diffusion_setup.py - Set up Stable Diffusion for Dragon Hills asset generation

# import torch
# import os
# import logging
# from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# from PIL import Image

# def setup_logging():
#     """Set up logging"""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler("stable_diffusion.log"),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# def setup_stable_diffusion(model_id="runwayml/stable-diffusion-v1-5"):
#     """Set up Stable Diffusion pipeline with Dragon Hills optimizations"""
#     logger = setup_logging()
#     logger.info(f"Setting up Stable Diffusion with model: {model_id}")
    
#     # Determine device and precision
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
#     # Load model with appropriate settings
#     try:
#         pipe = StableDiffusionPipeline.from_pretrained(
#             model_id,
#             torch_dtype=dtype,
#             safety_checker=None  # For efficiency, can be re-enabled
#         )
#         pipe = pipe.to(device)
        
#         # Optimize for inference
#         if device == "cuda":
#             pipe.enable_attention_slicing()
        
#         # Use efficient scheduler
#         pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
#         logger.info(f"Successfully set up Stable Diffusion on {device}")
#         return pipe
    
#     except Exception as e:
#         logger.error(f"Error setting up Stable Diffusion: {e}")
#         raise

# def generate_dragon_hills_asset(pipe, prompt, negative_prompt="", output_path=None, 
#                                asset_type="generic", theme="fantasy", 
#                                num_inference_steps=30, guidance_scale=7.5):
#     """Generate an image asset in Dragon Hills style"""
#     logger = setup_logging()
    
#     # Add Dragon Hills style to prompts
#     dragon_hills_style = "Dragon Hills game style, cartoon style, vibrant colors, strong silhouettes, clear edges"
    
#     # Theme-specific style additions
#     theme_styles = {
#         "fantasy": "medieval cartoon style, grassy landscapes, colorful castles",
#         "zombie": "post-apocalyptic cartoon style, gritty but vibrant colors, destroyed buildings",
#         "wild_west": "western cartoon style, desert colors, dusty atmosphere, old west buildings",
#         "space": "sci-fi cartoon style, neon colors, futuristic technology, alien landscapes"
#     }
    
#     # Asset type specific additions
#     asset_styles = {
#         "character": "full body character sprite, clear outline, game asset",
#         "enemy": "enemy sprite, menacing but cartoon style, game asset",
#         "terrain": "tileable terrain texture, top-down view, game asset",
#         "object": "destructible game object, clear silhouette, game asset",
#         "power_up": "collectible power-up icon, glowing effect, game asset",
#         "coin": "shiny coin collectible, sparkle effect, game asset",
#         "vehicle": "vehicle sprite, cartoon style, game asset"
#     }
    
#     # Combine prompt with styles
#     final_prompt = f"{prompt}, {dragon_hills_style}, {theme_styles.get(theme, '')}, {asset_styles.get(asset_type, '')}"
    
#     # Default negative prompt for Dragon Hills style
#     default_negative = "realistic, photorealistic, blurry, low contrast, desaturated, dark, muddy colors, complex details"
#     final_negative = f"{negative_prompt}, {default_negative}" if negative_prompt else default_negative
    
#     logger.info(f"Generating Dragon Hills asset with prompt: {final_prompt}")
    
#     try:
#         # Generate image
#         image = pipe(
#             prompt=final_prompt,
#             negative_prompt=final_negative,
#             num_inference_steps=num_inference_steps,
#             guidance_scale=guidance_scale
#         ).images[0]
        
#         # Save image if output_path is provided
#         if output_path:
#             os.makedirs(os.path.dirname(output_path), exist_ok=True)
#             image.save(output_path)
#             logger.info(f"Saved asset to {output_path}")
        
#         return image
    
#     except Exception as e:
#         logger.error(f"Error generating asset: {e}")
#         raise

# def load_lora_adapter(pipe, lora_path, adapter_name="dragon_hills"):
#     """Load LoRA weights for Dragon Hills style"""
#     logger = setup_logging()
#     logger.info(f"Loading Dragon Hills LoRA adapter from {lora_path}")
    
#     try:
#         from peft import PeftModel
        
#         # Load LoRA weights
#         pipe.unet = PeftModel.from_pretrained(
#             pipe.unet, 
#             lora_path, 
#             adapter_name=adapter_name
#         )
        
#         # Check if text encoder weights are available
#         if os.path.exists(os.path.join(lora_path, "text_encoder")):
#             pipe.text_encoder = PeftModel.from_pretrained(
#                 pipe.text_encoder,
#                 lora_path,
#                 adapter_name=adapter_name
#             )
        
#         logger.info(f"Successfully loaded Dragon Hills LoRA adapter: {adapter_name}")
#         return pipe
    
#     except Exception as e:
#         logger.error(f"Error loading LoRA adapter: {e}")
#         logger.warning("Continuing with base model")
#         return pipe

# def generate_dragon_hills_character(pipe, character_type="princess", pose="riding", 
#                                   theme="fantasy", output_path=None):
#     """Generate character assets specifically for Dragon Hills"""
    
#     character_prompts = {
#         "princess": {
#             "riding": "angry princess riding dragon, cartoon style, determined expression",
#             "attacking": "princess shooting gun while riding dragon, cartoon style",
#             "celebrating": "princess celebrating victory on dragon, cartoon style"
#         },
#         "dragon": {
#             "idle": "red dragon character, menacing but cute, cartoon style",
#             "diving": "dragon diving underground, dynamic pose, cartoon style",
#             "emerging": "dragon bursting from ground, explosive pose, cartoon style",
#             "flying": "dragon flying, wings spread, cartoon style"
#         },
#         "knight": {
#             "idle": "cartoon knight enemy, medieval armor, game sprite",
#             "attacking": "knight attacking with sword, cartoon style",
#             "fleeing": "knight running away scared, cartoon style"
#         },
#         "zombie": {
#             "idle": "cartoon zombie enemy, groaning pose, game sprite",
#             "shambling": "zombie shambling forward, cartoon style",
#             "reaching": "zombie reaching with arms, cartoon style"
#         }
#     }
    
#     if character_type not in character_prompts or pose not in character_prompts[character_type]:
#         raise ValueError(f"Unknown character type {character_type} or pose {pose}")
    
#     prompt = character_prompts[character_type][pose]
    
#     return generate_dragon_hills_asset(
#         pipe, 
#         prompt,
#         asset_type="character",
#         theme=theme,
#         output_path=output_path
#     )

# if __name__ == "__main__":
#     # Set up Stable Diffusion
#     pipe = setup_stable_diffusion()
    
#     # Test with Dragon Hills style generation
#     test_assets = [
#         ("princess riding dragon", "character", "fantasy"),
#         ("destructible medieval house", "object", "fantasy"),
#         ("cartoon zombie enemy", "enemy", "zombie"),
#         ("gold coin collectible", "coin", "fantasy"),
#         ("shield power-up", "power_up", "fantasy")
#     ]
    
#     for prompt, asset_type, theme in test_assets:
#         image = generate_dragon_hills_asset(
#             pipe,
#             prompt=prompt,
#             asset_type=asset_type,
#             theme=theme,
#             output_path=f"outputs/test_{asset_type}_{theme}.png"
#         )
    
#     print("Dragon Hills style asset generation test complete!")

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
    """Set up Stable Diffusion pipeline with Dragon Hills optimizations"""
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

def generate_dragon_hills_asset(pipe, prompt, negative_prompt="", output_path=None, 
                               asset_type="generic", theme="fantasy", 
                               num_inference_steps=30, guidance_scale=7.5):
    """Generate an image asset in Dragon Hills style"""
    logger = setup_logging()
    
    # Add Dragon Hills style to prompts
    dragon_hills_style = "Dragon Hills game style, cartoon style, vibrant colors, strong silhouettes, clear edges"
    
    # Theme-specific style additions
    theme_styles = {
        "fantasy": "medieval cartoon style, grassy landscapes, colorful castles",
        "zombie": "post-apocalyptic cartoon style, gritty but vibrant colors, destroyed buildings",
        "wild_west": "western cartoon style, desert colors, dusty atmosphere, old west buildings",
        "space": "sci-fi cartoon style, neon colors, futuristic technology, alien landscapes"
    }
    
    # Asset type specific additions
    asset_styles = {
        "character": "full body character sprite, clear outline, game asset",
        "enemy": "enemy sprite, menacing but cartoon style, game asset",
        "terrain": "tileable terrain texture, top-down view, game asset",
        "object": "destructible game object, clear silhouette, game asset",
        "power_up": "collectible power-up icon, glowing effect, game asset",
        "coin": "shiny coin collectible, sparkle effect, game asset",
        "vehicle": "vehicle sprite, cartoon style, game asset"
    }
    
    # Combine prompt with styles
    final_prompt = f"{prompt}, {dragon_hills_style}, {theme_styles.get(theme, '')}, {asset_styles.get(asset_type, '')}"
    
    # Default negative prompt for Dragon Hills style
    default_negative = "realistic, photorealistic, blurry, low contrast, desaturated, dark, muddy colors, complex details"
    final_negative = f"{negative_prompt}, {default_negative}" if negative_prompt else default_negative
    
    logger.info(f"Generating Dragon Hills asset with prompt: {final_prompt}")
    
    try:
        # Generate image
        result = pipe(
            prompt=final_prompt,
            negative_prompt=final_negative,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Check if images exist and are not empty
        if not hasattr(result, 'images') or len(result.images) == 0:
            logger.error(f"No images generated for prompt: {final_prompt}")
            return None
            
        image = result.images[0]
        
        # Save image if output_path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            logger.info(f"Saved asset to {output_path}")
        
        return image
    
    except Exception as e:
        logger.error(f"Error generating asset: {e}")
        return None  # Return None instead of raising to avoid crashing

def load_lora_adapter(pipe, lora_path, adapter_name="dragon_hills"):
    """Load LoRA weights for Dragon Hills style"""
    logger = setup_logging()
    logger.info(f"Loading Dragon Hills LoRA adapter from {lora_path}")
    
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
        
        logger.info(f"Successfully loaded Dragon Hills LoRA adapter: {adapter_name}")
        return pipe
    
    except Exception as e:
        logger.error(f"Error loading LoRA adapter: {e}")
        logger.warning("Continuing with base model")
        return pipe

def generate_dragon_hills_character(pipe, character_type="princess", pose="riding", 
                                  theme="fantasy", output_path=None):
    """Generate character assets specifically for Dragon Hills"""
    
    character_prompts = {
        "princess": {
            "riding": "angry princess riding dragon, cartoon style, determined expression",
            "attacking": "princess shooting gun while riding dragon, cartoon style",
            "celebrating": "princess celebrating victory on dragon, cartoon style"
        },
        "dragon": {
            "idle": "red dragon character, menacing but cute, cartoon style",
            "diving": "dragon diving underground, dynamic pose, cartoon style",
            "emerging": "dragon bursting from ground, explosive pose, cartoon style",
            "flying": "dragon flying, wings spread, cartoon style"
        },
        "knight": {
            "idle": "cartoon knight enemy, medieval armor, game sprite",
            "attacking": "knight attacking with sword, cartoon style",
            "fleeing": "knight running away scared, cartoon style"
        },
        "zombie": {
            "idle": "cartoon zombie enemy, groaning pose, game sprite",
            "shambling": "zombie shambling forward, cartoon style",
            "reaching": "zombie reaching with arms, cartoon style"
        }
    }
    
    if character_type not in character_prompts or pose not in character_prompts[character_type]:
        raise ValueError(f"Unknown character type {character_type} or pose {pose}")
    
    prompt = character_prompts[character_type][pose]
    
    return generate_dragon_hills_asset(
        pipe, 
        prompt,
        asset_type="character",
        theme=theme,
        output_path=output_path
    )

if __name__ == "__main__":
    # Set up Stable Diffusion
    pipe = setup_stable_diffusion()
    
    # Test with Dragon Hills style generation
    test_assets = [
        ("princess riding dragon", "character", "fantasy"),
        ("destructible medieval house", "object", "fantasy"),
        ("cartoon zombie enemy", "enemy", "zombie"),
        ("gold coin collectible", "coin", "fantasy"),
        ("shield power-up", "power_up", "fantasy")
    ]
    
    for prompt, asset_type, theme in test_assets:
        image = generate_dragon_hills_asset(
            pipe,
            prompt=prompt,
            asset_type=asset_type,
            theme=theme,
            output_path=f"outputs/test_{asset_type}_{theme}.png"
        )
        
        if image is None:
            print(f"Failed to generate asset for {prompt}")
        else:
            print(f"Successfully generated {asset_type} asset")
    
    print("Dragon Hills style asset generation test complete!")