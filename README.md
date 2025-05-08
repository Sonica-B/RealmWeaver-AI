# RealmWeaver-AI - Procedural Game Asset Generator

**RealmWeaver-AI** is a procedural 2D game asset generation framework that combines the power of **generative AI** and **Unity game development**. The system automates the creation of game-ready assets such as terrain tiles, environment props, and complete biome maps, helping indie developers reduce the time and effort needed for building immersive game worlds.

---

## 🔧 Core Components

- **Stable Diffusion**  
  Generates top-down terrain tiles and props from text prompts conditioned on biome and style.

- **CVAE-GAN (Conditional Variational Autoencoder GAN)**  
  Learns structural features from tiles to generate diverse and coherent tilemaps.

- **Wave Function Collapse Algorithm**  
  Helps generate procedurally varied but spatially consistent biome maps.

- **Unity Integration**  
  Automates prefab generation, reads biome CSVs, and renders maps using scripted tile placement.

---

## 🧠 Code Structure Summary

| File | Description |
|------|-------------|
| `Asset_Generator_cgans.py` | Implements Conditional GAN to generate stylized game assets based on class labels. |
| `Asset_Generator_cvaegans_copy2.py` | Uses a CVAE-GAN hybrid model for improved diversity in generated assets. |
| `generate_images_cvaegan.py` | Loads the trained CVAE-GAN model to generate batches of asset images. |
| `stable_diffusion_setup.py` | Initializes Stable Diffusion from HuggingFace's `diffusers` library. |
| `generate_terrain.py` | Text-to-image generation for biome-specific terrain using Stable Diffusion. |
| `generate_assets.py` | Wraps asset generation logic using prebuilt prompts. |
| `generate_map.py` | Renders full biome maps in Unity using CSV inputs and prefab IDs. |
| `lora_setup.py` | Sets up Low-Rank Adaptation (LoRA) weights for fine-tuning Stable Diffusion. |
| `map_rules.py` | Implements logic to convert biome values into terrain rules and constraints. |
| `wave_function_collapse.py` | Procedural map layout using Wave Function Collapse algorithm. |
| `vae.py` | Core VAE model class with encoder, decoder, reparam sampling. |
| `train_vae.py` | Trains a basic VAE on game environment datasets. |
| `train_environment_vae.py` | Environment-specific variant of `train_vae.py`. |
| `training_pipeline.py` | End-to-end training orchestration pipeline for asset models. |
| `environment_setup.py` | Sets up necessary folder structure, logging, and device config. |
| `visualize.py` | Asset and training output visualization utilities. |
| `day3_4_implementation.py` | Early prototype or experimental integration script. |

---

## 🛠️ How to Use

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate terrain tiles
```bash
python stable_diffusion_setup.py
python generate_terrain.py
```

### 3. Train CVAE-GAN model
```bash
python train_vae.py
python Asset_Generator_cvaegans_copy2.py
```

### 4. Generate tile maps
```bash
python generate_images_cvaegan.py
```

### 5. Export Unity biome maps
Prepare a biome CSV grid and run:
```bash
python generate_map.py
```

---

## 🔮 Future Work
- Fine-tune Stable Diffusion on specific art styles
- Add support for animated sprites or 3D assets
- Extend to conditional or transformer-based image generation
- Use Wave Function Collapse more tightly with gameplay logic
