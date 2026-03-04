# import torch
# from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
# from PIL import Image, ImageDraw, ImageFont
# import os

# # 1. Setup Environment
# device = "cpu"
# model_id = "runwayml/stable-diffusion-v1-5"

# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     model_id, 
#     torch_dtype=torch.float32
# )
# # Using a faster scheduler for CPU performance
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe.to(device)

# # 2. Load and Preprocess
# init_image = Image.open("image2.jpg").convert("RGB").resize((512, 512))

# # 3. Define Benchmarking Parameters
# # Strength: 0.0 (Original) to 1.0 (Complete noise)
# strengths = [0.3, 0.45, 0.6] 
# prompt = "same person, preserve face, muscular athletic physique, gym transformation, realistic, high detail"

# print(f"Running benchmark on {device}...")

# results = []
# for s in strengths:
#     print(f"Generating for strength: {s}...")
    
#     # Generate
#     image = pipe(
#         prompt=prompt,
#         image=init_image,
#         strength=s,
#         guidance_scale=8.0,
#         num_inference_steps=20 # Reduced for CPU speed during testing
#     ).images[0]
    
#     # Add label to image for the interview presentation
#     draw = ImageDraw.Draw(image)
#     draw.text((10, 10), f"Strength: {s}", fill="white")
#     results.append(image)

# # 4. Create a Comparison Grid
# grid_width = len(results) * 512
# grid = Image.new('RGB', (grid_width, 512))

# for i, img in enumerate(results):
#     grid.paste(img, (i * 512, 0))

# grid.save("transformation_benchmark.png")
# print("Benchmark grid saved as 'transformation_benchmark.png'!")
import torch
import time
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from PIL import Image, ImageDraw
import os

# --- 1. RESEARCH-GRADE SETUP ---
device = "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# Load with Safety Checker Disabled (to avoid the 'Black Image' bug)
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

# Optimization: Using DPM-Solver++ for faster CPU convergence
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(device)

# --- 2. CONFIGURATION & LOGGING ---
init_image = Image.open("image2.jpg").convert("RGB").resize((512, 512))
strengths = [0.3, 0.45, 0.6] 
prompt = "person in gym wear, muscular athletic physique, highly detailed, realistic skin, fit body"
negative_prompt = "deformed, distorted, cartoon, low quality, shirtless, nsfw" # Helps steer away from safety flags

results = []
log_data = []

print(f"Starting Benchmarking on {device}...")

# --- 3. EXECUTION LOOP ---
for s in strengths:
    start_time = time.time()
    
    # Inference call
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        strength=s,
        guidance_scale=8.0,
        num_inference_steps=20
    )
    
    image = output.images[0]
    latency = time.time() - start_time
    
    # Scientific Overlay: Labeling the image with Strength and Latency
    draw = ImageDraw.Draw(image)
    label = f"Strength: {s}\nLatency: {latency:.2f}s"
    draw.text((20, 20), label, fill="white")
    
    results.append(image)
    log_data.append({"strength": s, "latency": latency})
    print(f"Finished Strength {s} in {latency:.2f}s")

# --- 4. GRID GENERATION ---
grid_width = len(results) * 512
grid = Image.new('RGB', (grid_width, 520), color=(30, 30, 30)) # Extra height for bottom footer

for i, img in enumerate(results):
    grid.paste(img, (i * 512, 0))

grid.save("fitness_benchmark_results.png")
print("\nSuccess! Benchmarking grid saved.")

# --- 5. PERFORMANCE SUMMARY ---
avg_latency = sum(item['latency'] for item in log_data) / len(log_data)
print(f"Average Inference Time (CPU): {avg_latency:.2f} seconds")