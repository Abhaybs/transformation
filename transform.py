import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
#StableDiffusionControlNetPipeline
# Load image
init_image = Image.open("image2.jpg").convert("RGB")
init_image = init_image.resize((512, 512))

# Load CPU pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
)

pipe = pipe.to("cpu")

# Prompt
print("Select goal:")
print("1 - Muscle Gain")
print("2 - Fat Loss")

choice = input("Enter choice: ")


if choice == "1":

    prompt = "same person, preserve face, muscular athletic physique, gym transformation, realistic, high detail"

elif choice == "2":

    prompt = "same person, preserve face, lean slim athletic body, fat loss transformation, realistic, high detail"

else:

    print("Invalid choice")
    exit()

# Generate image
result = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.35,
    guidance_scale=8.5,
)

# Save
result.images[0].save("output_transformation1.png")

print("Transformation complete!")