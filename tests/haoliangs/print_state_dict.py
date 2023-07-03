from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

pipe.to("cuda")
for param_tensor in pipe.unet.state_dict():
    print(param_tensor, "\t", pipe.unet.state_dict()[param_tensor].size())