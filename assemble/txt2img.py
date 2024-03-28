from assemble.unet import  load_unet
from assemble.lms_scheduler import lms_scheduler
from assemble.clip import prompts_embedding
from assemble.vae import load_vae
from assemble.img_util import save_image
import torch


def txt2img():
    # unet
    unet = load_unet()
    # 调度器
    scheduler = lms_scheduler()
    scheduler.set_timesteps(100)
    # 文本编码
    prompts = ["a photograph of an astronaut riding a horse"]
    text_embeddings = prompts_embedding(prompts)
    text_embeddings = text_embeddings.cuda()  # (1, 77, 768)
    uncond_prompts = [""]
    uncond_embeddings = prompts_embedding(uncond_prompts)
    uncond_embeddings = uncond_embeddings.cuda()  # (1, 77, 768)
    # 初始隐变量
    latents = torch.randn((1, 4, 64, 64))  # (1, 4, 64, 64)
    latents = latents * scheduler.sigmas[0]  # sigmas[0]=157.40723
    latents = latents.cuda()
    # 循环步骤
    for i, t in enumerate(scheduler.timesteps):  # timesteps=[999.  988.90909091 978.81818182 ...100个
        latent_model_input = latents  # (1, 4, 64, 64)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma ** 2 + 1) ** 0.5)
        timestamp = torch.tensor([t]).cuda()

        with torch.no_grad():
            noise_pred_text = unet(latent_model_input, timestamp, text_embeddings)
            noise_pred_uncond = unet(latent_model_input, timestamp, uncond_embeddings)
            guidance_scale = 7.5
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, i, latents)

    vae = load_vae()
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents.cpu())  # (1, 3, 512, 512)
    save_image(image, "txt2img.png")


txt2img()