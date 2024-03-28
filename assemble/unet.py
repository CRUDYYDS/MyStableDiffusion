
#加载unet模型
import torch

from ldm.modules.diffusionmodules.openaimodel import UNetModel


def load_unet():
    unet_init_config = {
            "image_size": 32, # unused
            "in_channels": 4,
            "out_channels": 4,
            "model_channels": 320,
            "attention_resolutions": [ 4, 2, 1 ],
            "num_res_blocks": 2,
            "channel_mult": [ 1, 2, 4, 4 ],
            "num_heads": 8,
            "use_spatial_transformer": True,
            "transformer_depth": 1,
            "context_dim": 768,
            "use_checkpoint": True,
            "legacy": False,
    }
    unet = UNetModel(**unet_init_config)
    pl_sd = torch.load("../models/download_models/sd-v1-4_2.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]

    model_dict = unet.state_dict()
    for k, v in model_dict.items():
        model_dict[k] = sd["model.diffusion_model."+k]

    unet.load_state_dict(model_dict, strict=False)
    unet.cuda()
    unet.eval()
    return unet

def test_unet():
    #vae
    latent = torch.randn(1,4,64,64).cuda()
    #text
    text_embeddings =torch.randn(1, 77, 768).cuda()
    #timestamp
    timestamp = torch.tensor([0]).cuda()
    unet = load_unet()
    y = unet(latent.cuda(), timestamp.cuda(), text_embeddings.cuda())
    print(y.shape) #(1, 4, 64, 64)

test_unet()