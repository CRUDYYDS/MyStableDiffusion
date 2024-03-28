import numpy as np
import torch
import PIL
from PIL import Image

#载入图片
def load_image(path):
    image = Image.open(path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0   #(512, 512, 3)
    image = image[None].transpose(0, 3, 1, 2)           # (1, 3, 512, 512)
    image = torch.from_numpy(image)
    return 2.*image - 1.


#保存图片
def save_image(samples, path):
    samples = 255 * (samples/2+0.5).clamp(0,1)    # (1, 3, 512, 512)
    samples = samples.detach().numpy()
    samples = samples.transpose(0, 2, 3, 1)       #(1, 512, 512, 3)
    image = samples[0]                            #(512, 512, 3)
    image = Image.fromarray(image.astype(np.uint8))
    image.save(path)