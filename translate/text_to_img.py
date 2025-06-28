# conda create -n TestLLM python=3.12
# conda activate TestLLM
# pip install diffusers transformers accelerate hf_xet
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# python text_to_img.py

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

def text_to_img(text, out_img_filename):
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(text).images[0]

    # PIL image
    #print(type(image))

    image.save(out_img_filename)


text='''
赤毛の胴体に黒い瞳を持つ五匹の猿の姿をした悪魔がベッド枠によじ登り、尾をぱたぱた揺らす幻視を見た
'''

text_to_img(text, "out.jpg")
