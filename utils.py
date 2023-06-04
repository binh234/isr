import os
import torch
from basicsr.utils.download_util import load_file_from_url
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

from config import *
from srcnn import SRCNN


def get_upsampler(model_name, device=None):
    if model_name == "RealESRGAN_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        ]
    elif model_name == "RealESRNet_x4plus":  # x4 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
        ]
    elif model_name == "RealESRGAN_x4plus_anime_6B":  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        ]
    elif model_name == "RealESRGAN_x2plus":  # x2 RRDBNet model
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        ]
    elif model_name == "realesr-animevideov3":  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
        ]
    elif model_name == "realesr-general-x4v3":  # x4 VGG-style model (S size)
        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        netscale = 4
        file_url = [
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        ]
    elif model_name == "srcnn":
        model = SRCNN(device=device)
        model_path = os.path.join(ROOT_DIR, WEIGHT_DIR, model_name + ".pth")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        if device:
            model.to(device)
        return model
    else:
        raise ValueError(f"Wrong model version {model_name}.")

    model_path = os.path.join(ROOT_DIR, WEIGHT_DIR, model_name + ".pth")
    if not os.path.exists(model_path):
        print(f"Downloading weights for model {model_name}")

        for url in file_url:
            # model_path will be updated
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(ROOT_DIR, WEIGHT_DIR),
                progress=True,
                file_name=None,
            )

    if model_name != "realesr-general-x4v3":
        dni_weight = None
    else:
        dni_weight = [0.5, 0.5]
        wdn_model_path = model_path.replace(
            "realesr-general-x4v3", "realesr-general-wdn-x4v3"
        )
        model_path = [model_path, wdn_model_path]

    half = "cuda" in str(device)

    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        half=half,
        device=device,
    )


def get_face_enhancer(model_name, upscale=2, bg_upsampler=None, device=None):
    if model_name == "GFPGANv1.3":
        arch = "clean"
        channel_multiplier = 2
        file_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"
    elif model_name == "GFPGANv1.4":
        arch = "clean"
        channel_multiplier = 2
        file_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    elif model_name == "RestoreFormer":
        arch = "RestoreFormer"
        channel_multiplier = 2
        file_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
    else:
        raise ValueError(f"Wrong model version {model_name}.")

    model_path = os.path.join(ROOT_DIR, WEIGHT_DIR, model_name + ".pth")
    if not os.path.exists(model_path):
        print(f"Downloading weights for model {model_name}")
        model_path = load_file_from_url(
            url=file_url,
            model_dir=os.path.join(ROOT_DIR, WEIGHT_DIR),
            progress=True,
            file_name=None,
        )

    return GFPGANer(
        model_path=model_path,
        upscale=upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler,
        device=device,
    )
