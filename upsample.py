import argparse
import cv2
import os

from imutils import paths
from tqdm import tqdm
from config import *
from utils import get_face_enhancer, get_upsampler


def process(image_path, upsampler_name, face_enhancer_name=None, scale=2, device="cpu"):
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            output = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return output

        if (h < 300 and w < 300) and upsampler_name != "srcnn":
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            return img

        upsampler = get_upsampler(upsampler_name, device=device)

        if face_enhancer_name:
            face_enhancer = get_face_enhancer(
                face_enhancer_name, scale, upsampler, device=device
            )
        else:
            face_enhancer = None

        try:
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True
                )
            else:
                output, _ = upsampler.enhance(img, outscale=scale)
        except RuntimeError as error:
            print(f"Runtime error: {error}")

        return output
    except Exception as error:
        print(f"global exception: {error}")


def main(args: argparse.Namespace) -> None:
    device = args.device
    scale = args.scale

    upsampler_name = args.upsampler
    face_enhancer_name = args.face_enhancer

    if face_enhancer_name and ("srcnn" in upsampler_name or "anime" in upsampler_name):
        print(
            "Warnings: SRCNN and Anime model aren't compatible with face enhance. We will turn it off for you"
        )
        face_enhancer_name = None

    os.makedirs(args.output, exist_ok=True)
    if not os.path.exists(args.input):
        raise ValueError("The input directory doesn't exist!")
    elif not os.path.isdir(args.input):
        image_paths = [args.input]
    else:
        image_paths = paths.list_images(args.input)

    with tqdm(image_paths) as pbar:
        for image_path in pbar:
            filename = os.path.basename(image_path)
            pbar.set_postfix_str(f"Processing {image_path}")
            upsampled_image = process(
                image_path=image_path,
                upsampler_name=upsampler_name,
                face_enhancer_name=face_enhancer_name,
                scale=scale,
                device=device,
            )
            if upsampled_image is not None:
                save_path = os.path.join(args.output, filename)
                cv2.imwrite(save_path, upsampled_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs automatic detection and mask generation on an input image or directory of images"
        )
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to the output directory.",
    )

    parser.add_argument(
        "--upsampler",
        type=str,
        default="realesr-general-x4v3",
        choices=[
            "srcnn",
            "RealESRGAN_x2plus",
            "RealESRGAN_x4plus",
            "RealESRNet_x4plus",
            "realesr-general-x4v3",
            "RealESRGAN_x4plus_anime_6B",
            "realesr-animevideov3",
        ],
        help="The type of upsampler model to load",
    )

    parser.add_argument(
        "--face-enhancer",
        type=str,
        choices=["GFPGANv1.3", "GFPGANv1.4", "RestoreFormer"],
        help="The type of face enhancer model to load",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=2,
        choices=[1.5, 2, 2.5, 3, 3.5, 4],
        help="scaling factor",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to run upsampling on."
    )
    args = parser.parse_args()
    main(args)
