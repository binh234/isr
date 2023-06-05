import os
import cv2
import gradio as gr
from utils import get_upsampler, get_face_enhancer


def inference(img, task, model_name, scale):
    if scale > 4:
        scale = 4  # avoid too large scale value
    try:
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)

        h, w = img.shape[0:2]
        if h > 3500 or w > 3500:
            raise gr.Error(f"image too large: {w} * {h}")

        if (h < 300 and w < 300) and model_name != "srcnn":
            img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if task == "face":
            upsample_model_name = "realesr-general-x4v3"
        else:
            upsample_model_name = model_name
        upsampler = get_upsampler(upsample_model_name)

        if task == "face":
            face_enhancer = get_face_enhancer(model_name, scale, upsampler)
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
            raise gr.Error(error)

        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output
    except Exception as error:
        raise gr.Error(f"global exception: {error}")


def on_task_change(task):
    if task == "general":
        return gr.Dropdown.update(
            choices=[
                "srcnn",
                "RealESRGAN_x2plus",
                "RealESRGAN_x4plus",
                "RealESRNet_x4plus",
                "realesr-general-x4v3",
            ],
            value="realesr-general-x4v3",
        )
    elif task == "face":
        return gr.Dropdown.update(
            choices=["GFPGANv1.3", "GFPGANv1.4", "RestoreFormer"], value="GFPGANv1.4"
        )
    elif task == "anime":
        return gr.Dropdown.update(
            choices=["srcnn", "RealESRGAN_x4plus_anime_6B", "realesr-animevideov3"],
            value="RealESRGAN_x4plus_anime_6B",
        )


title = "ISR: General Image Super Resolution"

with gr.Blocks(css="style.css", title=title) as demo:
    with gr.Row(elem_classes=["container"]):
        with gr.Column(scale=2):
            input_image = gr.Image(type="filepath", label="Input")
            # with gr.Row():
            task = gr.Dropdown(
                ["general", "face", "anime"],
                type="value",
                value="general",
                label="task",
            )
            model_name = gr.Dropdown(
                [
                    "srcnn",
                    "RealESRGAN_x2plus",
                    "RealESRGAN_x4plus",
                    "RealESRNet_x4plus",
                    "realesr-general-x4v3",
                ],
                type="value",
                value="realesr-general-x4v3",
                label="model",
            )
            scale = gr.Slider(
                minimum=1.5,
                maximum=4,
                value=2,
                step=0.5,
                label="Scale factor",
                info="Scaling factor",
            )
            run_btn = gr.Button(value="Submit")

        with gr.Column(scale=3):
            output_image = gr.Image(type="numpy", label="Output image")

    with gr.Row(elem_classes=["container"]):
        gr.Examples(
            [
                ["examples/landscape.jpg", "general", 2],
                ["examples/cat.jpg", "general", 2],
                ["examples/cat2.jpg", "face", 2],
                ["examples/AI-generate.png", "face", 2],
                ["examples/Blake_Lively.png", "face", 2],
                ["examples/old_image.jpg", "face", 2],
                ["examples/naruto.png", "anime", 2],
                ["examples/luffy2.jpg", "anime", 2],
            ],
            [input_image, task, scale],
        )

    run_btn.click(inference, [input_image, task, model_name, scale], [output_image])
    task.change(on_task_change, [task], [model_name])

demo.queue(concurrency_count=4).launch()
