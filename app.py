import io
from typing import Any

import gradio as gr
from gradio_image_annotation import image_annotator
from PIL import Image
from src.main import Service

service = Service()


def process_prompt(img: Image.Image, prompt: str) -> Image.Image:
    output = service.run(image_dir=img, text_prompt=prompt)
    return Image.fromarray(output)


def on_change_prompt(img: Image.Image, prompt: str):
    return gr.update(interactive=bool(img and prompt))


desc = """
<center>
<h1>Object Eraser by Linh Truong, Hang Luong, and Huy Le</h1>
<p style="font-size: 1.25rem;">

</p>
<p>
Erase any object from your image just by naming it — no manual work required!
</p>
<p>
Pipeline: Delete any object using Pipeline GroundingDino -> SAM -> LAMA
</p>
<p>
GitHub
&nbsp;
<a href="https://github.com/truong-xuan-linh/Object-Removal-by-Prompt" target="_blank">
  <img
    src="https://img.shields.io/github/stars/truong-xuan-linh/Object-Removal-by-Prompt?style=social"
    style="display: inline; vertical-align: middle;"
  />
</a>
</p>
</center>
"""

with gr.Blocks() as demo:
    gr.HTML(desc)
    with gr.Tab("By prompt", id="tab_prompt"):
        with gr.Row():
            with gr.Column():
                iimg = gr.Image(type="pil", label="Input")
                prompt = gr.Textbox(label="What should we erase?")
            with gr.Column():
                oimg = gr.Image(show_label=False, label="Output")
        with gr.Row():
            btn = gr.Button("Erase Object", interactive=False)
        for inp in [iimg, prompt]:
            inp.change(
                fn=on_change_prompt,
                inputs=[iimg, prompt],
                outputs=[btn],
            )
        btn.click(fn=process_prompt, inputs=[iimg, prompt], outputs=[oimg])

        ex = gr.Examples(
            examples=[
                [
                    "examples/white-towels-rattan-basket-white-table-with-bright-room-background.jpg",
                    "soap",
                ],
                [
                    "examples/interior-decor-with-mirror-potted-plant.jpg",
                    "potted plant",
                ],
                [
                    "examples/detail-ball-basketball-court-sunset.jpg",
                    "basketball",
                ],
                [
                    "examples/still-life-device-table_23-2150994394.jpg",
                    "glass of water",
                ],
                [
                    "examples/knife-fork-green-checkered-napkin_140725-63576.jpg",
                    "knife and fork",
                ],
                [
                    "examples/city-night-with-architecture-vibrant-lights_23-2149836930.jpg",
                    "frontmost black car on right lane",
                ],
                [
                    "examples/close-up-coffee-latte-wooden-table_23-2147893063.jpg",
                    "coffee cup on plate",
                ],
                [
                    "examples/empty-chair-with-vase-plant_74190-2078.jpg",
                    "chair",
                ],
            ],
            inputs=[iimg, prompt],
            outputs=[oimg],
            fn=process_prompt,
            cache_examples=True,
        )

demo.launch(share=True)
