import argparse
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'
from datetime import datetime

import gradio as gr
import spaces
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
torch.jit.script = lambda f: f
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

@spaces.GPU(duration=120)
def submit_function(
    person_image,
    cloth_image,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):
    person_image, mask = person_image["background"], person_image["layers"][0]
    mask = Image.open(mask).convert("L")
    if len(np.unique(np.array(mask))) == 1:
        mask = None
    else:
        mask = np.array(mask)
        mask[mask > 0] = 255
        mask = Image.fromarray(mask)

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])):
        os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1:
        generator = torch.Generator(device='cuda').manual_seed(seed)

    person_image = Image.open(person_image).convert("RGB")
    cloth_image = Image.open(cloth_image).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))
    
    # Process mask
    if mask is not None:
        mask = resize_and_crop(mask, (args.width, args.height))
    else:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
    mask = mask_processor.blur(mask, blur_factor=9)

    # Inference
    result_image = pipeline(
        image=person_image,
        condition_image=cloth_image,
        mask=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    
    # Post-process
    masked_person = vis_mask(person_image, mask)
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)
    if show_type == "result only":
        return result_image
    else:
        width, height = person_image.size
        if show_type == "input & result":
            condition_width = width // 2
            conditions = image_grid([person_image, cloth_image], 2, 1)
        else:
            condition_width = width // 3
            conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, height), Image.NEAREST)
        new_result_image = Image.new("RGB", (width + condition_width + 5, height))
        new_result_image.paste(conditions, (0, 0))
        new_result_image.paste(result_image, (condition_width + 5, 0))
    return new_result_image

def person_example_fn(image_path):
    return image_path

# Custom CSS for enhanced visual appeal
css = """
footer {visibility: hidden}

/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
}

/* Header styling */
h1, h2, h3 {
    color: #2c3e50;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

/* Button styling */
button.primary-button {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    border: none;
    border-radius: 10px;
    color: white;
    padding: 12px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

button.primary-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
}

/* Image container styling */
.image-container {
    border-radius: 15px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s ease;
}

.image-container:hover {
    transform: scale(1.02);
}

/* Radio button styling */
.radio-group label {
    background-color: #ffffff;
    border-radius: 8px;
    padding: 10px 15px;
    margin: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}

.radio-group input:checked + label {
    background-color: #4CAF50;
    color: white;
}

/* Slider styling */
.slider-container {
    background: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.slider {
    height: 8px;
    border-radius: 4px;
    background: #e0e0e0;
}

.slider .thumb {
    width: 20px;
    height: 20px;
    background: #4CAF50;
    border-radius: 50%;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

/* Alert/warning text styling */
.warning-text {
    color: #ff5252;
    font-weight: bold;
    text-align: center;
    padding: 10px;
    background: rgba(255,82,82,0.1);
    border-radius: 8px;
    margin: 10px 0;
}

/* Example gallery styling */
.example-gallery {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    padding: 15px;
    background: white;
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.example-item {
    border-radius: 8px;
    overflow: hidden;
    transition: transform 0.3s ease;
}

.example-item:hover {
    transform: scale(1.05);
}
"""

def app_gradio():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"), css=css) as demo:
        gr.Markdown(
            """
            # Virtual Try-On App 👔
            Transform your look with AI-powered virtual clothing try-on!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                with gr.Box():
                    gr.Markdown("### 📸 Upload Images")
                    with gr.Row():
                        image_path = gr.Image(
                            type="filepath",
                            interactive=True,
                            visible=False,
                        )
                        person_image = gr.ImageEditor(
                            interactive=True,
                            label="Person Image",
                            type="filepath",
                            elem_classes="image-container"
                        )

                    with gr.Row():
                        with gr.Column(scale=1, min_width=230):
                            cloth_image = gr.Image(
                                interactive=True,
                                label="Clothing Item",
                                type="filepath",
                                elem_classes="image-container"
                            )
                        with gr.Column(scale=1, min_width=120):
                            gr.Markdown(
                                """
                                ### 🎯 Masking Options
                                1. Draw mask manually with 🖌️
                                2. Auto-generate based on clothing type
                                """
                            )
                            cloth_type = gr.Radio(
                                label="Clothing Type",
                                choices=["upper", "lower", "overall"],
                                value="upper",
                                elem_classes="radio-group"
                            )

                submit = gr.Button("🚀 Generate Try-On", elem_classes="primary-button")
                gr.Markdown(
                    """
                    <div class="warning-text">
                        ⚠️ Please click only once and wait patiently for processing
                    </div>
                    """
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    num_inference_steps = gr.Slider(
                        label="Quality Level",
                        minimum=10,
                        maximum=100,
                        step=5,
                        value=50,
                        elem_classes="slider-container"
                    )
                    guidance_scale = gr.Slider(
                        label="Style Strength",
                        minimum=0.0,
                        maximum=7.5,
                        step=0.5,
                        value=2.5,
                        elem_classes="slider-container"
                    )
                    seed = gr.Slider(
                        label="Random Seed",
                        minimum=-1,
                        maximum=10000,
                        step=1,
                        value=42,
                        elem_classes="slider-container"
                    )
                    show_type = gr.Radio(
                        label="Display Mode",
                        choices=["result only", "input & result", "input & mask & result"],
                        value="input & mask & result",
                        elem_classes="radio-group"
                    )
            with gr.Column(scale=2, min_width=500):
                result_image = gr.Image(
                    interactive=False,
                    label="Final Result",
                    elem_classes="image result_image = gr.Image(
                        interactive=False,
                        label="Final Result",
                        elem_classes="image-container"
                    )

                    with gr.Row():
                        # Photo Examples
                        root_path = "resource/demo/example"
                        with gr.Column():
                            gr.Markdown("#### 👤 Model Examples")
                            men_exm = gr.Examples(
                                examples=[
                                    os.path.join(root_path, "person", "men", _)
                                    for _ in os.listdir(os.path.join(root_path, "person", "men"))
                                ],
                                examples_per_page=4,
                                inputs=image_path,
                                label="Men's Examples",
                                elem_classes="example-item"
                            )
                            women_exm = gr.Examples(
                                examples=[
                                    os.path.join(root_path, "person", "women", _)
                                    for _ in os.listdir(os.path.join(root_path, "person", "women"))
                                ],
                                examples_per_page=4,
                                inputs=image_path,
                                label="Women's Examples",
                                elem_classes="example-item"
                            )
                            gr.Markdown(
                                '<div class="info-text">Model examples courtesy of <a href="https://huggingface.co/spaces/levihsu/OOTDiffusion">OOTDiffusion</a> and <a href="https://www.outfitanyone.org">OutfitAnyone</a></div>'
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### 👕 Clothing Examples")
                            condition_upper_exm = gr.Examples(
                                examples=[
                                    os.path.join(root_path, "condition", "upper", _)
                                    for _ in os.listdir(os.path.join(root_path, "condition", "upper"))
                                ],
                                examples_per_page=4,
                                inputs=cloth_image,
                                label="Upper Garments",
                                elem_classes="example-item"
                            )
                            condition_overall_exm = gr.Examples(
                                examples=[
                                    os.path.join(root_path, "condition", "overall", _)
                                    for _ in os.listdir(os.path.join(root_path, "condition", "overall"))
                                ],
                                examples_per_page=4,
                                inputs=cloth_image,
                                label="Full Outfits",
                                elem_classes="example-item"
                            )
                            condition_person_exm = gr.Examples(
                                examples=[
                                    os.path.join(root_path, "condition", "person", _)
                                    for _ in os.listdir(os.path.join(root_path, "condition", "person"))
                                ],
                                examples_per_page=4,
                                inputs=cloth_image,
                                label="Reference Styles",
                                elem_classes="example-item"
                            )
                            gr.Markdown(
                                '<div class="info-text">Clothing examples sourced from various online retailers</div>'
                            )

            image_path.change(
                person_example_fn,
                inputs=image_path,
                outputs=person_image
            )

            submit.click(
                submit_function,
                [
                    person_image,
                    cloth_image,
                    cloth_type,
                    num_inference_steps,
                    guidance_scale,
                    seed,
                    show_type,
                ],
                result_image,
            )
            
        gr.Markdown(
            """
            ### 💡 Tips & Instructions
            1. Upload or select a person image
            2. Choose or upload a clothing item
            3. Select clothing type (upper/lower/overall)
            4. Adjust advanced settings if needed
            5. Click Generate and wait for results
            
            For best results, use clear, front-facing images with good lighting.
            """
        )
        
    demo.queue().launch(share=True, show_error=True)

if __name__ == "__main__":
    app_gradio()