import argparse
import os
# Windows 환경에서는 아래 두 줄을 주석 처리하는 것이 일반적입니다.
# os.environ['CUDA_HOME'] = '/usr/local/cuda'
# os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'
from datetime import datetime

import gradio as gr
# import spaces # <--- 이 라인이 있다면 반드시 주석 처리하거나 삭제해야 합니다!
import numpy as np
import torch

# spaces 모듈이 없을 경우를 위한 더미(dummy) 클래스 정의
try:
    import spaces
except ImportError:
    print("로컬 환경에서 실행 중: 'spaces' 모듈을 위한 더미(dummy) 객체를 사용합니다.")
    class DummySpaces:
        def __init__(self):
            pass
        def GPU(self, *args, **kwargs): # @spaces.GPU 데코레이터를 위한 더미 메서드
            def decorator(func):
                return func
            return decorator
    spaces = DummySpaces() # spaces 라는 이름으로 더미 객체 할당

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image
torch.jit.script = lambda f: f # type: ignore
from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path", type=str, default="booksforcharlie/stable-diffusion-inpainting",
        help="The path to the base model to use for evaluation.")
    parser.add_argument(
        "--resume_path", type=str, default="zhengchong/CatVTON",
        help="The Path to the checkpoint of trained tryon model.")
    parser.add_argument(
        "--output_dir", type=str, default="resource/demo/output",
        help="The output directory where the model predictions will be written.")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--allow_tf32", action="store_true", default=True)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    args, unknown = parser.parse_known_args([])
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if hasattr(args, 'local_rank') and env_local_rank != -1 and args.local_rank == -1 :
        args.local_rank = env_local_rank
    elif not hasattr(args, 'local_rank'):
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
# 아래 모델 로딩 부분은 Gradio UI 테스트 중에는 시간이 오래 걸릴 수 있으므로,
# 필요하다면 임시로 주석 처리하고 테스트할 수 있습니다.
# 하지만 UI 초기화 오류는 이 부분과 직접적인 관련이 없을 가능성이 높습니다.
# print("모델 로딩 시작...")
# repo_path = snapshot_download(repo_id=args.resume_path)
# pipeline = CatVTONPipeline(
#     base_ckpt=args.base_model_path,
#     attn_ckpt=repo_path,
#     attn_ckpt_version="mix",
#     weight_dtype=init_weight_dtype(args.mixed_precision),
#     use_tf32=args.allow_tf32,
#     device='cuda'
# )
# mask_processor = VaeImageProcessor(
#     vae_scale_factor=8,
#     do_normalize=False,
#     do_binarize=True,
#     do_convert_grayscale=True
# )
# automasker = AutoMasker(
#     densepose_ckpt=os.path.join(repo_path, "DensePose"),
#     schp_ckpt=os.path.join(repo_path, "SCHP"),
#     device='cuda',
# )
# print("모델 로딩 완료.")

# 실제 submit_function은 일단 그대로 둡니다. (Minimal UI에서는 호출되지 않음)
@spaces.GPU(duration=120) # type: ignore
def submit_function(
    person_image_input, cloth_image_path, cloth_type,
    num_inference_steps, guidance_scale, seed, show_type
):
    # ... (기존 submit_function 내용) ...
    # 이 함수는 아래 minimal_app_gradio에서는 직접 사용되지 않습니다.
    print("실제 submit_function 호출됨 (이 메시지는 minimal 테스트에서는 보이지 않아야 함)")
    person_image_filepath = person_image_input["background"]
    mask_filepath = person_image_input["layers"][0] if person_image_input["layers"] else None

    if mask_filepath:
        mask = Image.open(mask_filepath).convert("L")
        if np.all(np.array(mask) == 0): mask = None
        else:
            mask_array = np.array(mask); mask_array[mask_array > 0] = 255; mask = Image.fromarray(mask_array)
    else: mask = None

    tmp_folder = args.output_dir
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    result_save_path = os.path.join(tmp_folder, date_str[:8], date_str[8:] + ".png")
    if not os.path.exists(os.path.join(tmp_folder, date_str[:8])): os.makedirs(os.path.join(tmp_folder, date_str[:8]))

    generator = None
    if seed != -1: generator = torch.Generator(device='cuda').manual_seed(int(seed))

    person_image = Image.open(person_image_filepath).convert("RGB")
    cloth_image = Image.open(cloth_image_path).convert("RGB")
    person_image = resize_and_crop(person_image, (args.width, args.height))
    cloth_image = resize_and_padding(cloth_image, (args.width, args.height))

    if mask is not None: mask = resize_and_crop(mask, (args.width, args.height))
    else:
        print("마스크가 제공되지 않았거나 유효하지 않아 AutoMasker를 사용합니다.")
        mask = automasker(person_image,cloth_type)['mask'] # type: ignore
    mask = mask_processor.blur(mask, blur_factor=9) # type: ignore

    # result_image = pipeline( # type: ignore
    #     image=person_image, condition_image=cloth_image, mask=mask,
    #     num_inference_steps=int(num_inference_steps), guidance_scale=float(guidance_scale), generator=generator
    # )[0]
    # 임시로 반환값 설정 (pipeline 호출 주석 처리 시)
    result_image = Image.new("RGB", (args.width, args.height), color="gray")


    masked_person = vis_mask(person_image, mask) # type: ignore
    save_result_image = image_grid([person_image, masked_person, cloth_image, result_image], 1, 4)
    save_result_image.save(result_save_path)

    if show_type == "result only": return result_image
    else:
        display_width, display_height = person_image.size
        if show_type == "input & result": condition_width = display_width // 2; conditions = image_grid([person_image, cloth_image], 2, 1)
        else: condition_width = display_width // 3; conditions = image_grid([person_image, masked_person, cloth_image], 3, 1)
        conditions = conditions.resize((condition_width, display_height), Image.Resampling.NEAREST)
        new_result_image = Image.new("RGB", (display_width + condition_width + 5, display_height))
        new_result_image.paste(conditions, (0, 0)); new_result_image.paste(result_image, (condition_width + 5, 0))
        return new_result_image

def person_example_fn(image_path_from_example):
    return image_path_from_example

# Custom CSS
css = """ footer {visibility: hidden} """ # CSS 간소화 (테스트 목적)

# --- 매우 단순화된 app_gradio 함수 ---
def app_gradio():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"), css=css) as demo: # type: ignore
        gr.Markdown("# 👔 Fashion Fit (Minimal Test)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Test Input")
                input_text = gr.Textbox(label="Test Input Textbox") # 아주 간단한 컴포넌트
                test_button = gr.Button("Minimal Test Button") # type: ignore
            with gr.Column():
                gr.Markdown("### Test Output")
                output_text = gr.Textbox(label="Test Output Textbox")

        def minimal_test_func(text):
            print(f"Minimal test function called with: {text}")
            return f"Processed: {text}"

        test_button.click(
            minimal_test_func,
            inputs=[input_text],
            outputs=[output_text]
        )

    demo.queue().launch(share=True, show_error=True)
# --- 여기까지 매우 단순화된 app_gradio 함수 ---

if __name__ == "__main__":
    # 모델 로딩을 app_gradio 호출 전에 할지, 혹은 Gradio 내부에서 할지 결정할 수 있습니다.
    # UI 초기화 오류 디버깅 중이므로, 일단은 app_gradio()만 호출합니다.
    app_gradio()