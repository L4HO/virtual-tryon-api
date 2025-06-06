import argparse
import os
from datetime import datetime
import io
import logging
from typing import Union, Optional

import numpy as np
import torch
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download

from model.cloth_masker import AutoMasker
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 설정값 ---
CONFIG = {
    "base_model_path": "booksforcharlie/stable-diffusion-inpainting",
    "resume_path": "zhengchong/CatVTON",
    "output_dir_fastapi": "resource/demo/output_fastapi_results",
    "width": 768,
    "height": 1024,
    "allow_tf32": True,
    "mixed_precision": "bf16",
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}

# --- 전역 변수로 모델 저장 ---
pipeline_model: Optional[CatVTONPipeline] = None
automasker_model: Optional[AutoMasker] = None
mask_processor_model: Optional[VaeImageProcessor] = None

app = FastAPI(title="Fashion Fit API")

@app.on_event("startup")
async def load_models_on_startup():
    global pipeline_model, automasker_model, mask_processor_model
    logger.info("애플리케이션 시작: 모델 로딩을 시작합니다...")

    if not os.path.exists(CONFIG["output_dir_fastapi"]):
        os.makedirs(CONFIG["output_dir_fastapi"], exist_ok=True)
        logger.info(f"FastAPI 결과 저장 폴더 생성: {CONFIG['output_dir_fastapi']}")

    try:
        repo_path = snapshot_download(repo_id=CONFIG["resume_path"])
        logger.info(f"리포지토리 다운로드 완료: {repo_path}")

        logger.info("모델 로딩 중...")
        pipeline_model = CatVTONPipeline(
            base_ckpt=CONFIG["base_model_path"], attn_ckpt=repo_path, attn_ckpt_version="mix",
            weight_dtype=init_weight_dtype(CONFIG["mixed_precision"]),
            use_tf32=CONFIG["allow_tf32"], device=CONFIG["device"]
        )
        mask_processor_model = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        automasker_model = AutoMasker(
            densepose_ckpt=os.path.join(repo_path, "DensePose"),
            schp_ckpt=os.path.join(repo_path, "SCHP"), device=CONFIG["device"],
        )
        logger.info("모든 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        logger.error(f"모델 로딩 중 심각한 오류 발생: {e}", exc_info=True)

# --- 핵심 로직을 담은 헬퍼 함수 (리팩토링) ---
def run_single_tryon(
    person_pil: Image.Image,
    cloth_pil: Image.Image,
    cloth_type: str,
    mask_pil: Optional[Image.Image] = None,
    person_pil_for_mask: Optional[Image.Image] = None, # 마스크 생성 시 사용할 별도 이미지 (선택)
    num_inference_steps: int = 50,
    guidance_scale: float = 2.5,
    seed: int = 42
) -> Image.Image:
    """한 번의 가상 착용을 수행하는 핵심 함수"""

    if not pipeline_model or not automasker_model or not mask_processor_model:
        raise RuntimeError("모델이 로드되지 않았습니다.")

    # 이미지 리사이징 및 패딩
    person_pil = resize_and_crop(person_pil, (CONFIG["width"], CONFIG["height"]))
    cloth_pil = resize_and_padding(cloth_pil, (CONFIG["width"], CONFIG["height"]))
    logger.info("입력 이미지 리사이징 및 패딩 완료.")

    # 마스크 처리
    if mask_pil is not None:
        logger.info("사용자 제공 마스크 처리 중...")
        mask_pil = resize_and_crop(mask_pil, (CONFIG["width"], CONFIG["height"]))
    else:
        logger.info("AutoMasker를 사용하여 마스크 생성 중...")
        # 마스크 생성용 이미지가 따로 제공되면 그것을 사용, 아니면 입력된 사람 이미지를 사용
        image_for_masking = person_pil_for_mask if person_pil_for_mask else person_pil
        image_for_masking = resize_and_crop(image_for_masking, (CONFIG["width"], CONFIG["height"]))

        mask_pil = automasker_model(image_for_masking, cloth_type)['mask'] # type: ignore
        logger.info("AutoMasker 마스크 생성 완료.")

    final_mask = mask_processor_model.blur(mask_pil, blur_factor=9) # type: ignore
    logger.info("마스크 블러 처리 완료.")

    # 생성기 초기화
    generator = None
    if seed != -1:
        generator = torch.Generator(device=CONFIG["device"]).manual_seed(seed)

    # 추론 실행
    logger.info(f"추론 시작: cloth_type='{cloth_type}', steps={num_inference_steps}, guidance={guidance_scale}")
    result_image_pil = pipeline_model( # type: ignore
        image=person_pil,
        condition_image=cloth_pil,
        mask=final_mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    )[0]
    logger.info("추론 완료.")
    return result_image_pil

# --- 새로운 엔드포인트: 상하의 모두 적용 ---
@app.post("/virtual-tryon-outfit/")
async def virtual_tryon_outfit_endpoint(
    person_image_file: UploadFile = File(..., description="원본 사람 이미지 파일"),
    upper_garment_file: UploadFile = File(..., description="상의 이미지 파일"),
    lower_garment_file: UploadFile = File(..., description="하의 이미지 파일"),
    num_inference_steps: int = Form(50, ge=10, le=100, description="추론 스텝 수"),
    guidance_scale: float = Form(2.5, ge=0.0, le=7.5, description="Guidance Scale"),
    seed: int = Form(42, description="시드 값 (-1 이면 랜덤)")
):
    logger.info("상/하의 전체 착용 요청 수신.")
    try:
        # 1. 모든 이미지 파일 읽기
        person_bytes = await person_image_file.read()
        upper_bytes = await upper_garment_file.read()
        lower_bytes = await lower_garment_file.read()

        original_person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        upper_pil = Image.open(io.BytesIO(upper_bytes)).convert("RGB")
        lower_pil = Image.open(io.BytesIO(lower_bytes)).convert("RGB")
        logger.info("상/하의 및 원본 사람 이미지 로드 완료.")

        # 2. 상의 적용
        logger.info("--- 1단계: 상의 적용 시작 ---")
        image_with_top_pil = run_single_tryon(
            person_pil=original_person_pil.copy(), # 원본 이미지를 복사하여 사용
            cloth_pil=upper_pil,
            cloth_type="upper",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        logger.info("--- 1단계: 상의 적용 완료 ---")

        # 3. 하의 적용
        logger.info("--- 2단계: 하의 적용 시작 ---")
        final_outfit_pil = run_single_tryon(
            person_pil=image_with_top_pil, # 이전 단계의 결과 이미지를 입력으로 사용
            cloth_pil=lower_pil,
            cloth_type="lower",
            # 중요: 마스크 생성 시에는 '원본 사람 이미지'를 기준으로 사용
            person_pil_for_mask=original_person_pil.copy(),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        logger.info("--- 2단계: 하의 적용 완료 ---")


        # 최종 이미지를 응답으로 전송
        img_byte_arr = io.BytesIO()
        final_outfit_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        logger.info("최종 결과 이미지 전송.")
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"전체 착용 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 서버 내부 오류: {str(e)}")

# --- 기존 엔드포인트: 단일 아이템 적용 (수정됨) ---
@app.post("/virtual-tryon/")
async def virtual_tryon_endpoint(
    person_image_file: UploadFile = File(..., description="사람 이미지 파일"),
    cloth_image_file: UploadFile = File(..., description="옷 이미지 파일"),
    person_mask_file: Optional[UploadFile] = File(None, description="사람 마스크 파일 (선택)"),
    cloth_type: str = Form("upper", enum=["upper", "lower", "overall"], description="옷 종류"),
    num_inference_steps: int = Form(50, ge=10, le=100, description="추론 스텝 수"),
    guidance_scale: float = Form(2.5, ge=0.0, le=7.5, description="Guidance Scale"),
    seed: int = Form(42, description="시드 값 (-1 이면 랜덤)")
):
    logger.info(f"단일 착용 요청 수신 (cloth_type: {cloth_type}).")
    try:
        person_bytes = await person_image_file.read()
        cloth_bytes = await cloth_image_file.read()
        person_pil = Image.open(io.BytesIO(person_bytes)).convert("RGB")
        cloth_pil = Image.open(io.BytesIO(cloth_bytes)).convert("RGB")

        mask_pil = None
        if person_mask_file and person_mask_file.filename:
            mask_bytes = await person_mask_file.read()
            if mask_bytes: mask_pil = Image.open(io.BytesIO(mask_bytes)).convert("L")

        # 핵심 로직 함수 호출
        result_image_pil = run_single_tryon(
            person_pil=person_pil, cloth_pil=cloth_pil, cloth_type=cloth_type, mask_pil=mask_pil,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, seed=seed
        )

        img_byte_arr = io.BytesIO()
        result_image_pil.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return StreamingResponse(img_byte_arr, media_type="image/png")

    except Exception as e:
        logger.error(f"단일 착용 처리 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 서버 내부 오류: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Fashion Fit API에 오신 것을 환영합니다. /docs 에서 API 문서를 확인하세요."}