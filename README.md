# Fashion Fit API (패션 핏 API 서버)

`zhengchong/CatVTON` 모델을 기반으로 한 가상 의류 착용(Virtual Try-On) 기능을 제공하는 FastAPI 서버입니다. 사용자는 사람 이미지와 상의, 하의 이미지를 업로드하여 전체 옷을 착용한 이미지를 생성할 수 있습니다.

This is a FastAPI server that provides a virtual try-on feature based on the `zhengchong/CatVTON` model. Users can upload an image of a person, along with top and bottom garments, to generate a new image with the full outfit applied.

## 주요 기능 (Features)

  * **단일 의류 착용**: 상의, 하의, 또는 원피스 등 단일 아이템을 가상으로 입혀볼 수 있습니다. (`/virtual-tryon/`)
  * **전체 의상 착용**: 상의와 하의를 순차적으로 적용하여 전체 의상을 입힌 이미지를 생성합니다. (`/virtual-tryon-outfit/`)
  * **FastAPI 기반**: 빠르고 효율적인 비동기 API 인터페이스를 제공하며, 자동 API 문서를 지원합니다.
  * **상세 파라미터 조절**: 추론 스텝 수(`num_inference_steps`), 스타일 강도(`guidance_scale`), 시드(`seed`) 등 생성 과정의 세부 파라미터를 조절할 수 있습니다.

## 설치 및 실행 방법 (Installation & Setup)

#### 1\. 리포지토리 복제 (Clone Repository)

```bash
git clone [GitHub 리포지토리 주소]
cd [프로젝트 폴더명]
```

#### 2\. 가상 환경 생성 및 활성화 (Setup Virtual Environment)

Python 3.9 버전을 기반으로 합니다.

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate

# 가상 환경 활성화 (macOS/Linux)
# source venv/bin/activate
```

#### 3\. 필요 라이브러리 설치 (Install Dependencies)

GPU를 지원하는 PyTorch를 포함하여 필요한 라이브러리를 설치합니다.
(RTX 30 시리즈 이상 사용 시 `cu118` 또는 `cu121` 권장)

```bash
# 먼저 PyTorch GPU 버전을 설치합니다. (예: CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 그 다음 requirements.txt로 나머지 라이브러리를 설치합니다.
pip install -r requirements.txt
```

> **참고**: `requirements.txt` 파일이 없다면, 다음 명령어로 생성할 수 있습니다:
> `pip freeze > requirements.txt`

#### 4\. (Windows만 해당) 심볼릭 링크 경고 비활성화

Hugging Face 모델 다운로드 시 `OSError: [WinError 1314]` 오류를 방지하기 위해 PowerShell에서 아래 환경 변수를 설정합니다.

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

#### 5\. API 서버 실행 (Run Server)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서버가 성공적으로 실행되면, 웹 브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하여 API 문서를 확인할 수 있습니다.

## API 사용법 (API Usage)

### 전체 의상 착용 (`/virtual-tryon-outfit/`)

사람 이미지와 상의, 하의 이미지를 모두 사용하여 최종 결과물을 생성합니다.

**Python `requests` 예시:**

```python
import requests

# API 서버 주소
url = "http://127.0.0.1:8000/virtual-tryon-outfit/"

# 전송할 파일들
files = {
    'person_image_file': open('path/to/your/person.jpg', 'rb'),
    'upper_garment_file': open('path/to/your/top.jpg', 'rb'),
    'lower_garment_file': open('path/to/your/bottom.jpg', 'rb'),
}

# 전송할 폼 데이터 (파라미터)
data = {
    'num_inference_steps': 30, # 시간을 줄이려면 값을 낮게 설정 (예: 20-30)
    'guidance_scale': 4.0,
    'seed': 123,
}

# POST 요청 보내기
response = requests.post(url, files=files, data=data)

# 응답 확인
if response.status_code == 200:
    # 성공 시, 응답으로 받은 이미지 데이터를 파일로 저장
    with open('result_outfit.png', 'wb') as f:
        f.write(response.content)
    print("성공! 'result_outfit.png' 파일로 저장되었습니다.")
else:
    print(f"오류 발생: {response.status_code}")
    print(response.json())
```

-----

## 출처 및 감사 (Acknowledgements)

이 프로젝트는 다른 훌륭한 연구와 오픈소스 프로젝트들을 기반으로 만들어졌습니다. 아래에 원본 출처를 밝힙니다.

  * **Core Model**: [zhengchong/CatVTON](https://huggingface.co/zhengchong/CatVTON)
  * **Base Model**: [booksforcharlie/stable-diffusion-inpainting](https://huggingface.co/booksforcharlie/stable-diffusion-inpainting)
  * **Original Demo**: [VIDraft/Fashion-Fit Hugging Face Space](https://huggingface.co/spaces/VIDraft/Fashion-Fit)

각 프로젝트의 라이선스 정책을 반드시 확인하고 준수해주시기 바랍니다.

## 라이선스 (License)

이 프로젝트의 기반이 되는 원본 모델 및 코드의 라이선스는 **CC BY-NC 4.0** 입니다.

[][cc-by-nc]

따라서 이 프로젝트 역시 동일한 라이선스 정책을 따릅니다.

  * **저작자 표시 (Attribution)**: 이 프로젝트를 사용하거나 수정할 경우, 반드시 원본 출처(위에 명시된 `Acknowledgements` 항목)를 밝혀야 합니다.
  * **비영리 (Non-Commercial)**: 이 프로젝트는 **상업적인 목적으로 사용할 수 없습니다.** 개인적인 학습, 연구, 포트폴리오 등의 비영리적인 용도로만 사용이 허용됩니다.

자세한 내용은 [라이선스 원문](https://creativecommons.org/licenses/by-nc/4.0/deed.ko)을 참고해주세요.


[cc-by-nc]: https://www.google.com/search?q=%5Bhttps://creativecommons.org/licenses/by-nc/4.0/deed.ko%5D\(https://creativecommons.org/licenses/by-nc/4.0/deed.ko\)
