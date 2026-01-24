# LGAimers Lab

## 시작하기

이 프로젝트는 `uv`를 사용해 의존성을 관리하며, `make`를 사용하여 워크플로우를 자동화합니다.

### 사전 요구사항

- **Python**: 3.11 (`uv`가 자동으로 관리)
- **uv**: Python 패키지 및 프로젝트 매니저
- **Git**

### 1. `uv` 설치

먼저 시스템에 `uv`를 설치해주세요.

#### macOS
```bash
# Homebrew 사용
brew install uv

# 또는 curl 사용
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows
```powershell
# PowerShell 사용
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 또는 pip 사용 (Python이 이미 설치된 경우)
pip install uv
```

---

### 2. 프로젝트 설정

`uv` 설치가 완료되면 저장소를 클론하고 환경을 설정합니다.

#### macOS / Linux
```bash
# 저장소 클론
git clone https://github.com/bbun-550/lgaimers-lab.git
cd lgaimers-lab

# 환경 설정 (의존성 설치)
make setup

# 설치 확인 (테스트 실행)
make test
```

#### Windows
Windows 사용자는 기본적으로 `make`가 설치되어 있지 않을 수 있습니다. 이 경우 `uv` 명령어를 직접 사용하세요.

```powershell
# 저장소 클론
git clone https://github.com/bbun-550/lgaimers-lab.git
cd lgaimers-lab

# 환경 설정
uv sync

# 설치 확인
uv run pytest
```

> **Windows 사용자 참고**: 
> `make` 명령어를 사용하려면 [Make for Windows](https://gnuwin32.sourceforge.net/packages/make.htm)를 설치하거나 WSL(Windows Subsystem for Linux)을 사용하는 것을 권장합니다.

---

### 3. 사용 방법

일반적인 작업은 `Makefile`을 통해 실행할 수 있습니다.

| 명령어            | 설명                          |
| ----------------- | ----------------------------- |
| `make setup`      | 의존성 설치 및 가상 환경 생성 |
| `make preprocess` | 데이터 전처리 실행            |
| `make analyze`    | 모델 구조 및 파라미터 분석    |
| `make train`      | 모델 학습 (경량화 변형 지원)  |
| `make eval`       | 모델 성능 평가                |
| `make report`     | 비교 리포트 생성              |
| `make clean`      | 출력물 및 캐시 정리           |

**수동 실행 (Make 없이 실행):**
`make`를 사용할 수 없는 경우 `uv run`을 사용하여 스크립트를 직접 실행할 수 있습니다.

```bash
# 예시: 모델 학습
uv run python src/models/train.py --config-name=config
```

## 프로젝트 구조

- `src/models/base`: 원본 EXAONE 레퍼런스 모델
- `src/models/variants`: 경량화된 모델 변형들
- `src/data`: 데이터 로딩 및 전처리 스크립트
- `src/compression`: 분석 및 경량화 로직
- `configs/`: Hydra 설정 파일