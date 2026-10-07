# Small Planner 추론 서버 (vLLM) 다시 세우기

로봇이 쓰는 Small Planner(Qwen3-4B + LoRA 어댑터 3개)를 새 GPU 서버에 띄우는 절차입니다.
2026-10-04까지 쓰던 서버와 같은 구성이며, 이 폴더의 파일은 그 서버에서 그대로 가져왔습니다
(외부 주소만 지움). **서버 주소와 API 키는 이 저장소에 적지 않습니다.**

| 파일 | 내용 |
|---|---|
| `serve_planner.sh` | 시작/중지 스크립트. 어댑터를 자동으로 찾아 LoRA 옵션을 붙임 |
| `env.sh` | 캐시·설치 경로를 모두 `/data` 아래에 두는 환경 변수 |
| `requirements-vllm.txt` | 검증된 서버의 패키지 버전 목록 |

## 1. 필요 조건

- **GPU 메모리 10 GB 이상.** 이전 서버는 A100 80GB의 MIG 1g.10gb 조각이었습니다.
  실측값: 가중치 7.78 GiB, LoRA 3개를 켜고 `--enforce-eager`일 때 KV 캐시 1.04 GiB(7,536 토큰).
  메모리가 이보다 크면 스크립트를 그대로 써도 됩니다.
- **vLLM 0.19.1.** 학습 결과와의 일치 검증을 이 버전으로 했습니다. 0.20 이상은 CUDA 13(새 드라이버)이
  필요하고 결과가 달라질 수 있으니, 바꾸면 아래 5번 검증을 다시 해야 합니다.
- 이전 서버: Ubuntu 20.04, NVIDIA 드라이버 535, Python 3.11, torch 2.10.0, transformers 5.17.0.
- 디스크 약 20 GB(기본 모델 7.6 GB + 패키지 + 로그).

## 2. 설치

설치 폴더(아래 예시는 이전 서버 배치 `/data/llm`)에 `env.sh`, `serve_planner.sh`를 두고,
배치가 다르면 같은 폴더에 `serve.conf`를 만듭니다(아래 "공유 서버 배치" 참고).

```bash
mkdir -p /data/llm/logs
cp env.sh serve_planner.sh /data/llm/ && chmod +x /data/llm/serve_planner.sh
cd /data/llm && source ./env.sh            # serve.conf가 있으면 먼저: source ./serve.conf
mkdir -p bin
# uv 실행 파일만 받음 (설치 스크립트는 셸 설정 파일을 고치므로 쓰지 않음)
curl -LsSf https://github.com/astral-sh/uv/releases/download/0.9.5/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar xz -C bin --strip-components=1
bin/uv python install 3.11
bin/uv venv .venv --python 3.11
grep -v "^#" requirements-vllm.txt > requirements-exact.txt     # 이 폴더의 파일
bin/uv pip sync --python .venv/bin/python requirements-exact.txt  # 검증된 버전과 정확히 같게
bin/uv pip freeze --python .venv/bin/python | diff - <(sort requirements-exact.txt) && echo 일치
```

`uv pip install vllm==0.19.1`만 하면 그 사이 나온 새 버전이 섞입니다(2026-10-07 재설치 때 transformers
5.17.0 → 5.19.0 등 39개). 채팅 템플릿·토크나이저가 바뀔 수 있으니 `sync`로 정확히 맞춥니다.

## 3. 기본 모델 (학습과 같은 revision으로 고정)

```bash
source /data/llm/env.sh
/data/llm/.venv/bin/hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
    --local-dir /data/models/Qwen3-4B
echo 1cfa9a7208912126459214e8b04321603b3df60c > /data/models/Qwen3-4B/REVISION
```

revision이 다르면 어댑터가 맞지 않습니다. 학습·평가는 모두 이 revision으로 했습니다.

## 4. API 키와 어댑터

```bash
openssl rand -hex 24 > /data/llm/api_key && chmod 600 /data/llm/api_key
```

- 키는 노트북 `.env`의 `ATHOME_PLANNER_API_KEY`에만 넣습니다. 채팅·저장소에 붙이지 않습니다.
- vLLM은 기동할 때 실행 옵션(키 포함)을 `logs/vllm.log`에 적으니, 로그를 공유하지 않습니다.

어댑터는 노트북의 `outputs/adapters/upload/`에 있습니다(각 폴더에 `meta.json` 포함).

```bash
# 노트북에서
scp -r outputs/adapters/upload/room_v1 <서버>:/data/adapters/room/
scp -r outputs/adapters/upload/search_location_v1 <서버>:/data/adapters/search_location/
scp -r outputs/adapters/upload/workspace_v1 <서버>:/data/adapters/workspace/
# 서버에서
for n in room search_location workspace; do ln -sfn ${n}_v1 /data/adapters/$n/current; done
```

| 서빙 이름 | 폴더 | 내용 | 가중치 sha256 앞 12자리 |
|---|---|---|---|
| room | room_v1 | 방 선택 (SFT) | d5ad378eaa34 |
| search_location | search_location_v1 | 탐색 위치 선택 (SFT) | 599be423b0ce |
| workspace | workspace_v1 | 워크스페이스 선택 (SFT + GRPO 200스텝, 제안서 방식) | bf63944d4292 |

평가용 어댑터는 `/data/adapters/eval/<서빙 이름>/`에 두면 함께 등록됩니다(로봇은 쓰지 않음).
GPU에는 3개만 올리고 나머지는 요청 때 바꿔 끼워서 메모리는 그대로입니다.

## 공유 서버 배치 (다른 사람과 GPU·폴더를 같이 쓸 때)

2026-10-07부터 쓰는 연구실 서버(A100 80GB를 여러 사람이 공유하는 컨테이너)의 배치입니다.
**다른 사람의 폴더·conda·프로세스는 건드리지 않고, 모든 파일을 한 폴더(`/data/athome`) 안에만 둡니다.**
설치 폴더 `/data/athome/llm`에 둔 `serve.conf`:

```bash
CACHE_ROOT=/data/athome/cache          # uv·pip·HF·vLLM 캐시
TOOLS_DIR=/data/athome/llm             # uv 실행 파일, uv가 설치한 Python
ISOLATED_HOME=/data/athome/home        # HOME·설정·임시 파일도 이 안으로 (root 홈, /tmp에 안 씀)
MODEL_DIR=/data/athome/models/Qwen3-4B
ADAPTER_DIR=/data/athome/adapters
HOST=127.0.0.1                         # 컨테이너 밖에서는 SSH 터널로만 접근
PORT=6007
GPU_UTIL=0.15                          # A100 80GB의 15% ≈ 12 GB (공유 GPU)
```

- 실측: 가중치 7.78 GiB, KV 캐시 3.83 GiB(27,872 토큰). 전용 10 GB 조각보다 여유 있음.
- `DRY_RUN=1 ./serve_planner.sh`로 실행 옵션(키는 가림)과 경로를 먼저 확인합니다.
- 바깥 접속은 SSH 포트 하나로 터널을 엽니다(서비스 포트를 따로 열지 않음, 요청·키가 SSH로 암호화됨).
  ```bash
  ssh -f -N -o ServerAliveInterval=30 -L 16007:127.0.0.1:6007 -p <SSH 포트> root@<서버>
  # 노트북·로봇 쪽 base_url: http://127.0.0.1:16007 (로봇 장비에서는 autossh로 상시 유지)
  ```

## 5. 실행과 검증 (꼭 할 것)

```bash
/data/llm/serve_planner.sh            # 중지: /data/llm/serve_planner.sh stop
tail -f /data/llm/logs/vllm.log        # "Loaded new LoRA adapter" 3번, "Starting vLLM server"
```

1. **모델 목록:** `curl -H "Authorization: Bearer <키>" http://<서버 또는 터널>/v1/models`
   → `qwen3-4b`, `room`, `search_location`, `workspace`.
2. **배포 검증 (노트북에서, 약 45분):** 같은 에피소드를 새 서버로 돌려 이전 서버 결과와 비교합니다.
   ```bash
   set -a; . ./.env; set +a
   PYTHONPATH=src:scripts python scripts/evaluate_student.py episodes --planner-url http://<서버>:<포트> \
       --scenes holdout --output outputs/eval_server/v1_<새서버>_episodes_holdout.json
   PYTHONPATH=scripts python scripts/compare_episode_evals.py outputs/eval_server/v1_server_episodes_holdout.json \
       outputs/eval_server/v1_<새서버>_episodes_holdout.json --names old new
   ```
   기대값: 비용 차이 신뢰구간이 0을 포함하고, 대부분의 에피소드가 같아야 합니다.
   이전 서버와 학습 서버(HF) 비교에서는 440개 중 412개가 같았고 비용 차이는 −0.34(CI −0.96 ~ +0.10)였습니다.
   요약의 `student_policy_fallbacks`가 0이 아니면 서버 오류로 대체된 판단이 있는 것입니다.
3. 로봇 설정(`configs/robot/demo.yaml`의 `planner.llm.base_url`)은 로컬에서만 새 주소로 바꿉니다.

## 이전 서버 백업

노트북 `outputs/planner_server_backup/`(저장소에 올리지 않음):

- `llm/`: 원래 README, 실행 스크립트(평가 어댑터 추가 전 원본 포함), 패키지 목록, 모델 비교 결과, 로그
- `adapters/`: 서버에 있던 어댑터 전부(평가용 포함, sha256으로 서버와 일치 확인)

API 키는 백업하지 않았습니다. 새 서버에서 새로 만듭니다.
