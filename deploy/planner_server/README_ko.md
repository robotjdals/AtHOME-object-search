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

```bash
mkdir -p /data/llm/logs /data/models /data/adapters
cp env.sh serve_planner.sh /data/llm/          # 이 폴더의 파일
chmod +x /data/llm/serve_planner.sh
source /data/llm/env.sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv를 /data/llm/bin에 설치
uv python install 3.11
uv venv /data/llm/.venv --python 3.11
uv pip install --python /data/llm/.venv/bin/python vllm==0.19.1
uv pip freeze --python /data/llm/.venv/bin/python | grep -E "^(vllm|torch|transformers|flashinfer-python)=="
# → requirements-vllm.txt의 같은 줄과 비교 (torch 2.10.0, transformers 5.17.0, flashinfer 0.6.6)
```

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

## 5. 실행과 검증 (꼭 할 것)

```bash
/data/llm/serve_planner.sh            # 중지: /data/llm/serve_planner.sh stop
tail -f /data/llm/logs/vllm.log        # "Loaded new LoRA adapter" 3번, "Starting vLLM server"
```

1. **모델 목록:** `curl -H "Authorization: Bearer <키>" http://<서버>:<포트>/v1/models`
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
