# GPU 서버 작업 인계: LoRA-SFT → GRPO (Small Planner, Qwen3-4B)

작성: 2026-09-29, 데이터셋 작업 창. 받는 쪽: GPU 서버에서 학습을 돌리는 Claude 세션.

## 0. 규칙 (반드시 지킬 것)
- **출력 덮어쓰기 금지.** 모든 스크립트는 기존 출력 폴더가 있으면 멈춘다. 새 이름을 쓴다.
- **비밀 정보 출력 금지.** 토큰·키는 `.env`에서 `set -a; source .env; set +a`로 읽고 화면에 찍지 않는다.
- **HM3D 데이터는 공개 저장소에 올리지 않는다**(Matterport 이용 약관).
- **GPU 시간은 과금된다.** 학습이 끝나면 사용자에게 알리고 인스턴스를 끄도록 안내한다.
- 사용자에게는 **한국어**로 보고한다.

## 1. 해야 할 일 (순서)
1. 환경 준비 (2절)
2. 초소형 모델로 코드 경로 시험 (3.1절, CPU로도 가능, 약 3분)
3. 실제 모델로 짧은 시험 실행: 메모리·속도 확인 (3.2절, 약 10분)
4. **LoRA-SFT 본 학습**: `room`, `search_location` 어댑터 2개 (4절, 약 1~1.5시간 / L40S 기준 추정)
5. 결과를 사용자에게 보고 → **GRPO는 사용자 확인 뒤** (5절, 약 4~10시간 추정)

## 2. 환경
- 권장 사양: GPU 48GB 1장(L40S, RTX A6000 또는 A100 40GB 이상), CPU 16코어 이상, 디스크 200GB 이상.
- 코드(공개 저장소, 브랜치 `dataset-v5-training`):
  ```bash
  git clone https://github.com/robotjdals/AtHOME-object-search.git AtHOME
  cd AtHOME && git checkout dataset-v5-training
  ```
- 데이터는 git에 없다(`data/`, `outputs/`는 제외 대상). 사용자가 로컬에서 직접 복사한다. 예(서버 주소는 사용자가 채움):
  ```bash
  rsync -avP outputs/sft_v5g <서버>:AtHOME/outputs/                      # SFT 학습
  rsync -avP outputs/teacher_v5/episodes_*.grouped <서버>:AtHOME/outputs/teacher_v5/   # SFT 평가(decisions의 GT 정답률), 189 MB
  rsync -avP outputs/hm3d_v5 outputs/eval_v5_everygoal outputs/eval_v5 <서버>:AtHOME/outputs/   # GRPO·평가
  rsync -avP --include='*/' --include='*.semantic.glb' --include='*.semantic.txt' --exclude='*' \
      data/versioned_data/hm3d-0.2/hm3d/train data/versioned_data/hm3d-0.2/hm3d/val <서버>:AtHOME/data/versioned_data/hm3d-0.2/hm3d/
  ```
  서버에서 `data/scene_datasets/hm3d`는 `data/versioned_data/hm3d-0.2/hm3d`를 가리키는 상대 심볼릭 링크로 만든다.
- 파이썬 3.9(로컬 시험과 같음) 권장. 패키지:
  ```bash
  pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128   # 서버 드라이버에 맞는 CUDA 빌드
  pip install -r requirements/train_gpu.txt
  pip install -e .            # src/athome 패키지 (없으면 PYTHONPATH=src:scripts)
  ```
- 기본 모델: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`로 **고정**한다. 로봇의 vLLM 서버가 같은 revision이다. 설정 파일에 이미 들어 있다.
- 옮길 데이터:
  | 용도 | 경로 | 크기 |
  |---|---|---|
  | SFT | `outputs/sft_v5g/` | 38 MB |
  | SFT 평가 | `outputs/teacher_v5/episodes_*.grouped/` (Teacher 기록, 정답 후보 목록) | 189 MB |
  | GRPO | `outputs/hm3d_v5/` | 7.3 GB |
  | GRPO | `outputs/eval_v5_everygoal/` | 4.6 MB |
  | GRPO | `data/scene_datasets/hm3d/train/*/*.semantic.glb`, `*.semantic.txt` (145개 장면) | 8.1 GB |
  | 설정 | `configs/` 전체 | 작음 |
  - 장면 메쉬는 실제 파일이 `data/versioned_data/hm3d-0.2/hm3d/train/...`에 있고 `data/scene_datasets/hm3d`는 그쪽을 가리키는 심볼릭 링크다. 서버에서도 저장소 안 같은 상대 경로에 둔다. 주석 파일에 남은 이 노트북의 절대 경로는 `athome.data.hm3d.layout.repo_path`가 저장소 기준으로 다시 해석한다.
  - GRPO 롤아웃에 habitat-sim은 필요 없다(numpy, scipy, shapely만).

## 3. 시험 실행
### 3.1 초소형 모델 (코드 경로 확인)
```bash
T=outputs/smoke/tiny   # 새 폴더
python scripts/make_tiny_planner.py --output $T
for a in room search_location; do
  python scripts/train_sft_lora.py --config $T/sft_lora.yaml --data outputs/sft_v5g/train/$a.jsonl \
      --output $T/sft_$a --limit 200 --epochs 1
done
python scripts/train_grpo.py --config $T/grpo.yaml --output $T/grpo_run --max-steps 2
```
- 로컬 결과(기준값): SFT는 `trainable params: 32,768`(7개 선형 층 모두), 어댑터 저장. GRPO는 2스텝, `groups: {"usable": 2}`, 두 번째 스텝 `mean_kl` 약 1e-6, `step_0000N/workspace` 저장.
- 토크나이저 "incorrect regex pattern" 경고가 나올 수 있다. 로컬에서 학습 예제 600개로 원본 토크나이저와 결과가 같음을 확인했다(무해).

### 3.2 실제 모델 (메모리·속도 확인)
```bash
python scripts/train_sft_lora.py --data outputs/sft_v5g/train/room.jsonl --output outputs/smoke/sft_room_real \
    --limit 400 --epochs 0.2
```
- `nvidia-smi`로 최대 메모리를 보고, 초당 샘플 수로 본 학습 시간을 다시 추정해 사용자에게 알린다.
- 메모리가 부족하면 `configs/training/sft_lora.yaml`의 `per_device_batch_size`를 줄이고 `gradient_accumulation_steps`를 늘린다(곱 16 유지). 설정 변경은 사용자에게 알린다.

## 4. LoRA-SFT 본 학습
```bash
python scripts/train_sft_lora.py --data outputs/sft_v5g/train/room.jsonl --output outputs/adapters/room_v1
python scripts/train_sft_lora.py --data outputs/sft_v5g/train/search_location.jsonl --output outputs/adapters/search_location_v1
```
- 설정(`configs/training/sft_lora.yaml`)과 근거
  - LoRA rank 16, alpha 32, dropout 0.05
  - **모든 선형 층**(q, k, v, o, gate, up, down). LoRA Without Regret(2025) 근거로 사용자와 합의(제안서의 q, k, v, o에서 변경)
  - 학습률 2e-4, cosine, warmup 3%, 2 에폭, 배치 8 × 누적 2. 에폭마다 검증 손실을 재고 **가장 낮은 에폭의 가중치**를 저장한다(학습 데이터에 같은 결정의 섞은 복사본이 있어 과적합 방지). `train_report.json`의 `best_checkpoint`와 `best_eval_loss`로 확인한다.
  - 손실은 답 토큰에만 건다.
  - 입력 형식은 로봇 추론과 같다: Qwen3 chat template, 생성 프롬프트, `enable_thinking=False`, 답 뒤에 `<|im_end|>`
- 검증: 건물 해시 5%(7개 건물)를 검증 손실용으로 뗀다. 데이터 규모: 방 train 8,008 / val 538, 탐색 위치 train 4,761 / val 231.
- 데이터 출처: `outputs/sft_v5g/manifest.json`
  - 프롬프트 0.6(단독 물체 후보 종류별 묶음)
  - 후보 순서: 기록 순서 + 섞은 복사본 1개(RankVicuna 방식)
  - 목표 범주 v5(학습 209개)
- 결과로 보고할 것: `outputs/adapters/<name>/train_report.json`의 train/eval loss 추이, 걸린 시간, 최대 메모리.

## 4.5 SFT 평가 (본 학습 직후, API 없음)
```bash
for a in room search_location; do
  python scripts/evaluate_student.py decisions --data outputs/sft_v5g/train/$a.jsonl \
      --adapter outputs/adapters/${a}_v1/adapter --output outputs/eval_student/${a}_decisions.json
done
python scripts/evaluate_student.py episodes --room outputs/adapters/room_v1/adapter \
    --search-location outputs/adapters/search_location_v1/adapter --output outputs/eval_student/sft_episodes.json
```
- **decisions**: 검증 건물(학습에서 뺀 7개)에서 Student 선택이 Teacher 라벨과 같은 비율을 두 방식으로 잰다.
  - `score`: 답 전체의 확률이 가장 높은 후보(GRPO가 쓰는 방식)
  - `greedy`: 후보 목록 안에서 한 토큰씩 고름(로봇 vLLM과 같은 방식)
  - 두 방식의 일치율(`score_greedy_agreement`)도 본다. Qwen 토크나이저는 숫자를 한 자리씩 잘라서, 후보가 10개 이상이면 두 방식이 다를 수 있다. `10+ candidates` 항목의 일치율이 낮으면(예: 95% 미만) 사용자에게 알린다. 로봇 쪽을 후보 채점 방식으로 바꿀지 결정해야 한다.
  - 무작위 기대값(`random_expected`)과 비교해 학습 효과를 판단한다.
- **episodes**: 검증 건물의 시작 위치에서 Student(greedy), 가장 가까운 곳, 무작위를 같은 시작 위치·같은 후보 순서로 비교한다. 지표는 성공률, SPL, 거리, 방문 수, 보상 비용(D + 3N, 낮을수록 좋음).
  - 라벨 데이터 기준 참고값(train 전체, Teacher): 보상 비용 평균 24.6, 가장 가까운 곳 56.8, 무작위 52.6
- 결과 JSON을 사용자에게 보고한다. 로컬 초소형 모델로 두 모드 모두 동작을 확인했다.

## 5. GRPO (사용자 확인 뒤)
```bash
python scripts/train_grpo.py --config configs/training/grpo.yaml --output outputs/grpo/run1 --max-steps 2   # 먼저 짧게
python scripts/train_grpo.py --config configs/training/grpo.yaml --output outputs/grpo/run1_full
```
- 워크스페이스 어댑터만 학습한다. 시작값은 탐색 위치 SFT 어댑터 복사본이다. 방 어댑터와 탐색 위치 어댑터는 고정이고, 탐색 위치 어댑터가 기준 정책(KL 대상)이다.
- 정책: 후보마다 답 문자열의 로그 확률을 채점해서 그 분포에서 행동을 뽑는다(GLAM 방식). 워크스페이스 결정만 샘플링하고, 방·단독 물체 결정은 최빈값을 쓴다. 후보 순서는 질의마다 섞는다.
- 목적함수(제안서 6-4, Shao et al. 2024)
  - 보상 R = −D − λ_s·N, λ_s = 3
  - G = 8, 그룹 내 상대 이점
  - 확률비 잘라내기 ε = 0.2, KL β = 0.04, 학습률 1e-5
  - 실패한 롤아웃이 있거나 보상이 모두 같은 그룹은 제외한다.
- 시작 상태: `outputs/eval_v5_everygoal`("모든 지점에서 보이면 찾을 수 있음" 규칙). SFT 검증 건물은 제외한다.
- GRPO 뒤 평가: 위 `episodes`에 `--workspace outputs/grpo/<run>/step_XXXXX/workspace`를 더해 SFT 결과와 비교한다.
- 기록: `log.jsonl`(스텝별 성공률, 평균 보상·거리·방문, 제외 그룹 사유, KL, 확률비), `step_XXXXX/workspace` 어댑터.
- 시간은 불확실하다. 첫 스텝 시간(`seconds`)으로 전체 시간을 다시 추정해 사용자에게 알린다.

## 5.5 최종 평가: 처음 보는 건물 (Val / Test)
```bash
python scripts/evaluate_student.py episodes --room outputs/adapters/room_v1/adapter \
    --search-location outputs/adapters/search_location_v1/adapter [--workspace <GRPO 어댑터>] \
    --start-states "outputs/eval_v5/*/start_states.jsonl" --scenes val --output outputs/eval_student/val_episodes.json
```
- 공식 val 폴더의 36개 건물을 건물 ID 해시로 Val 24개, Test 12개로 나눴다. 설정은 `--scenes val` 또는 `--scenes test`이다. **Test는 최종 보고 때 한 번만 쓴다**(설정 조정은 Val로).
- 결과는 학습 범주(seen), 동의어 범주(synonym), 처음 보는 범주(unseen)별로 나온다(HM3D-OVON 방식).
- `outputs/eval_v5`에는 train 장면의 옛 시작 위치도 있지만, `--scenes val/test`가 공식 val 폴더 건물만 고른다.

## 5.9 추론 서버 호환 조건 (추론 서버 담당 전달 사항 요약)
- 기본 모델 revision 고정, **bf16 LoRA**, all-linear, **rank 16**(더 크게 하려면 먼저 추론 서버 담당에게 알릴 것: 서빙 메모리 검증이 rank 16 × 어댑터 3개 기준)
- 토큰 추가·vocab 변경, `modules_to_save`(embed_tokens / lm_head), DoRA 등 LoRA 변형 **금지**
- **merge하지 않은** PEFT 형식(`adapter_config.json` + `adapter_model.safetensors`)으로 저장한다. 우리 스크립트가 이렇게 저장한다.
- 어댑터 폴더마다 **`meta.json`**이 자동 생성된다(`athome.training.adapter_meta`).
  - 기록 내용: 기본 모델 revision, rank, alpha, 적용 층, 학습 방식(SFT/GRPO), 데이터 버전, PROMPT_VERSION, 날짜, git 커밋, 라이브러리 버전
  - 평가 결과는 `evaluate_student.py --update-meta <어댑터 폴더>`로 `validation` 항목에 추가한다(Teacher 일치율 `accuracy_*`, GT 정답률 `gt_accuracy_*`, 두 디코딩 방식 일치율).
- 폴더 이름에 버전을 붙인다: `room_v1`, `search_location_v1`, `workspace_v1`
- 추론 서버 주소, 계정, API 키는 **공개 저장소에 적지 않는다.** 사용자에게 따로 받는다.

## 6. 끝나면
- `outputs/adapters/room_v1/adapter`, `outputs/adapters/search_location_v1/adapter`(이후 `workspace`)를 로컬로 가져온다. 어댑터당 수십 MB다.
- 로봇 vLLM 서버에는 어댑터 이름 `room`, `search_location`, `workspace`로 올린다(`configs/robot/demo.yaml` planner.llm.models).
- 인스턴스 종료를 사용자에게 안내한다.
