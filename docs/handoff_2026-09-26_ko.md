# AtHOME 작업 인계 (2026-09-26 최종)

HM3DSem 단일 장면(`wcojb4TFT35`)의 데이터 파이프라인을 처음부터 다시 정리했고, Symbolic 탐색 환경을 실제 데이터에 연결해 검증까지 마쳤다. 다음 단계는 Teacher(GPT-4.1) 라벨링 파일럿이다. 아래 내용은 모두 로컬 저장소에서 실행·확인했다. 파일은 직접 읽고 확인한 뒤 수정할 것.

---

## 0. 사용자 작업 방식 (반드시 지킬 것)

- **한국어로 답한다.** 영어 용어를 남발하지 말고 쉬운 말로 설명한다. 사용자가 여러 번 "한국말로"를 요청했다.
- **정석적인 방법으로 한다.** 임시방편 금지. "빠른 방법 vs 제대로 된 방법" 선택지를 내밀지 않는다. 방법을 정할 때 선행연구·표준을 근거로 든다(사용자가 "다른 프로젝트에서도 쓰는 방법인지" 자주 확인한다. 필요하면 WebSearch로 원문 확인).
- **수동 검토를 떠넘기지 않는다.** 데이터로 근거를 계산해 규칙을 만든다. 사용자에게는 한 줄로 답할 수 있는 결정만 묻는다(범위, 하드웨어, 외부 전송·비용, 연구 방법).
- "ㄱㄱ", "진행"은 **구현하라**는 뜻이다.
- **OpenAI 제출은 승인 후에만.** 키는 출력하지 않는다: `set -a; source .env; set +a`.
- **기존 산출물을 덮어쓰지 않는다.** 새 버전 폴더에 만든다. 이번 세션에서 만든 파일을 교체할 때는 `outputs/hm3d_v3/superseded/`로 옮긴다. 리팩터링 뒤에는 기존 산출물이 바이트 단위로 재현되는지 확인한다.
- 제안서: `BARAM_26년도_후반기_제안서` (AtHOME). 사용자 이성민 = Scene Graph·탐색·실행기·LLM 연결·SFT/GRPO 담당. 윤동건 = 인지(FAST-LIO2, 검출), 문상훈 = MPPI.
- 확신하지 못하는 것을 확신하듯 말하지 않는다(예: "다수결이 무조건 맞다"고 했다가 사용자에게 지적받음 → 측정 후 결정).

## 1. 환경

```bash
cd /home/min/projects/AtHOME
source ~/miniconda3/etc/profile.d/conda.sh && conda activate athome-data   # Python 3.9.23, habitat-sim 0.3.3, NumPy 1.26.4
python -m pip install -e .
python -m pytest -q        # 100 통과, 1 실패(test_load_map_server: pyyaml 미설치, 기존 문제)
```

- pytest는 수동 설치했다. `environments/athome-data.yml`에 pytest·pyyaml이 없다.
- GPU EGL 렌더링 가능(RTX 4060). HM3DSem v0.2 semantic ID는 vertex color이므로 `use_semantic_textures=False`가 필요하다(기본값이면 전부 0).
- 쌍별 렌더링 비교는 약 23분 걸린다. 오래 걸리는 명령은 백그라운드로 돌린다.

## 2. 버전과 경로

| 버전 | 위치 | 상태 |
|---|---|---|
| legacy | `outputs/hm3d/`, `outputs/teacher/` | 원본. 변경 금지. 재현 기준 |
| meshbbox_v2 | `outputs/hm3d_meshbbox_v2/`, `outputs/teacher_meshbbox_v2/` | 중간 단계. v3로 대체. 라벨 안정성 측정(`label_stability/`)만 참고 |
| **v3 (현재)** | `outputs/hm3d_v3/`, `outputs/teacher_v3/` | mesh bbox + 범위 제외(수납가구 내부, 1.5 m 초과) + 라벨 프로토콜 v2 + 관측 모델 `wall_los_2d` |

- 층·A/B 영역·경로·Symbolic 스크립트는 `export ATHOME_HM3D_LAYOUT=outputs/hm3d_v3/layout.json`으로 버전을 고른다(`src/athome/data/hm3d/layout.py`). 없으면 legacy 경로이며 legacy 결과를 그대로 재현한다. 경로는 저장소 루트 기준.
- 문서: `docs/hm3d_pipeline_v3_ko.md`(v3 전체 생성 명령), `docs/mesh_bbox_ko.md`(bbox 규칙·LLM batch 명령), `docs/symbolic_review_ko.md`(관측 모델·검증 결과).

## 3. 확정한 결정 (근거 포함)

### 3-1. GPT 수정본 적용
`athome-symbolic-update` 12개 파일 중 11개는 원래 해시 검사로 적용했다. `toy_env.py`는 로컬 변경(perception 형식 static features)을 보존하고 `floor_z_m` 한 줄만 수동 병합했다. 백업은 `outputs/code_backups/symbolic_ghpsrq2f`. **다시 적용하지 말 것.**

### 3-2. bbox = semantic mesh 기반 (`src/athome/data/hm3d/mesh_bbox.py`, 규칙 `semantic_mesh_aabb_v2`)
- 근거: Habitat OBB에서 만든 AABB가 객체 약 8%에서 mesh를 5 cm 넘게 놓친다(예: `chair_199` 등받이 46 cm, 램프 윗부분).
- 규칙: 꼭짓점 0.10 m 이내 연결로 조각 분리 → 면적 5% 미만 잡티 제거 → 벽·바닥·천장은 남은 조각 전부 유지(가구에 가려 끊기는 게 정상) → 그 외 강체는 0.30 m 이내 묶음 중 가장 큰 묶음만.
- 사람 검토 없음(39개 수동 검토안을 사용자가 수행하기 어려워해 근거 기반 규칙으로 대체).
- 좌표 왕복(z-up ↔ habitat) 정확성 검사 포함.

### 3-3. 범위 제외 1: 수납가구 내부 (`configs/data/scene_scope.json`, `src/athome/data/hm3d/scope.py`)
- 사용자 결정: 선반·책장·수납장 **안쪽 단**의 물체는 이번 프로젝트 범위 밖(검출 한계가 아니라 범위 결정).
- 판정: 같은 방, footprint 90% 이상이 컨테이너 안, 바닥 아래로 나가지 않음, 윗면이 컨테이너 윗면보다 5 cm 이상 아래. 컨테이너는 개방형 수납가구 범주로 한정(소파·침대·계단은 쿠션 등을 잘못 포함하므로 제외).
- v3에서 164개 제외(책 94, container 11, box 8, bag 8 …). 이 장면의 책은 범위 안에 1개(`book_429`, 다른 층)만 남는다.

### 3-4. 범위 제외 2: 높이 1.5 m 초과 (`src/athome/scene_graph/query.py`)
- 사용자 결정: 천장 조명은 찾지 않는다.
- `MAX_BASE_ABOVE_FLOOR_M = 1.5`, `above_search_height()`. 아랫면이 방 바닥(바닥 객체 중심 높이 중앙값)보다 1.5 m 넘게 높으면: 탐색 위치 X, **Known 판정 X**(천장 조명 때문에 "lamp를 안다"고 잘못 판단하지 않게), 목표 GT X.
- `scripts/build_target_catalog.py`가 catalog `height_scope.excluded_object_ids`에 제외 목록을 기록한다. `--no-height-scope`로 legacy catalog 재현.
- Masking은 제안서 6.2(1)대로 **범주 전체**를 가린다(높은 인스턴스 포함). `build_component_masked_graphs.py` 검증이 이를 인정하도록 수정했다.
- v3 catalog: lamp 25→6, bag 1·box 2 제외.

### 3-5. 라벨 프롬프트 v2 + 라벨링 프로토콜 v2
- 파일: `src/athome/scene_graph/semantic_labeling.py`(프롬프트·compact 입력·모델·프로토콜 단일 정의), `label_voting.py`(다수결), `source_review.py`(tray 제외 등 공통 Source 정책).
- 프롬프트 v2: 수납가구도 다른 가구처럼 **윗면으로만** Source 여부를 판단하고 내부는 범위 밖임을 명시.
- 프로토콜 v2: 9회 샘플(temperature 1, 요청 하나에 `n=9`) → 5회 이상 선택된 Source 채택, 방·기능 라벨 최빈값, 동률은 결정적으로 깨고 표시, 득표를 라벨 `votes`에 기록.
- 근거(측정): v1에서 선반들이 방 문맥에 따라 한꺼번에 선택·제외됐고, 자기 입력이 같은 선반의 선택률이 100%→30%로 떨어졌다(프롬프트 모호성). v2 프롬프트로 방 라벨 흔들림 0, 해당 선반 100%로 안정. 남은 애매 객체에서 temperature 0 단일 답이 소수 의견인 경우 2건 → 다수결 채택.
- 실제 로봇 라벨링도 같은 코드: `scripts/build_scene_graph.py` → `OpenAIChat.sample`(n=9, temperature 1, 모델 `gpt-4.1-2025-04-14` 고정) → `pipeline.build_scene_graph` → `label_rooms`(다수결). 공통 Source 정책(`apply_source_policy`, tray 제외 등)과 그래프 검증(`validate_workspace_graph`)도 HM3D와 똑같이 적용하고, 그래프 provenance에 `labeling_protocol`을 기록한다(테스트: `tests/test_label_voting.py`, `tests/test_scene_graph_pipeline.py`). 지도당 한 번 오프라인이라 실행 중 지연 없음.
- v1은 기존 산출물 재현용(`compact_semantic_batch.py --protocol v1`로 바이트 단위 재현 확인).

### 3-6. Symbolic 관측 모델 = `wall_los_2d` (확정)
- 파일: `src/athome/symbolic/wall_los.py`, `environment.py`, 설정 `configs/data/symbolic_observation.json`.
- 근거: 제안서 6.2절 "센서를 물리적으로 시뮬레이션하지 않음", "GT 위치 + 관측 거리 조건". 거리만 쓰면 벽 너머 오탐이 생겨서 다중 물체 탐색 연구의 센서 모델(Wandzel et al., ICRA 2019, OO-POMDP / pomdp_py: 정적 구조물 + 제한 거리 + 가림)을 따른다. **카메라 높이 불필요.**
- 관측: Goal 도착 후 1회(4방향 회전 → 360° 원형). 이동 중·시작 셀 관측 없음(실제 `VisitExecutor`와 같은 규약).
- 거리: footprint 최근접점까지 3.0 m(D435i 권장 범위).
- 가림: 스캔 상태의 불투명 구조물(wall, staircase wall, door, closet door, garage door)을 바닥+0.3~1.8 m 대역으로 잘라 **정확한 평면 기하**로 사용. 닫힌 문 `door_261`이 A/B 욕실을 나누므로 문 포함(NavMesh도 스캔 상태). 유리(window, shower door, sliding door, glass)는 제외.
- 표준 ray casting: footprint 내부 3×3 표본점(1/4, 1/2, 3/4)까지 첫 가림이 2 cm 넘게 앞이면 가려짐. 벽에 붙은 물체는 방 쪽에서 보이고 벽 반대편은 가려짐.
- 관측 대상: 같은 층(floor_environments) 목표 범주 인스턴스.
- 폐기한 방식: 격자 래스터(벽 반대편 붙은 물체 오탐), 10 cm 예외 규칙.
- 검증(`scripts/compare_observation_models.py`, 결과 `outputs/hm3d_v3/wcojb4TFT35/observation_model_comparison.height_scope.review.json`): 모든 Goal 후보 자세(A 1,884 + B 2,365) × 같은 층 3 m 이내 목표, 높이 범위 안 24,440쌍. **`wall_los_2d` vs 12방향 렌더(ObjectNav oracle-visibility) 일치율 92.6%, κ 0.848**(κ 0.81 이상 "거의 완전한 일치"). 4방향 렌더 vs 12방향 렌더는 93.9%, κ 0.875.
- 남은 차이: 1 m 이내 가까운 높은/낮은 물체(고정 각도 카메라 화면 밖), 3 m 경계(Habitat depth가 광축 거리). 알려진 한계: 벽 안쪽 수납 공간처럼 일부 높이만 뚫린 곳은 막힌 것으로 봄(`box_279`).
- `habitat_render`(D435i RGB 1920×1080, HFOV 69.4°, 3 m, 1024 px, 카메라 높이 **0.88 m 임시**)는 검증 기준으로만 유지(`habitat_observer.py`). `--reuse-render`로 저장된 렌더 결과를 재사용해 2D만 수초에 재비교 가능.

### 3-7. 복도 객체 A/B 소속 결정 25개 승계
- `scripts/carry_over_membership_decisions.py`: 미배정 객체 집합이 같고 객체별 근거 변화가 허용치(0.05) 이하일 때만 승계, 넘으면 재검토 목록을 내고 중단.
- v3 최대 변화: `floor_317`의 B 겹침 +0.029 m²(기존 B 판단 방향과 일치).

## 4. 현재 결과 (v3 최종)

- 그래프: Room 17, Workspace 58, Object 803. A: Room 4 / Workspace 5 / Object 108. B: Room 5 / Workspace 11 / Object 150(legacy 250). A/B NavMesh 삼각형·Room 후보·격자 배열은 legacy와 동일.
- 라벨 경계 득표: `piano_424` 5/9, `shelf_1036` 5/9, `fireplace_350` 6/9, `sink_547` 7/9, `tray_553` 6/9(Source 정책으로 제외).
- 경로: Workspace 16개 중 경로 14, Goal 후보 없음 2(`shelf_1046`, `bedside cabinet_205`).
- 영역별 GT: A — bag 1, box 2, pillow 1, towel 2 / B — bag 2, box 3, cup 1, mug 1, pillow 8, towel 3, lamp 3.
- Symbolic 최종(`outputs/hm3d_v3/wcojb4TFT35/symbolic_wall_los_2d.include_absent.review.json`, MinCost 정책, 최대 30회): **목표가 있는 11개 중 10개 발견.** 실패는 B/bag(방 `_15`의 가방 2개, 30회 안에 `_15` 미방문). GT 없는 조합(A book·cup·mug·lamp, B book)은 전 위치 방문 후 실패로 끝남(오탐 없음).

## 5. Teacher 라벨링 파일럿 (완료)

구현: `src/athome/training/teacher.py`, `verify.py`, `scripts/generate_teacher_episodes.py`, 테스트 `tests/test_teacher.py`. 입력은 Planner 공통 형식(`src/athome/inference/prompts.py`, PROMPT_VERSION 0.1) 그대로. 실행은 일반 API(각 선택이 다음 상태를 정하므로 Batch 불가). 시작 셀은 영역 free 셀에서 고정 seed 균등 추출, component×target(GT 있는 11조합)×3 = 33 에피소드.

- 1차(`outputs/teacher_v3/pilot_teacher_gpt41_seed0`, 약 $0.2): Teacher가 "가능성 + 거리"로 한 개 선택. 방 선택 정답률 57%(가까운 곳 기준선 47%). 오답 다수가 거리에 끌린 선택(수건 → 가까운 침실).
- 문헌 조사(검증 인용): SG-Nav, LFG, ESC, VoroNav는 LLM에게 의미적 가능성만 받고 거리는 LLM 밖 수식으로 결합, HSG-ON은 가능성 순위, SmallPlan은 SFT로 의미·RL로 효율. → Teacher 프롬프트 v0.2(`TEACHER_PROMPT_VERSION`): 이유 먼저, 후보별 가능성(0~1), 거리 무시. 라벨 = argmax `가능성 / (경로비용 + λ_s)`(탐색 이론의 확률/비용 순서, λ_s는 GRPO 보상의 단계 비용과 같은 값, 제안서대로 검증 데이터로 결정). 기록에 λ_s 0.5/1/3/10/∞별 선택·채점을 함께 저장.
- 2차(`outputs/teacher_v3/pilot_teacher_gpt41_likelihood_seed0`, λ_s=3, 약 $0.74 — 출력이 길어 예상 $0.5 초과): 방 선택 정답률 54~58%로 λ_s와 무관(λ=∞도 58%). 의미가 분명한 목표는 완벽(towel 6/6, pillow 6/6, lamp 3/3). 남은 오답은 의미로 구별할 수 없는 경우(가방·상자·머그가 침실 여러 곳 중 하나, 컵이 욕실에 있음)라 채점 필터로 제외되는 것이 정상. 통과 라벨 46개, 에피소드 효율은 오히려 나빠짐(평균 거리 21 m, 4개 max_steps) — 효율은 GRPO 몫.
- 결론: Teacher 설계는 v0.2 유지. 이 장면 규모로는 λ_s를 정할 수 없음(여러 장면 검증 데이터 필요).
- 고려할 점: 의미상 동등한 후보 중 GT가 있는 쪽만 통과시키는 필터는 해당 상태의 라벨을 "운"에 따라 고른다(결과 조건부 선택 편향). 대량 생성 때 soft label(가능성 분포) 보관·활용을 검토.

## 5-1. 다음 작업

1. 데이터 규모 확보: 수동 단계 자동화는 **완료**(`docs/hm3d_pipeline_v3_ko.md` 마지막 절, `outputs/hm3d_v3_auto/`; 층·계단 분리, 로봇 바닥 면적 기준 방 후보, 복도 객체 배정 규칙이 수동 결과를 재현하고 Symbolic 16개 에피소드 동일). 남은 일반화: `inspect_component_rooms.py`의 `floor_2`·단일 섬 가정 → 모든 섬·층 순회 구동부. 현재 로컬은 minival 10장면 중 주석 4장면(00800, 00802, 00803, 00808)뿐 → HM3DSem train/val 다운로드 필요(사용자 계정·약관).
2. 건물 단위 Train/Val/Test Split, λ_s 검증 데이터 결정.
3. 대량 라벨링 방식 결정: Teacher 경로 수집(일반 API) vs 상태 선생성 + Batch 채점. 파일럿에서 Teacher 방 정답률이 기준선과 큰 차이가 없어 Teacher 경로를 따를 이점이 작음 → 선생성 + Batch 권장.
4. 통과 라벨 → SFT 학습 파일(Room/Search Location adapter별) 변환 스크립트.

## 6. 그 밖의 남은 과제

- Goal 후보 없는 Standalone이 영역마다 약 15개(offset·footprint 설정 점검).
- Masking 후 재라벨에서 방 라벨이 바뀌는 경우(예: lamp 마스킹 시 `_8` garage→storage, `_9` hallway→other). 제안서 설계대로 재라벨하지만 영향 검토 필요.
- B/bag처럼 30회 안에 도달 못 하는 경우 → max_steps 기준과 Location 수 검토.
- 제안서 문구 갱신 제안: 6.2절 관측 조건(거리 + 벽·문 가림), 라벨링 다수결, 범위(수납가구 내부·1.5 m 초과), 경로 비용 표기(구현 Dijkstra, 제안서 A* — 최단 비용 동일).
- 카메라 설치 높이는 렌더링 기준에만 필요(사용자가 나중에 알려줄 예정). `wall_los_2d`에는 불필요.
- 이전 판정 결과(`symbolic_radius_1p0*`, `symbolic_bbox_radius_1p0*`, `symbolic_habitat_render*`, `symbolic_d435i_h0p88*`)는 비교용 기록. 새 기준 결과와 섞지 말 것.

## 7. 주요 파일

- 데이터 생성: `scripts/build_mesh_bbox_annotations.py`, `review_mesh_bboxes.py`, `apply_scene_scope.py`, `build_hm3d_workspace_graph.py`, `build_target_catalog.py`, `build_masked_inputs.py`, `carry_over_membership_decisions.py`
- LLM batch: `make_semantic_batch.py`, `validate_semantic_batch.py`, `compact_semantic_batch.py --protocol`, `submit_batch.py`, `collect_semantic_batch.py`, `make_masked_semantic_batch.py`, `collect_masked_semantic_batch.py`, `build_masked_graphs.py`(경로 인자 추가, 기본값은 legacy)
- 라벨 안정성: `scripts/make_label_stability_batch.py`, `analyze_label_stability.py`
- Symbolic: `scripts/run_symbolic_review.py --observation-model wall_los_2d|habitat_render`, `scripts/compare_observation_models.py [--reuse-render]`
- 라이브러리: `src/athome/data/hm3d/{semantic_mesh,mesh_bbox,scope,layout}.py`, `src/athome/scene_graph/{semantic_labeling,label_voting,source_review,query}.py`, `src/athome/symbolic/{environment,wall_los,habitat_observer,surface}.py`, `src/athome/inference/chat.py`(`ChatJSON.sample`)
- 테스트: `tests/test_{symbolic,mesh_bbox,scope,label_voting,wall_los,height_scope}.py`
- 메모리: `/home/min/.claude/projects/-home-min-projects-AtHOME/memory/` (사용자 선호·결정 기록)
