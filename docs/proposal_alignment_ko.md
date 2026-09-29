# 제안서 대비 구현 점검 (2026-09-27)

기준: `BARAM_26년도_후반기_제안서` (2026.09.18) 4~6장. 제안서와 다르게 진행하는 항목은 사용자에게 설명하고 합의한 뒤 바꾼다. 이 문서는 그 목록이다.

## 제안서와 일치

| 제안서 | 구현 |
|---|---|
| 6-1 HM3DSem: region → GT Room, object → Object Node | 동일(`export_hm3d_annotations.py`) |
| 4-2 기하 관계 d_xy, Δz, O_xy → Child 후보 | 동일(0.5 m, 0.1 m, 0.5) |
| 4-2 LLM이 Room label, Workspace, Function 추론 | 동일(GPT-4.1) |
| 4-3 Room / Workspace / Source / Child / Standalone | 동일 |
| 6-2 ⑴ Target masking, GT 별도 유지 | 동일(masked graph + target catalog) |
| 6-2 ⑵ 선택 → A* cost 누적 → Visited 갱신 → 다음 State, GT 발견까지 rollout | 동일(`SearchSession`, `SymbolicEnvironment`) |
| 5-2 Goal 방향각 θ = atan2(bbox 중심 − 후보) | 동일 |
| 5-1 Stage 1 Room 선택, Stage 2 Search Location 선택, 방문한 곳 제외 | 동일 |
| 6-3 Student는 Candidate ID만 출력 | 동일(`prompts.build_messages`) |
| 6-4 R(τ) = −D(τ) − λ_s N(τ) | 동일. Teacher 라벨도 같은 λ_s를 쓴다 |
| 6-1 Train / Validation / Test 환경 분리 | 예정(건물 단위 분리, 장면 확보 후) |

## 합의된 변경

| 항목 | 제안서 | 변경 | 합의 |
|---|---|---|---|
| Room·Workspace 라벨 | LLM 1회 | 9회 샘플링 다수결(Self-consistency). 단일 응답의 Source 선택이 문맥에 따라 뒤집힘 | 2026-09-26 |
| Symbolic 관측 | (명시 없음, 시나리오 4: 회전 관측) | 목표 지점에서 360° 관측, 3 m, 벽·문 2D 가시선(Wandzel ICRA 2019 방식). 렌더링 대비 일치 92.6% | 2026-09-26 |
| 범위 | (명시 없음) | 수납가구 안쪽 단 물체 제외, 바닥 위 1.5 m 초과 물체 제외 | 2026-09-26 |
| 목표 범주 별칭 | (명시 없음) | 휴대형 하위 종류 포함(table lamp → lamp 등) | 2026-09-26 |
| Teacher 출력(6-3) | Candidate ID + reason | reasoning + 후보별 가능성, 라벨 = argmax p/(A* cost + λ_s). Student는 ID만 출력(제안서와 같음) | 2026-09-27 |

## 제안서 식에 맞춘 수정 (2026-09-27)

### 로봇 크기 (5-2의 r_robot, 합의 2026-09-27)

- 근거: 제안서 사양 Tidybot++ 50 × 54 cm, Nav2 costmap의 정의.
- `configs/robot/demo.yaml`에 `footprint_m`만 적고 반경은 코드(`athome.config.RobotGeometry`)가 계산한다.

| 항목 | 값 | 정의 |
|---|---|---|
| 이동 반경 | 0.25 m | 내접 반경. `inflation_radius`, Nav2 INSCRIBED. 2D 경로 계획(NavFn)이 통과 여부를 판단하는 기준이고, 몸체 충돌은 MPPI가 footprint로 확인한다 |
| 회전 여유 | 0.368 m | 외접 반경. 목표 지점마다 360° 제자리 회전 |
| d_off | 0.468 m | r_robot(외접) + d_safe 0.10 |

- Nav2 costmap에는 `footprint` 다각형을 넘긴다. 실제 로봇 탐색 서버와 데이터셋이 같은 설정을 쓴다.
- TidyBot(2023)은 외접 반경 + 10 cm로 전체 공간을 막았다. 그 프로젝트의 제어기가 충돌을 확인하지 않았기 때문이다. 우리는 MPPI가 footprint를 확인하므로 Nav2 방식을 따른다.

### 목표 지점 (5-2, 합의 2026-09-27)

- 규칙: d_off 이상이고 회전 여유가 있는 칸 중 물체에 가장 가까운 거리대(한 칸 폭)를 후보로 쓴다. 상한은 1.0 m(`navigation.goal_max_offset`). 그 거리대 안에서는 제안서 식대로 min A* cost 지점을 고른다.
- 조사한 실물 로봇 프로젝트:

| 프로젝트 | 목적 | 규칙 |
|---|---|---|
| OK-Robot (실제 가정 10곳, Stretch) | 집기 | 도달 가능한 칸 중 "물체 거리 + 0.4 m 미만 벌점 + 장애물 근처 벌점"이 최소인 곳 |
| HomeRobot 실제 로봇 설정 | 집기 | 물체를 0.5 m 부풀린 영역 |
| TidyBot | 집기·놓기 | 고정 거리 0.55~0.80 m |
| GOAT (Spot, Stretch) | 물체 탐색 | 1 m 안이면 성공 |
| Habitat ObjectNav | 물체 탐색 | 1.0 m 안이고 물체가 보이면 성공 |

  팔로 집는 프로젝트는 0.4~0.55 m, 관측·탐색만 하는 과제는 1 m를 쓴다. 우리는 관측만 하므로 1.0 m를 상한으로 둔다.
- 결과(v4 네 장면): 경로가 있는 Workspace가 늘었다(00808: 43 → 52/57, 00800: 10 → 12/17). 남은 "후보 없음"은 Workspace가 이동 가능 영역에서 1.1~3 m 떨어졌거나, 회전할 수 없는 좁은 틈에 있어 Tidybot이 실제로 다가갈 수 없는 경우다. 발견 31/42.
- 남은 실패는 대부분 30회 방문 제한이다.

## Search Location 단위 비교 (2026-09-27, 로봇 반경 적용, max 30회, min-cost 정책)

| 방식 | 00800 발견 | 00808 발견 | 탐색 위치 수(중앙값) | 목표가 없을 때 종료까지 방문(중앙값) |
|---|---|---|---|---|
| 현재: Standalone도 탐색 위치 | 6/8 | 15/16 | 72.5 / 28 | 16 / 11 |
| (A) Workspace만(제안서 문구) | 6/8 | 9/16 | 6.5 / 4.5 | 1 / 4 |
| (B) 관측된 물체의 위치를 Visited 처리 | 7/8 | 13/16 | 72.5 / 28 | 2 / 2 |
| A+B | 5/8 | 9/16 | 6.5 / 4.5 | 1 / 2 |

- (A)는 Workspace 밖에 놓인 목표(바닥의 상자·수건 등)를 보는 지점이 없어져 00808에서 6개를 더 놓친다.
- (B)가 놓친 경우는 "물체가 보였다 = 그 위치를 탐색했다"가 성립하지 않는 경우다. 예: `box_158`은 `hanger_139` 앞 지점에서 보이는데, 옷걸이가 다른 곳에서 먼저 보여 그 지점을 건너뛰었다. 탐색 위치를 방문하는 가치는 물체가 아니라 그 지점에서의 360° 시야다.
- 결론: 기본값은 현재 방식(제안서 5-1 "Search Location Selection", 로봇 설정의 `standalone: search_location` 어댑터와 같은 구조)을 유지한다. (A)·(B)는 옵션(`--workspace-locations-only`, `--coverage`)으로만 둔다.
- 0.1 m 반경에서 보이던 30회 제한 실패(00808 A 영역)는 로봇 반경을 적용한 뒤 거의 사라졌다(15/16).
- 목표가 없을 때 탐색을 빨리 끝내는 효과(B)를 정석대로 얻으려면, 물체 단위가 아니라 방문 지점의 관측 영역이 다른 방문에서 이미 관측되었는지를 보는 시점 커버리지가 필요하다. GRPO 보상(N 단계)에 영향이 크면 그때 검토한다.

## 종료 조건과 탐색 위치 정책 (2026-09-27, 합의)

### 종료 조건

| 구분 | 종료 조건 | 구현 |
|---|---|---|
| 실제 로봇 | 발견 / 전부 방문 / 시간 예산 | `SearchSession(time_budget_s, elapsed_s)`, `CommandStatus.TIME_BUDGET`, action `STATUS_TIME_BUDGET=5`, `configs/robot/demo.yaml`의 `search.time_budget_s` |
| 평가(Symbolic) | 실제 로봇과 같은 시간 예산 | `run_symbolic_review.py --time-budget`. 모델 시간 = 이동 거리 ÷ `nominal_speed_mps` + 방문마다 (2π ÷ `rotation_speed_radps` + heading 수 × `observation_window`) |
| SFT 생성 | 비용 관리용 방문 상한 | `generate_teacher_episodes.py --max-steps` |
| GRPO | 평가 예산 이상의 horizon, 끊기면 실패 | GRPO 연결부 작성 시 적용 |

- 근거:
  - GenMOS(Spot·MOVO 실제 로봇): 목표를 모두 찾거나 시간 예산(10분)이 되면 종료.
  - HomeRobot 실제 로봇 300 스텝, GOAT 200 스텝.
  - GiGPO(ALFWorld 최대 50스텝), RAGEN(5턴/10행동): 끊긴 궤적은 실패.
- 확정이 필요한 값:
  - `time_budget_s`(팀)
  - `nominal_speed_mps`, `rotation_speed_radps`(문상훈 님)
  - 값이 비어 있으면 `--time-budget` 평가는 오류로 멈춘다.

### 탐색 위치 정책 (mpcat40)

- `configs/data/search_locations.json`: 공식 HM3DSem → mpcat40 매핑(habitat-sim, MIT, 해시 기록)을 쓴다.
- 건물 요소·부착물 분류(wall, floor, ceiling, door, window, stairs, column, beam, railing, blinds, curtain, picture, board_panel, lighting)는 Standalone 탐색 위치에서 뺀다.
- 적용 범위: 데이터셋(layout의 `search_locations`)과 실제 로봇(`configs/robot/demo.yaml`의 `scene_graph.search_locations`)에 같은 정책을 쓴다. 플래너 입력이 학습과 같아야 하기 때문이다.
- Known 탐색은 탐색 위치에서 분리했다. 탐색 위치가 아닌 물체(예: lighting으로 빠진 lamp)는 물체 목표 `object:<ID>`로 곧장 간다(`SceneGraph.object_goals`).
  - 이 목표는 미지 목표 탐색의 후보에 들어가지 않으므로 LLM 입력과 학습 데이터는 바뀌지 않는다. 00800에서 에피소드가 동일한 것을 확인했다.
  - 이동 후보 계산은 Known 목표가 필요할 때만 한다.
- 측정 결과:

| 장면 | 탐색 위치 수(중앙값) |
|---|---|
| 00800 | 98.5 → 78 |
| 00808 | 46 → 38 |
| wcojb4 | 61.5 → 54 |
| 00803 | 18.5 → 6.5 |

  제한 없이 돌리면 발견 36/42로 변화가 없다. wcojb4에서는 부착물 위치에서 보이던 목표가 있어 최대 방문 수가 23 → 58로 늘었다.

## GRPO 보상과 "찾을 수 있는 시작 상태" (2026-09-27, 합의)

- 보상은 제안서대로 R(τ) = −D(τ) − λ_s N(τ)를 유지한다. 성공 항을 추가하는 안은 철회했다.
- 이 보상은 "모든 rollout이 목표를 찾는다"는 가정 위에 있다. 그 가정을 데이터로 보장한다.
  - **시작 상태 필터** (`athome.training.findability.findable_from`):
    - 시작 위치에서 도달할 수 있는 탐색 위치의 goal 후보 중 하나라도 목표를 관측하면, 그 시작 상태를 쓴다.
    - GT는 시작 상태를 고르는 데만 쓰고, 플래너 입력에는 넣지 않는다.
    - 근거: Habitat 에피소드 생성기는 호환되는(풀 수 있는) 에피소드만 남긴다. DAPO dynamic sampling은 학습 신호가 없는 프롬프트를 뺀다.
  - **Teacher 에피소드:** 시작 위치를 "찾을 수 있는 위치"에서만 균등하게 뽑는다(거절 샘플링). 판정은 4-연결 영역 단위로 한 번만 한다.
  - **남는 실패:** 필터가 goal 후보 선택에 대해 낙관적이라, 실제 세션이 다른 goal을 골라 목표를 못 보는 경우가 드물게 있다. 이런 rollout은 실패로 표시하고 학습에서 뺀다(DAPO와 같은 처리).
- 측정(v4, 방문 제한 없음, mpcat40 탐색 위치 정책):
  - GT가 영역 안에 있는 조합 42개 중 찾을 수 있는 것은 37개다. 그중 36개를 발견했다.
  - 걸러진 5개: 00800의 A/box·B/towel, wcojb4의 C/bag·C/towel·E/cup.
  - 필터를 통과했는데 실패한 1개: wcojb4 C/box(위 "남는 실패").

## SFT 데이터 내보내기 (2026-09-27)

- 도구: `scripts/export_sft_dataset.py`. Teacher 기록 → `<split>/<adapter>.jsonl`(학습용 messages) + `.meta.jsonl`(출처, Teacher 이유) + `manifest.json`.
- 남기는 것: GT 검증 통과, 후보 2개 이상, Teacher 오류 없음. 정답은 Student 출력 형식 `{"selected_id": "C3"}`이다(`llm_policy`와 같음).
- 어댑터:
  - room → `room`
  - workspace·standalone → `search_location`. GRPO의 workspace 어댑터는 이 가중치에서 시작한다(제안서 6-4).
- Student system 프롬프트가 현재 로봇 플래너 프롬프트와 다르면 중단한다. 학습 입력과 실제 입력을 같게 하기 위해서다.
- Train/Val/Test(제안서 6-1), `configs/data/scene_splits.json`:
  - 건물(HM3D 장면) 단위로 나눈다.
  - HM3D 공식 train은 Train, 공식 val(minival 포함)은 장면 ID 해시로 Val/Test로 나눈다.
  - 로컬 장면 4개는 모두 minival이므로 Val/Test 쪽이다. 실제 Train 데이터는 HM3DSem train 장면을 받은 뒤에 생긴다.
- 시범(wcojb4 pilot):
  - 라벨 328개 중 검증 통과 46개, 중복을 빼면 41개(Room 28, Search Location 13).
  - 검증 실패 282개는 Teacher 판단이 GT와 맞지 않는 경우다. 대량 생성 전에 원인 분석과 수율 개선이 필요하다.

## 방 재선택: 방문마다 Stage 1부터 다시 (2026-09-27, 합의)

- 이전 구현: 고른 방의 탐색 위치를 모두 돌기 전에는 다른 방을 고르지 않았다. 제안서에 없는 규칙이다.
- Teacher 시범에서 검증 실패 282개 중 251개가 이 규칙 때문에 생긴 "정답 후보가 없는 상태"였다. 목표가 없는 방 안을 계속 돈 경우다.
- 실물 로봇 프로젝트:

| 프로젝트 | 로봇 | 방식 |
|---|---|---|
| LFG | LoCoBot | 매 스텝 다시 계획(τ = 1). 고른 하위 목표에 도착할 때까지만 유지 |
| Inter-POMDP | RealMan, 여러 방 실내 | 관측마다 믿음 갱신 후 상위 목표 재선택 |
| 개인화 온톨로지 탐색 | Turtlebot | 방 순서 없이 전체 가구를 확률·거리로 순위화 |
| SayPlan | 실제 이동 매니퓰레이터 | 방 노드 펼침·접음을 매 반복 결정 |
| POMDP 탐색(Wandzel, GenMOS) | 실제 로봇 | 관측마다 다시 계획 |

- 변경: `SearchSession(commit_to_room=False)`가 기본값이다. 방문 한 번마다 방 선택(Stage 1)부터 다시 한다. 이전 방식은 `--commit-to-room` 옵션으로 남겨 비교용으로 쓴다.
- 제안서 시나리오 6("찾지 못하면 탐색 상태를 갱신하고 다음 탐색 위치를 결정")과 맞는다.
- 효과(오프라인 Teacher, 장면 3개):
  - 판정 가능한 비율 27~35% → 57~64%.
  - 검증 통과 라벨 29/51/57 → 80/108/143, 약 2.5배.
  - Teacher 호출은 방 단계가 매번 추가되어 약 1.6배로 늘었다.
  - 발견 수와 방문 수는 같다.
- 남은 판정 불가 상태는 대부분 목표가 없는 방 안의 위치 선택이다. 오프라인 Teacher는 가장 가까운 방만 골라 틀린 방에 오래 머문다. 실제 LLM Teacher로 다시 측정해야 한다.

## 현재 방 표시(프롬프트 0.2)와 실제 Teacher 시범 (2026-09-27)

- 방 선택 입력에서 로봇이 있는 방에 `(current room)`을 표시한다. `PlanningContext.current_room`을 쓰고 `PROMPT_VERSION`은 0.2다. 0.1로 만든 Teacher 기록은 SFT export에서 거부된다(학습 입력과 실제 입력을 같게 유지).
- 실제 GPT-4.1 Teacher 시범(`outputs/teacher_v4/pilot_replan_p02_*`): 조합당 시작 위치 1개, 방문 상한 60, 방문마다 방 재선택.

| 장면 | Teacher 질의 | 판정 가능 | 통과 | 방 단계 통과 | 방 유지/전환 | A→B→A |
|---|---|---|---|---|---|---|
| 00800 | 64 | 33 | 13 | 10/28 | 21/15 | 2 |
| 00808 | 59 | 42 | 24 | 17/23 | 23/7 | 3 |
| wcojb4 (크레딧 소진으로 29회 실패, 무효) | 58 | 38 | 16 | 12/33 | 14/7 | 2 |

- 00800+00808 통과율은 30%(123 중 37)다. 이전 시범(방 고정, 프롬프트 0.1)은 14%였다.
- 왔다 갔다 하는 경우는 적다(장면당 2~3회).
- 남은 판정 불가 상태는 대부분 목표가 없는 방 안의 Workspace·Standalone 선택이다.
- OpenAI 계정 크레딧이 소진되어 wcojb4 실행 중 HTTP 429가 났다. 이후 API 사용을 중단했다.

## 방 단계 검증: 목표가 들어 있는 방 (2026-09-27)

- 이전 규칙: 그 방의 탐색 위치(현재 위치에서 가장 가까운 목표 지점)에서 목표가 보이면 정답 방.
  - 문 너머로 보이는 방이 정답이 되고, 목표가 실제로 있는 방은 오답이 될 수 있었다.
  - 예: 00800에서 수건 → 욕실이 오답, 수건 → 부엌이 통과.
- 새 규칙(`GroundTruth(room_rule="containment")`, 기본값): 목표 인스턴스가 들어 있는 방(HM3DSem region = 제안서의 GT Room)이면서 아직 갈 수 있는 탐색 위치가 남은 방.
  - Workspace·Standalone 단계는 지금처럼 "그 위치에서 보이는가"로 검증한다.
- 근거:
  - HOV-SG(Spot): 물체를 GT 방 배치에 배정해 방 단위 질의를 평가한다.
  - MoMa-LLM(Toyota HSR): 물체를 위치로 방에 배정하고, 탐색 성공은 "목표를 관측했는가"로 판정한다.
- 시범 재채점(00800): 방 단계 통과 10 → 12.
  - 수건 → 욕실 4개가 통과로 바뀌었다.
  - 수건 → 부엌·거실 2개가 실패로 바뀌었다.
  - 00808은 변화 없음.
- 이전 규칙은 `--room-verification observation`으로 남겨 비교용으로 쓴다.

## 로봇 NavMesh: 단차·경사 한계 (2026-09-27)

- 이전: Habitat 기본(사람 기준) 값이었다. 오를 수 있는 단차 0.2 m, 경사 45°, 높이 해상도 0.2 m. 계단은 뒤에서 층 분리 규칙으로 잘라냈다.
  - 이 규칙이 계단 발치의 평평한 통로까지 자르는 문제가 있었다. 예: 00808 B–E가 잘못 분리됨.
- 과제 조건: 실제 환경에 경사진 공간이 없고 평평한 바닥만 다닌다.
- 설정(`configs/robot/demo.yaml`):

| 항목 | 값 | 근거 |
|---|---|---|
| `max_step_m` | 0.02 | ADA 2010 404.2.5의 기존 문턱 상한 3/4 in(19 mm). HM3D 바닥을 스캔 잡음으로 쪼개지 않는 최소값(0.013은 00800 바닥을 쪼갬) |
| `max_slope_deg` | 20 | 문턱 부분의 가파른 면이 바닥을 쪼개지 않는 최소값(10°·15°는 wcojb4 1층을 쪼갬). ADA 문턱 경사 허용 1:2(26.6°) 이내 |

- 높이 해상도는 단차의 절반(0.01 m)으로 둔다.
- 계단은 단차 한계 때문에 경사 30°에서도 연결되지 않는다(높이 폭 0.3 m가 넘는 영역이 0개).
- 이렇게 만든 NavMesh는 로봇이 못 가는 곳을 스스로 뺀다. 그래서 층 분리 규칙 대신 연결 조각을 그대로 영역으로 쓴다(`segment_navmesh_levels.py --mode islands`, 실행기가 로봇 설정을 보고 자동 선택).
- 결과(v4):
  - 00808: B·E가 하나로 합쳐짐(28 m²). 0.3 m 단차가 있는 C·D는 분리 유지.
  - wcojb4: 1층 문턱 분리가 해소됨(26.9 m²), 경로가 있는 Workspace 25 → 36/47.
  - 00800: 5 cm가 넘는 단차가 있는 부분이 빠져 면적 46.6 → 40.3 m²(로봇이 오를 수 없는 곳).
  - 찾을 수 있는 조합 37/40, 그중 36개 발견(방문 제한 없음).
- 이전 Teacher 시범(`outputs/teacher_v4/pilot_replan_p02_*`)은 바뀌기 전 영역으로 만든 것이다.

## Workspace 단계 검증과 재검증 도구, 새 시범 (2026-09-27)

- Workspace 단계 정답(`GroundTruth(workspace_rule="observation")`, 기본값): GT Workspace(목표가 그 위에 있음)이거나, 그 Workspace 목표 지점에서 목표가 보이는 것.
  - 근거: 탐색 성공 = 목표 관측(MoMa-LLM, Habitat ObjectNav). Standalone 단계와 같은 규칙이다.
  - 이전 규칙에서는 목표가 바닥 등에 따로 놓이면(Standalone 목표) 어떤 Workspace도 정답이 될 수 없었다.
- `scripts/reverify_teacher_records.py`: 기록된 상태(위치, 방문 목록)로 현재 규칙에 따라 검증만 다시 한다. Teacher 궤적과 선택은 그대로 두므로 API 호출이 없다.
- 새 시범(`outputs/teacher_v4/pilot_p02_flat_*`, 바뀐 영역, 프롬프트 0.2):
  - 질의 142, 판정 가능 91(64%), 통과 47.
  - 재검증 후 통과 59(42%). 이전 시범은 14%였다.
  - 방 왕복은 장면당 0~1회.
- SFT 내보내기 시험: `outputs/sft_v4_pilot`. 로컬 장면은 모두 공식 val이라 Val/Test로만 나뉜다.

## 방 선택의 음의 증거: 관측 면적 기반 베이즈 갱신 (2026-09-27)

- 문제: 이미 찾아보고 실패한 방을 계속 다시 골랐다(wcojb4에서 같은 방 최대 5~6회 연속). 방 선택 입력에 "얼마나 찾아봤나"가 없었고, Teacher 확률도 고정되어 있었다.
- 틀: 베이즈 최적 탐색 이론(Koopman 1946, Stone 1975).
  - 기존 구성 요소: 사전 확률 = LLM, 비용 = A* + λ_s, 순서 = 확률 ÷ 비용, 매 방문 후 재계획.
  - 빠져 있던 요소: "찾아봤는데 없으면 확률을 낮춘다".
- 구현:
  - `athome.search.coverage.RoomCoverage`: 방 바닥(시맨틱 메쉬 바닥 삼각형)을 20 cm 간격으로 표본화한다. 방문 지점에서 목표 발견과 같은 관측 모델(3 m, 벽 가시선)로 본 점의 비율을 방별로 누적한다.
  - Student 입력: 방 후보마다 `Observed: N% of this room, target not seen`(프롬프트 0.4). MoMa-LLM이 미탐색 영역과 행동 기록을 넣는 것과 같은 역할이다.
  - Teacher 라벨: 방 확률 p × (1 − 관측 비율)(`after_search`).
  - Teacher 입력에서는 관측 줄을 뺀다(Teacher 프롬프트 0.4). 0.3에서 LLM이 스스로 확률을 낮춰 이중으로 할인된 것을 막기 위해서다.
- 방문한 위치 수를 쓴 0.3은 효과가 없었다. 방당 위치가 13~40곳이라 방문 한 번에 할인이 1/13에 그쳤다. 실제로는 한 번 방문하면 방 바닥의 97~100%를 본다.
- 결과(실제 GPT-4.1 Teacher, 장면 3개, 조합당 시작 1개):

| 장면 | 0.2 통과율 | 0.4 통과율 | 0.2 → 0.4 호출 | 방 단계 |
|---|---|---|---|---|
| wcojb4 | 44% | **60%** | 70 → 53 | 20/43 → 21/34 |
| 00808 | 47% | 47% | 36 → 38 | 9/10 → 10/11 |
| 00800 | 31% | 27% | 36 → 41 | 7/11 → 7/14 |

  - wcojb4는 같은 방 최대 연속 선택이 5 → 3으로, 평균 방문 수가 2.2 → 1.8로 줄었다.
  - 00800은 에피소드 8개라 변동이 크다.
- 실제 로봇 적용: 방 분할 지도(제안서 4-1)의 방별 셀과 점유 지도 가시선으로 같은 `RoomCoverage`를 구성하면 된다. 아직 연결하지 않았다.

### 실제 로봇 연결 (2026-09-27)

- `scripts/build_scene_graph.py`: 방 분할 결과를 `<graph>.rooms.npz`(labels, origin, resolution)로도 저장한다.
- `search_server`: 시작할 때 방 지도와 원본 점유 지도(반경 0)를 읽는다. 명령마다 `RoomCoverage`를 새로 만들어 `SearchSession(coverage=...)`에 넘긴다.
  - 관측 판정: 점유 셀이 광선을 가리지 않고 `search.observation_range_m`(3 m, D435i 권장 거리) 안이면 관측.
  - 방 지도나 거리 설정이 없으면 경고를 남기고 방문 수 방식으로 대체한다. 이 경우 학습 입력과 다르다.
- 알려진 차이: 데이터셋 관측 모델은 벽·문만 가림으로 본다(렌더링 대비 일치 92.6%로 검증). 로봇의 점유 지도는 가구도 가림으로 보므로 관측 비율이 보수적(낮게)으로 나온다. 실제 로봇 데이터로 카메라 관측과 비교해 확인할 항목이다.

## 대량 실행 준비 (2026-09-27)

- **API 호출 안정성**(`athome.inference.chat.ChatJSON`, OpenAI 권장 방식):
  - 일시적 오류(429 속도 제한, 5xx, 연결 실패)는 지수 백오프로 재시도한다. Teacher는 5회.
  - 크레딧·할당량 소진(`insufficient_quota`)은 재시도하지 않고 `QuotaExceeded`로 멈춘다. Teacher 스크립트는 그때까지 모은 결과를 저장하고 `aborted`에 기록한다.
- **실제 사용량**: 호출마다 API가 알려 주는 입력·출력·캐시 토큰을 기록한다(`QueryRecord.usage`). 요약의 `api_usage`에 합계와 비용을 남긴다. 요금은 `configs/data/llm_pricing.json`에 두며 출처와 확인 날짜를 적는다.
- **대체 진행 제외**: Teacher 호출이 실패하면 그 에피소드는 대체 정책으로 이어진다. 실패한 단계부터는 Teacher 궤적이 아니므로 SFT 내보내기에서 뺀다(`after_teacher_fallback`).
- **실행기**: `run_scene_pipeline.py`에 두 단계를 추가했다.
  - `start_states`: 평가·GRPO용 고정 시작 상태
  - `teacher`: `--teacher openai --submit`, `--teacher-starts`, `--teacher-max-steps`
- **일괄 실행**: `scripts/run_dataset.py`.
  - 주석이 있는 장면을 장면마다 별도 프로세스로 돌린다. 실패해도 다음 장면으로 넘어간다.
  - 상태와 로그를 `outputs/hm3d_<version>/dataset_status.<split>.json`, `logs/`에 남긴다.
  - `--limit`으로 1차(비용 확인) 실행을 할 수 있다.

## 개방 어휘 목표 범주 (2026-09-27, 합의)

- 근거:
  - HM3D-OVON(같은 HM3DSem): 379개 범주를 쓰고, 학습에서 뺀 범주로 따로 평가한다.
  - HomeRobot OVMM(실제 Stretch): 129개 범주를 쓰고, 처음 보는 범주를 평가한다.
  - 예전 ObjectNav는 6~21개 고정 범주였다.
- `scripts/build_target_categories.py` (train 장면의 scoped 물체 목록으로 만든다):
  1. 공식 HM3DSem 매핑으로 이름을 정규화한다(오타 통합).
  2. 라벨이 있는 모든 범주를 후보로 둔다(mpcat40 "unlabeled"만 제외). 분류 부분집합이나 크기 기준은 손으로 정하지 않는다(2026-09-27 변경: 1 m 크기 기준·휴대 분류 제한 삭제. 임의 기준이고 담요·자전거·옷걸이 등을 잘못 뺐다).
  3. 복수형을 단수형으로 합친다.
  4. train 장면 3곳 이상에 나오는 범주만 남긴다.
  5. GPT-4.1이 9회 투표로 "사람이 로봇에게 찾아 달라고 할 옮길 수 있는 물건인가"를 판정한다(설비·가구 제외, 범주 이름당 한 번). "찾을 만한 물건인가"는 이 의미 판단 하나로 정한다.
  6. 25%를 평가 전용으로 떼어 두고, HM3D-OVON(논문 III-B절, 표 II)대로 SentenceBERT 코사인 유사도로 나눈다: 학습 범주와 0.68 초과면 "동의어 평가"(couch→sofa), 아니면 "처음 보는 범주". OVON은 모델 이름을 밝히지 않아 sentence-transformers 권장 범용 모델 all-mpnet-base-v2를 쓴다.
  - 뜻이 비슷한 범주(toy / plush toy 등)는 OVON처럼 합치지 않는다. 정답 판정은 범주 이름 그대로다.
  - "실제로 보이는가"는 범주 단계가 아니라 에피소드 단계의 findable 시작 상태 필터가 맡는다(HM3D-OVON의 가시성 기준에 해당).
- `scripts/select_scene_targets.py`: 장면마다 목록 범주 중 그 장면에 있는 것에서 K개(기본 12)를 뽑는다(시드 고정). Train은 학습 범주만, Val/Test는 처음 보는 범주까지 포함한다.
- 실행기: `--target-categories <목록> --targets-per-scene 12`. 목표 목록 작성 전에 `scene_targets` 단계를 둔다. 분할(Train/Val/Test)은 `configs/data/scene_splits.json` 규칙으로 정한다.
- 진행 순서:
  1. 1차 실행(`--submit` 없음): 모든 train 장면을 씬그래프 라벨링 제출 직전까지 돌린다.
  2. 범주 목록을 만든다.
  3. 2차 실행(`--submit --teacher openai --target-categories ...`).
- minival 시험(범주 기준을 장면 1곳 이상으로 낮춤): 94개 범주. 복수형 5건 통합. LLM이 설비·가전 22개를 제외했다(수도꼭지, 스위치, 화재경보기, 수건걸이, 그림, 전자레인지 등). 판정 비용 약 0.09달러.
- minival 재시험(크기·분류 기준 삭제 후, 범주 기준 장면 2곳 이상): 후보 97개 → LLM이 48개 제외(문, 벽, 의자, 소파, 싱크대, 냉장고, 수도꼭지, 천장등 등) → 49개. 담요·자전거·옷걸이·가방·청소기·사다리가 남았다. 경계 사례(표 5~7/9): 벽등, 게시판, 러그, 칼꽂이, 샤워 커튼, 조각상.
- **train 145개 장면 결과(v2, `configs/data/target_categories.v2.json`)**: 314개 범주(학습 236, 동의어 평가 37, 처음 보는 범주 41). 드묾 623개, 찾을 물건 아님 360개 제외. 투표 비용 약 0.53달러.

### HM3DSem 원본 데이터 이상 처리 (2026-09-27)
train 1차 실행에서 145개 중 10개가 원본 데이터 이상으로 멈췄다. 공식 로더(habitat-sim `HM3DSemanticScene.cpp`)와 같게 처리한다.
- 같은 색을 가진 인스턴스(7개 장면, 19개): 공식 로더는 `insert`를 써서 파일에서 먼저 나온 인스턴스가 색을 가진다. 나머지는 메쉬가 없는 물체로 기록(`shadowed_ids`).
- 짧은 16진수 색 문자열(`c`): 공식 로더처럼 `std::stoul(.., 16)`으로 읽는다.
- 메쉬 없는 노드(조명·카메라)의 변환: glTF 규격상 메쉬 위치에 영향이 없어 허용한다.
- 부피가 0인 메쉬 박스(삼각형 1개짜리 주석 조각, 3개): 물체가 아니므로 삼각형 없는 물체와 같이 `bbox: null`.
- 결과: 145개 장면 모두 라벨링 제출 단계까지 통과.

## 관측 방향 6방향 (2026-09-28, 합의)
- 내비게이션 점검에서 D435i RGB 시야 69.4° × 4방향 = 278°로 사각지대가 있음이 확인됐다.
- VLFM(Spot 실물: 시작 시 한 바퀴 회전), Gervet et al.(Stretch·D435i: 42° 시야에 30° 회전), Habitat ObjectNav(79° 시야, 30° 회전)처럼 회전 간격을 시야보다 작게 해 한 바퀴를 빈틈없이 덮는다.
- `VisitConfig.heading_count` 4 → 6(60° 간격). 학습 관측 모델의 360° 원판 가정이 실제와 같아지므로 라벨은 그대로 유효하다. 로봇 회전 yaw 오차가 4.7°보다 크면 7방향으로 올린다(내비게이션 쪽 확인 중).
- 내비게이션 쪽 회신: `docs/navigation_alignment_ko.md`.

### 목표 범주 v4 (2026-09-28)
- 10개 장면 라벨 점검에서 뜻이 넓거나 모호한 목표(appliance, container, device, clutter, chest 등)가 라벨 품질을 떨어뜨리는 것을 확인했다.
- 투표 질문에 "이름을 대고 찾을 구체적인 물건인가(포괄적·모호한 이름 제외)"를 더해 다시 투표했다(v3, 약 0.54달러).
- 온도 1 투표의 경계 흔들림으로 v3에만 들어온 8개(curtain, mirror 등)가 있었다. 그래서 두 질문을 **모두** 다수결로 통과한 범주만 남긴다(v2 ∩ v3, API 추가 없음) → **275개**.
- 평가 전용 범주는 v2에서 뽑은 것을 유지한다(`--held-out-from`). 빼는 기준이 무작위 뽑기와 무관해 남은 집합도 무작위 표본이다. 이미 라벨링한 10개 장면의 split과도 일치한다. → 학습 209 / 동의어 평가 31 / 처음 보는 범주 35.
- `export_sft_dataset.py --categories`: Train은 학습 범주만, Val/Test는 목록 전체. 10개 장면 통과 라벨 609개 중 533개 유지(빠진 범주 76개 제외).
- 파일: `configs/data/target_categories.v4.json` (v2·v3는 투표 기록으로 보존, 재추출한 split은 `superseded/`).

### 하위 종류와 프롬프트 0.5 (2026-09-28)
- 9월 26일 합의한 별칭 규칙(table lamp → lamp)이 개방 어휘로 바꾸면서 빠져 있었다. train 장면의 (장면, 목표) 조합 7%에서 하위 종류가 함께 있었다.
- `scripts/build_target_hyponyms.py`: 이름이 "수식어 + 목록 범주"인 범주 쌍(62쌍)을 GPT-4.1 9회 다수결로 "X는 Y의 한 종류인가" 판정 → 55쌍 인정(거부 예: paper towel → towel, computer mouse → mouse). 비용 약 0.06달러. `configs/data/target_categories.v5.json`의 `hyponyms`.
- 장면별 목표 파일의 `target_members`: 하위 종류 인스턴스도 부모 목표의 정답(catalog)이고 함께 가린다. 인스턴스 하나가 여러 목표에 속할 수 있다.
- 프롬프트 0.5(Student·Teacher 공통 입력): 반복 가구·물체를 개수로 표시(MoMa-LLM). 거리는 숫자 유지(라벨이 정밀한 거리에 의존, Student는 학습으로 숫자를 배움). Teacher 이유의 Student 학습 활용(Distilling step-by-step)은 SFT 단계에서 결정.
- 앞선 10개 장면은 목표 선택부터 다시 돌려 145개 장면이 같은 규칙을 쓰게 했다(이전 결과는 `outputs/superseded/targets_v2/`).

### SFT 후보 순서 섞기 (2026-09-28)
- 후보가 이동 거리순이라 Student가 "C1 고르기" 지름길을 배울 수 있다. 실제로 탐색 위치 예제의 51%가 정답 C1이었다(무작위 기대 32%).
- RankVicuna(Pradeep et al. 2023, GPT 순위 판단의 소형 모델 증류)처럼 기록된 순서의 예제는 유지하고, 후보 순서를 섞은 복사본을 추가한다(Train만, `--shuffled-copies 1`, 시드 고정). Val/Test는 로봇이 보는 기록 순서 그대로.
- 모든 예제는 기록(후보·상황)으로 프롬프트를 다시 만들어 저장된 Student 프롬프트와 같은지 검사한 뒤 섞는다. `current_room`은 예전 기록에 없어서 "(current room)" 표시로 복원한다.
- 결과(3개 장면 시험): 정답 C1 비율 탐색 위치 51% → 합친 데이터 39%(기대 31%), 방 29%(기대 33%).
- 남은 일: 라벨링이 끝난 뒤 `teacher._context_dict`에 current_room 추가, Teacher 순서 편향 측정(기록 100개 재질의, 약 0.3달러, 실행 전 확인).

## 전체 라벨링 결과 검증 (2026-09-29)
- 145개 장면, 에피소드 3,958개(발견 98.9%), SFT 7,244개(기록 순서) + 섞은 복사본 → `outputs/sft_v5` 13,525개. 비용 약 100.7달러.
- 무결성: 정답 별칭 오류 0, 평가 전용 범주 섞임 0, 가림 실패(같은 이름) 0, 재구성 불일치 0. 정답 C1 비율이 무작위 기대치 수준(방 26%/24%, 탐색 위치 34%/30%).
- 한 번에 맞힌 비율(정답 있는 질의): 방 Teacher 45% vs 가장 가까운 곳 29% vs 무작위 26%. 방 안 단계는 360°·3 m 관측이라 무작위도 69%가 "정답"이라 이 지표로는 구별되지 않음.
- 에피소드 효율(같은 시작 칸 짝 비교, `scripts/compare_teacher_baselines.py`, `outputs/baseline_compare_v5`): 제안서 보상 비용 D + λ_s·N(λ_s=3) 평균 Teacher 24.6 / 가장 가까운 곳 56.8 / 무작위 52.6, 중앙값 9.6 / 15.6 / 17.0. Teacher 승 1,892 · 동률 1,024 · 패 1,042. SPL은 Teacher 0.406, 가장 가까운 곳 0.395, 무작위 0.272. 가장 가까운 곳은 방문이 4배(14.1 vs 3.4회).
- 발견한 문제: (1) 시작 위치 필터 `findable_from`이 탐색 위치의 아무 goal에서나 보이면 가능으로 봄(로봇은 가장 가까운 goal 하나만 씀) → 에피소드 0.6%가 실제로는 찾을 수 없음. 라벨 판정(`verify.py`)은 실제 goal 기준이라 영향 없음. 평가·GRPO 시작 상태는 고칠 것. (2) 단독 물체 후보가 최대 약 500개(창고 등) → Teacher가 점수를 빠뜨린 질의 272건(이후 기록 1,402개 제외), 로봇 추론 지연 우려. (3) 제안서의 LoRA 적용 위치(q,k,v,o)는 LoRA Without Regret의 "모든 선형 층" 권고와 다름 → SFT 전에 결정.

## 단독 물체 후보 종류별 묶기, LoRA 모든 선형 층 (2026-09-29, 합의)
- 단독 물체 후보가 HM3D의 촘촘한 주석 때문에 방마다 최대 약 500개(창고의 상자 44개 등)였다. MoMa-LLM(실물 로봇)처럼 방 안 물체를 종류별 개수로 보여 주고 종류를 고르게 한다: 후보 = "storage box x44"(가장 가까운 것까지의 거리), 로봇은 그 종류 중 가장 가까운 열린 물체로 간다(`SearchSession(group_standalone=True)`, `standalone_groups`). 프롬프트 0.6.
- 제안서의 단계 구조(방 → 워크스페이스 → 단독 물체)는 그대로이고, 단독 물체 단계의 후보 표현만 물체 하나 → 물체 종류로 바뀐다. 로봇의 추론 입력 한도(vLLM 4,096토큰) 안에 들어온다.
- 기존 기록 변환(`scripts/regroup_standalone_records.py`, API 없음): 종류 확률 = 그 종류 물체들의 Teacher 확률 최댓값, 라벨 = Teacher 규칙(`search_index_choice`) 그대로, 판정 = 가장 가까운 물체에서 목표가 보이는지(`GroundTruth`). 판정 재계산은 기존 기록과 72/72 일치. 결과: 단독 후보 최대 496 → 52, 종류 하나만 남아 선택이 필요 없어진 질의 413개, 라벨의 종류가 바뀐 질의 16개.
- 새 SFT: `outputs/sft_v5g` (방 8,546 / 탐색 위치 4,992, 섞은 복사본 포함).
- LoRA: `configs/training/sft_lora.yaml`, `scripts/train_sft_lora.py`. 모든 선형 층(q,k,v,o,gate,up,down), rank 16(vLLM 서버와 같음), 학습률 2e-4(전체 미세조정의 약 10배, LoRA Without Regret), 답 토큰에만 손실, 추론과 같은 형식(Qwen3, thinking 끔). 건물 5%(7개)를 검증 손실용으로 분리. 제안서의 q,k,v,o에서 변경.
- 남은 것: 점수 누락으로 빠진 1,402개 기록은 후보가 줄었으니 다시 라벨링하면 살릴 수 있다(API 비용, 필요 시 결정).
