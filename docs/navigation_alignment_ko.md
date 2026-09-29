# 로봇–학습 데이터 정합 회신 (데이터셋 담당 → 내비게이션 시스템)

작성: 2026-09-28. 내비게이션 쪽 점검 결과(어긋나는 부분 5곳 + 실측 필요 3곳)에 대한 회신입니다.
원칙: 학습 데이터는 제안서와 이미 합의된 규칙을 기준으로 두고, 로봇이 같은 규칙을 따르도록 맞춥니다.
학습 데이터 내용이 바뀌는 항목은 없습니다(1번은 로봇 동작을 데이터셋 가정에 맞추는 쪽으로 정했습니다).

## 요약

| # | 항목 | 결론 | 할 쪽 |
|---|---|---|---|
| 1 | 카메라 시야 360° 가정 | **관측 방향 4 → 6 (60° 간격)**. 공용 `VisitConfig.heading_count` 변경 완료. 회전 정확도 확인 필요 | 데이터셋(완료) + 내비게이션(확인) |
| 2 | 높이 1.5 m 필터가 로봇에서 꺼짐 | 로봇 그래프에 방 바닥 높이 `floor_z_m`을 넣는다 | 내비게이션/그래프 생성 |
| 3 | GRPO·평가에서 방 관측 비율 누락 | 데이터셋 버그, **수정 완료** | 데이터셋(완료) |
| 4 | 수납장 안 물체 제외 규칙 없음 | 로봇 그래프 생성에 같은 규칙(`scene_scope.json`) 적용 | 내비게이션/그래프 생성 |
| 5 | 어휘 차이 | 인지 라벨을 학습 어휘로 정규화(검출 프롬프트를 학습 범주명으로), 하위 종류도 목표로 인정 | 인지/그래프 생성 |

## 1. 관측 방향: 4 → 6

**문제**: D435i RGB 수평 시야 69.4° × 4방향 = 278°. 학습 관측 모델(`configs/data/symbolic_observation.json`)과 로봇의 방 관측 비율은 360° 원판을 가정합니다.

**다른 프로젝트 (논문 원문 확인)**
- VLFM (Spot 실물, arXiv 2312.03275): 시작 시 "rotates in place for a complete turn". 본 영역은 카메라 시야 부채꼴로 기록.
- Gervet et al., Navigating to Objects in the Real World (Stretch 실물, D435i, arXiv 2212.00922): 세로 장착으로 수평 시야 42°, 30°씩 회전, 지도에는 시야 안만 기록.
- Habitat ObjectNav 표준 (ZSON 등): 시야 79°, 30° 회전.
- 공통 원칙: (1) 회전 간격 < 카메라 시야여서 한 바퀴가 빈틈없이 덮인다. (2) "본 곳"은 실제 시야 기준으로 기록한다.

참고: 예전 관측 모델 검증에서 4방향 렌더링과 12방향 렌더링의 일치율은 93.9%였습니다. 사각지대 때문에 판정이 약 6% 달라졌던 셈입니다.

**결정**: 60° 간격 6방향. 시야 69.4°가 방향마다 9.4° 겹쳐 360°를 덮으므로 원판 가정이 실제와 같아집니다. 12방향 렌더링으로 한 관측 모델 검증(92.6% 일치)도 그대로 유효하고, 학습 라벨은 다시 만들 필요가 없습니다.

**적용된 변경 (공용 코드)**
- `src/athome/execution/visit.py`: `VisitConfig.heading_count` 4 → 6. 로봇 `VisitExecutor`(ROS `search_server.py`, `run_visit.py`)와 데이터셋이 같은 값을 씁니다.
- 방문당 시간 모델(`symbolic/environment.py`): 한 바퀴 회전 + 6 × observation_window(1.0 s). 평가 시간 예산에만 영향이 있습니다.

**내비게이션 쪽 확인 요청**
- 방향 간 겹침이 한쪽 4.7°뿐이라, 제자리 회전의 yaw 오차가 이보다 작아야 틈이 생기지 않습니다. Nav2 회전 동작의 yaw 허용 오차(예: goal checker `yaw_goal_tolerance`, spin 동작의 정밀도)를 알려 주세요.
  - 4.7° 이하: 6방향 그대로.
  - 더 크면: 7방향(51.4° 간격, 겹침 한쪽 9°) 또는 VLFM처럼 한 바퀴를 연속으로 돌며 검출. 어느 쪽이든 원판 가정은 유지됩니다.
- 검출에 쓰는 카메라가 RGB(69.4°)인지 깊이(87°)인지도 확인 부탁합니다. 계산은 RGB 기준입니다.

## 2. 높이 1.5 m 필터: 로봇 그래프에 방 바닥 높이 제공

**규칙 (학습과 동일)**: `SceneGraph`는 방 바닥에서 1.5 m 넘게 떠 있는 물체(`MAX_BASE_ABOVE_FLOOR_M = 1.5`, `src/athome/scene_graph/query.py`)를 Standalone 탐색 위치와 Known 판정에서 뺍니다. 바닥 높이는 방의 `floor` 물체 bbox 중심 z의 중앙값에서 얻고, `room.floor_z_m`이 있으면 그 값을 우선합니다. 둘 다 없으면 필터가 꺼집니다(현재 로봇 상태).

**요청**
- 실제 그래프 생성(`src/athome/scene_graph/pipeline.py`)에서 방마다 `floor_z_m`을 넣어 주세요. 단층 데모라면 map 좌표계에서의 바닥 높이 하나를 모든 방에 넣으면 됩니다(FAST-LIO2 지도의 바닥 평면 높이).
- 샘플 z가 음수인 것은 map 원점이 바닥이 아니라 센서 높이 등일 가능성이 큽니다. 원점과 관계없이 `floor_z_m`을 그 좌표계의 바닥 높이로 주면 필터가 맞게 동작합니다.

## 3. GRPO·평가의 방 관측 비율: 수정 완료

- `src/athome/training/grpo.py` `rollout(..., coverage=None)` 추가, `scripts/grpo_rollouts.py`에서 rollout마다 새 `problem.coverage()`를 넘김.
- `scripts/evaluate_policies.py`도 `coverage=problem.coverage()`를 넘김.
- 이제 Teacher, GRPO, 평가, 로봇 모두 방 단계 입력이 `Observed: N% of this room` 형식입니다. 테스트를 추가했습니다(`tests/test_search_session.py`).

## 4. 수납장 안 물체: 같은 규칙을 로봇 그래프에 적용

**규칙 (2026-09-26 합의, `configs/data/scene_scope.json`)**: 수납 가구(선반, 수납장, 책장, 옷장 등 목록) 안에 든 물체는 프로젝트 범위 밖입니다. 판정은 기하 규칙입니다.
- 물체 바닥 면적의 90% 이상이 수납 가구의 바닥 면적 안에 있고,
- 물체 윗면이 수납 가구 윗면보다 5 cm 이상 아래(수직 허용 오차 2 cm)면 "안에 있음".

**요청**: 로봇 그래프 생성에 같은 단계를 넣어 주세요. 판정 함수 `athome.data.hm3d.scope.apply_storage_scope(room_object_map, scope)`(내부 `find_contained`)를 그대로 쓰면 학습과 기준이 일치합니다. 입력은 z-up 물체 목록(`semantic_tag`, `bbox`)입니다. 그래프에 들어가기 전에 빼면 `Objects on it:` 목록과 Known 판정이 모두 학습과 같아집니다.

## 5. 어휘: 인지 라벨을 학습 어휘로 정규화

**학습 어휘**
- 이름 정규화: HM3DSem 공식 매핑 `configs/data/hm3dsem_category_mappings.tsv`(raw → category → mpcat40), 복수형은 단수형으로 합침.
- 목표 범주 275개: `configs/data/target_categories.v5.json` (학습 209 / 동의어 평가 31 / 처음 보는 범주 35, HM3D-OVON 방식, 하위 종류 목록 포함).
- 탐색 위치 제외 정책: `configs/data/search_locations.json` (mpcat40 기준).

**요청**
- 가장 확실한 방법은 개방 어휘 검출기의 프롬프트를 **학습 범주 이름 그대로**(목표 범주 + 탐색 위치 범주) 주는 것입니다. 그러면 라벨이 처음부터 학습 어휘 안에 있습니다.
- 다른 어휘를 쓰는 검출기라면, 그 라벨 목록 → HM3DSem category 매핑 표를 한 번 만들어 그래프 생성과 발견 판정에 같이 적용해 주세요.
- 발견 판정의 라벨 비교도 같은 정규화(매핑 + 복수형 → 단수형) 뒤에 해야 합니다. 목표 이름에는 띄어쓰기가 있습니다("paper towel").
- **하위 종류도 목표로 인정**: 목표 "lamp"는 bedside lamp, table lamp 등도 찾은 것으로 봅니다. 목록은 `configs/data/target_categories.v5.json`의 `hyponyms`(GPT-4.1 9회 다수결로 판정, 55쌍)입니다. 학습의 정답·가림이 이 규칙을 따르므로, 로봇의 발견 판정과 Known 판정(`SceneGraph.known_locations`)도 목표 이름 + 그 하위 종류로 비교해 주세요.

## 실측이 필요한 차이 (지금 라벨링을 막지는 않음)

- **방 관측 비율의 가림 기준**: 로봇은 occupancy(가구 포함), 데이터셋은 벽·문만. 분모도 다름. 실제 지도로 방문 후 관측 비율을 비교해 봐야 합니다.
- **방 분할**: 로봇 watershed vs 학습 HM3D 방 주석. 데모 지도에서 방 개수와 크기를 비교해 주세요.
- **발견 판정**: 로봇은 검출 + `is_static` + 라벨 일치. 라벨 일치는 5번 정규화로 맞춥니다.

## 변경 파일 (데이터셋 쪽)

- `src/athome/execution/visit.py` (heading_count 6), `src/athome/symbolic/wall_los.py`·`docs/symbolic_review_ko.md` (설명 문구)
- `configs/data/symbolic_observation.json` (근거 문구: 6방향)
- `src/athome/training/grpo.py`, `scripts/grpo_rollouts.py`, `scripts/evaluate_policies.py` (방 관측 비율)
- `tests/test_visit_executor.py` (6방향), `tests/test_search_session.py` (GRPO 관측 비율) — 전체 153개 테스트 통과

## 프롬프트 변경 알림 (0.4 → 0.5)

- 같은 가구·물체가 여러 개면 한 번만 쓰고 개수를 붙입니다(예: `Workspaces: shelf (general_storage) x5, table`). MoMa-LLM(실물 로봇)이 방의 같은 노드를 개수로 요약하는 방식을 따랐습니다.
- 로봇 `LLMPolicy`도 같은 `build_messages`를 쓰므로 코드 변경은 필요 없습니다. 다만 로봇 그래프가 반복 물체를 한 개로 합치지 않고 그대로 넘겨야 개수가 맞게 나옵니다.

## 프롬프트 0.6: 단독 물체 후보를 종류별로 묶음 (2026-09-29)

- `SearchSession(group_standalone=True)`(기본값): 단독 물체 단계의 후보가 물체 하나가 아니라 **물체 종류**입니다. 후보 ID는 `standalone_group:<room>:<category>`이고, 정보는 종류, 개수, 가장 가까운 물체입니다.
- 플래너가 종류를 고르면 세션이 그 종류 중 **가장 가까운 열린 물체**를 방문 위치로 정합니다. `SearchDecision.location_id`는 실제 물체 ID 그대로라 방문 실행 쪽은 바뀌지 않습니다.
- 프롬프트 예: `Object: storage box x44` / `A* path cost: 2.1 m (nearest)`.
- 이유: HM3D 기준 방당 단독 후보가 최대 약 500개였고, 이는 vLLM 입력 한도 4,096토큰을 넘습니다. 묶으면 최대 52개입니다. MoMa-LLM(실물 로봇)이 물체를 종류별 개수로 보여 주고 이름으로 이동하는 방식과 같습니다.
- 로봇 쪽 코드 변경은 필요 없습니다(공용 `SearchSession`). 다만 인지 라벨 정규화(5번)가 되어야 같은 종류가 한 후보로 묶입니다.

