# HM3D Scene Graph 파이프라인 v3

v3 = semantic mesh bbox(v2 규칙) + 수납가구 내부 객체 범위 제외 + 라벨 프롬프트 v2. 기존 `outputs/hm3d/`, `outputs/teacher/`, `outputs/hm3d_meshbbox_v2/`는 변경하지 않는다.

## 결정 사항 (2026-09-26)

- bbox: semantic mesh 기반(`docs/mesh_bbox_ko.md`). Habitat OBB AABB는 객체 약 8%에서 mesh를 5 cm 넘게 놓친다.
- 범위: 선반·책장·수납장 안쪽 단의 객체는 이번 프로젝트에서 다루지 않는다. 실제 로봇 그래프에 없을 객체가 학습 그래프에 들어가지 않도록 그래프와 목표 GT에서 제외한다(`configs/data/scene_scope.json`, `src/athome/data/hm3d/scope.py`). 컨테이너는 개방형 수납가구 범주로 한정한다. 소파·침대·계단은 쿠션 등을 잘못 "포함"하기 때문이다.
- 라벨 프롬프트 v2: 수납가구도 다른 가구와 같이 윗면으로만 Source 여부를 판단하고 내부는 범위 밖임을 명시한다. v1의 "선반 단을 추론하지 말라"만으로는 선반을 Source로 볼지가 정해지지 않았다. 측정 결과 선반들이 한 방에서 한꺼번에 선택되거나 빠졌고, 자기 입력이 같은 선반의 선택률이 방 문맥 변화로 100%에서 30%로 떨어졌다(`outputs/teacher_meshbbox_v2/label_stability/`).
- 라벨링 프로토콜 v2: 프롬프트 v2 + 9회 샘플링(temperature 1) 다수결(5회 이상 선택된 Source 채택, 방·기능 라벨은 최빈값, 동률은 결정적으로 깨고 표시, 득표는 라벨의 `votes`에 기록). `src/athome/scene_graph/label_voting.py`. 근거: v2 프롬프트 측정에서 방 라벨은 안정했으나 Source 65개 중 10개가 선택률 20~80%였고 temperature 0 단일 답이 소수 의견인 경우가 2건(`sink_547` 8/10 선택인데 단일 답은 제외). 실제 로봇 라벨링(`label_rooms`, `ChatJSON.sample`)도 같은 프로토콜을 쓴다. 지도당 1회 오프라인이며 `n`은 요청 하나로 처리되어 입력은 한 번만 과금된다.
- 프롬프트·compact 입력·모델 버전(`gpt-4.1-2025-04-14`)은 `src/athome/scene_graph/semantic_labeling.py` 하나에서 정의한다. HM3D batch(`compact_semantic_batch.py`)와 실제 로봇 라벨링이 같은 정의를 쓴다. v1은 기존 산출물 재현용이다(바이트 단위 재현 확인).

## 생성 순서

```bash
S=wcojb4TFT35; V=outputs/hm3d_v3; T=outputs/teacher_v3
python scripts/build_mesh_bbox_annotations.py --annotations outputs/hm3d/$S.annotations.json \
  --output $V/$S.annotations.json --audit $V/$S.mesh_bbox.audit.json
python -m athome.data.hm3d.converter --input $V/$S.annotations.json --output $V/$S.room_object_map.json
python -m athome.data.hm3d.coordinates --input $V/$S.room_object_map.json --output $V/$S.room_object_map.z_up.json
python scripts/apply_scene_scope.py --input $V/$S.room_object_map.z_up.json \
  --config configs/data/scene_scope.json --output $V/$S.room_object_map.z_up.scoped.json
python -m athome.scene_graph.geometric_relations --input $V/$S.room_object_map.z_up.scoped.json \
  --output $V/$S.geometric_relations.json --distance 0.5 --height 0.1 --overlap 0.5
python -m athome.scene_graph.semantic_inputs --scene $V/$S.room_object_map.z_up.scoped.json \
  --geometry $V/$S.geometric_relations.json --output $V/$S.semantic_inputs.json
python scripts/make_semantic_batch.py --input $V/$S.semantic_inputs.json --scene-id $S --output $T/$S.semantic.batch.jsonl
python scripts/validate_semantic_batch.py --input $V/$S.semantic_inputs.json --batch $T/$S.semantic.batch.jsonl
python scripts/compact_semantic_batch.py --input $T/$S.semantic.batch.jsonl \
  --output $T/$S.semantic.vote.batch.jsonl --protocol v2
python scripts/submit_batch.py --input $T/$S.semantic.vote.batch.jsonl   # 외부 전송·비용
python scripts/collect_semantic_batch.py --state $T/$S.semantic.vote.batch.state.json \
  --labels-output $T/$S.semantic_labels.vote.review.json
```

이후 그래프·catalog·masking 단계는 `docs/mesh_bbox_ko.md`의 명령과 같다(경로만 v3, 라벨 파일은 `semantic_labels.vote.review.json`, masked batch 기준은 `semantic.vote.batch.jsonl`).

층·A/B 영역·경로·Symbolic 단계는 layout(`src/athome/data/hm3d/layout.py`)으로 버전을 고른다. 환경 변수가 없으면 기존 `outputs/hm3d` 경로이며, 기존 결과를 그대로 재현한다(masked graph·navigation 검증 통과).

```bash
export ATHOME_HM3D_LAYOUT=outputs/hm3d_v3/layout.json
python scripts/prepare_floor_environments.py --graph $V/$S.workspace_graph.json --building-id $S --output $V/$S/floor_environments.json
python scripts/inspect_component_rooms.py
python scripts/prepare_component_membership.py
python scripts/carry_over_membership_decisions.py \
  --decisions outputs/hm3d/$S/component_membership.decisions.review.json \
  --old-membership outputs/hm3d/$S/component_membership.review.json
python scripts/build_component_graphs_review.py
python scripts/build_component_grids.py
python scripts/generate_workspace_goal_candidates.py
python scripts/check_workspace_paths.py
python scripts/check_masked_workspace_paths.py
python scripts/build_component_masked_graphs.py
python scripts/check_component_navigation.py
python scripts/run_symbolic_review.py --max-steps 30 --include-absent --output $V/$S/symbolic_<이름>.review.json
```

복도 객체 소속 결정(25개)은 수동 검토 결과다. 새 버전의 미배정 객체 집합이 같고 객체별 근거(bbox, 바닥 겹침)의 변화가 허용치(0.05) 이하일 때만 승계한다. v3에서는 최대 변화가 `floor_317`의 B 겹침 +0.029 m²(B 판단 방향과 일치)였다.

## 결과 (현재)

- 범위 밖 객체 164개 제외(책 94, container 11, box 8, bag 8 등), 유지 803개. 감싼 가구는 shelf 118, bookshelf 34, cabinet 7, bathroom cabinet 5.
- 이 장면의 book 목표는 범위 안 인스턴스가 1개(`book_429`, Room `_4`)만 남는다.
- 라벨(프로토콜 v2): Workspace 58개. 득표가 경계인 Source는 `piano_424` 5/9, `shelf_1036` 5/9, `tray_553` 6/9(Source 정책으로 제외), `fireplace_350` 6/9, `sink_547` 7/9.
- A/B 영역 삼각형·Room 후보·격자 배열은 기존과 동일하다. A: Room 4, Workspace 5, Object 108. B: Room 5, Workspace 11, Object 150(기존 250, 선반 안 책 등 제외).
- 경로: Workspace 16개 중 경로 14, 후보 없음 2(`shelf_1046`, `bedside cabinet_205`).
- Symbolic(카메라 높이 0.88 m 임시): 목표가 있는 13개 중 11개 발견. B의 Search Location이 약 130개에서 약 62개로 줄어 B/cup·B/towel이 29번째 방문에서 욕실 `_13`에 도달해 발견된다. 실패는 B/bag(`_15`의 가방 2개, 30회 안에 `_15` 미방문)과 GT가 없는 조합(A book·cup·mug, B book).
- `$T/$S.semantic.compact.batch.*`와 `$T/label_stability/`는 프롬프트 v2 단일 응답·10회 샘플링 측정 기록이다. 그래프 입력은 `semantic.vote.batch.*`(프로토콜 v2)이다.

## 수동 단계 자동화 (2026-09-26, `outputs/hm3d_v3_auto/`)

장면마다 사람이 하던 두 단계를 규칙으로 바꿨다. `outputs/hm3d_v3_auto/layout.json`은 v3의 그래프·catalog·masked graph를 그대로 쓰고 영역 단계만 자동 결과로 만든다.

1. 층·계단 분리(`src/athome/data/hm3d/navmesh_levels.py`, `scripts/segment_navmesh_levels.py`): 기울기 10° 이하 삼각형이 이어진 1 m² 이상 면 = 층 플랫폼. 플랫폼 높이 ± NavMesh 단차(`agent_max_climb` 0.2 m)를 벗어난 삼각형에서 시작해 이웃한 기울어진 삼각형까지 계단으로 확장(계단 첫 칸도 계단). 나머지 대역 안 삼각형 중 플랫폼과 이어진 것이 이동 영역. 수동 선택 대비: A는 계단 위 평평한 바닥 조각 2개(0.15 m²) 추가, B는 계단 첫 칸(기울어진 2개, 수동은 B 쪽만 유지) 제외. 수동 코드의 삼각형 82 하드코딩은 규칙이 제외해 불필요.
2. 방 바닥 후보(`scripts/inspect_component_rooms.py`): 영역 NavMesh가 로봇 바닥 면적(Tidybot++ 0.50×0.54 m = 0.27 m²) 이상 덮는 바닥만 후보. 계단참 `floor_318`처럼 0.01 m² 스친 경우는 `grazed_floors_below_robot_footprint`로만 기록. 기존 보고서에는 해당 사례가 없어 legacy가 바이트 단위로 재현된다.
3. 복도 객체 A/B 배정(`scripts/decide_component_membership.py`): 한 영역 바닥하고만 겹치고, 물체 아랫면이 그 바닥보다 0.1 m(단차의 절반) 넘게 낮지 않으면 배정, 아니면 보류. 수동 결정 25개와 전부 일치(자료가 허용하는 기준 범위 0.03~0.2 m).

검증: 자동 결과로 그래프·격자·Goal 후보·경로·masked graph·navigation·Symbolic까지 전 단계 통과. Symbolic 16개 에피소드가 수동 기반 v3와 전부 같다. 통행 가능 면적 A 15.84→15.99 m², B 14.75→14.36 m².

## 여러 장면·층·섬 (2026-09-26)

- `segment_navmesh_levels.py`: `--island`를 생략하면 NavMesh의 모든 섬을 분리해 `islands` 목록(schema 0.2)으로 저장한다. `--island`를 주면 기존 단일 섬 형식이다.
- `inspect_component_rooms.py`: 각 영역을 넓이 가중 NavMesh 높이에 가장 가까운 층(`floor_environments.json`의 바닥 중심 높이 범위, 층 간격의 절반 이내)에 배정하고, 그 층의 바닥만 Room 후보로 본다. 제외 규칙:
  - Room 후보가 없는 영역(가구 윗면 등)은 제외한다.
  - Room마다 바닥을 가장 많이 덮는 영역이 그 Room의 주 영역이다. 어느 Room의 주 영역도 아닌 영역은 다른 영역 Room 안의 고립된 틈으로 보고 제외한다. Habitat rearrangement의 최대 섬 관례를 Room 단위로 적용한 것이다. 이렇게 해도 계단으로 나뉜 복도 `_9`처럼 한 Room이 여러 영역에 속할 수 있다.
  - 제외된 영역은 `discarded_components`에 사유와 함께 기록한다.
  - 영역 이름은 넓이 순으로 A, B, …, Z, AA, … 이다.
- 이후 스크립트는 A/B 대신 보고서의 영역 목록(`Layout.component_names()`)을 돈다. 격자는 영역별 섬에서 만든다.
- 영역 masked graph 검사는 masking audit의 제거 객체(예: `basket with books`)도 가려진 것으로 본다(`Layout.masking_audit`).
- 단일 섬 입력(legacy·v3·v3_auto)은 방 후보 보고서·격자·masked graph·navigation이 바이트 단위로 같다.

`outputs/hm3d_v3_allfloors/`는 wcojb4TFT35 전 층 결과다(v3 그래프·라벨 재사용, LLM 추가 호출 없음).

| 층 | 영역 | 섬 | Room |
|---|---|---|---|
| floor_1 | A | 2 | `_2`–`_6` |
| floor_0 | B | 3 | `_0`, `_1` |
| floor_2 | C | 0 | `_9`–`_12` |
| floor_2 | D | 0 | `_9`, `_13`–`_16` |
| floor_1 | E | 1 | `_7` |
| floor_1 | F | 1 | `_8` |

- 제외된 틈 영역: 섬 5, 섬 6, 섬 1 조각.
- C/D의 Symbolic 에피소드 16개는 v3_auto의 A/B와 같다. 그래프 해시만 다르다.
- Symbolic 48개 중 18개 발견. 오프라인 Teacher(mincost)는 에피소드 66개(floor_2만일 때 39개)를 만든다.

### 장면 구동부 `scripts/run_scene_pipeline.py`

```bash
python scripts/run_scene_pipeline.py --scene-dir data/scene_datasets/hm3d/minival/00800-TEEsavR23oF --version v4 [--dry-run]
```

- 출력이 없는 단계만 실행하고, 이미 있는 출력은 덮어쓰지 않는다.
- LLM batch 두 번(unmasked, masked)은 관문이다. 제출 명령을 출력하고 멈춘다. 제출 후 다시 실행하면 수집하고 이어서 진행한다.
- 영역 단계용 layout은 `outputs/hm3d_<version>/<scene>.layout.json`에 저장된다.
- 로컬 주석 장면 00800·00803·00808은 unmasked vote batch 생성까지 끝났다(Room 13/10/19개, 제출 전).

### 여러 층에 걸친 Room과 공유 바닥 (2026-09-26)

- **층 먼저 분리(Hydra·HOV-SG의 floor → room → object 계층, `src/athome/data/hm3d/floors.py`):**
  - 바닥 객체가 여러 층 높이에 있는 Room(계단실, 복층)은 층별로 나눈다.
  - 각 객체는 자기 아랫면 이하에서 가장 높은 바닥의 층에 속한다(허용 0.1 m).
  - 나뉜 Room은 해당하는 모든 층의 `room_ids`에 들어가고, 층별 객체 목록은 `partial_room_object_ids`에 기록한다.
  - 영역 후보, 주 영역(Room·층 단위), 소속 후보, Symbolic·Teacher의 층 내 목표 인스턴스가 모두 이 층 정보를 쓴다.
  - catalog 높이 범위도 이런 Room에서는 객체가 있는 층의 바닥을 기준으로 한다.
- **공유 바닥:**
  - 한 Room의 바닥 객체 하나를 여러 영역이 같이 딛고 있으면 바닥 겹침만으로는 소속을 가를 수 없다.
  - 이 경우에만 가장 가까운 이동 가능 면(NavMesh XY 거리, 유일할 때)으로 정한다. Habitat ObjectNav가 가장 가까운 이동 가능 지점의 섬에서 객체에 도달한다고 보는 관례를 따른 것이다.
  - wcojb4TFT35에는 이 경우가 없어 수동 결정 25개 일치가 유지된다.
  - 영역 그래프에 해당 Room의 바닥 객체가 없으면(공유 바닥이 보류된 경우) `floor_z_m`을 그 영역 Room 후보의 바닥 높이로 정한다.
- **Workspace:**
  - Source와 그 위 객체를 한 단위로 본다. 소속 결정 후 모두 한 영역에 있으면 그 영역에 넣고, 아니면 Workspace 전체를 보류한다.
  - Goal 후보와 masked 경로 검사는 최종 영역 그래프의 Workspace를 쓴다.
- **재현:** 기존 결과(legacy·v3·v3_auto·allfloors)는 그대로 재현된다. 예외는 masked 경로 요약의 `unresolved_object_count`다. 결정 전 25개가 아니라 결정 후 보류 14개를 센다.

**v4 진행 상황**

| 장면 | 상태 |
|---|---|
| 00800 (TEEsavR23oF) | unmasked 라벨 수집 완료, masked batch(14개 요청) 제출 대기 |
| 00808 (y9hTuugGdiq) | unmasked 라벨 수집 완료, masked batch(19개 요청) 제출 대기 |
| 00803 (k1cupFYWXJ6) | 전 단계 완료 |

00803 세부:
- 전시 건물이다(벽, 그림, 전시창, 안내판). 목표 범주 인스턴스가 0개라서 masked 재라벨이 필요 없었고, Symbolic 112개 조합이 모두 GT 없음이다.
- 13개 층에서 영역 14개가 나왔다. 복층 Room `_8`은 두 영역(B, C)으로 나뉘었다.

### 목표 설정 v0.2와 자동 제출 (2026-09-26)

- **목표 설정 v0.2** (`configs/data/pilot_targets.v0.2.json`, v4 기본값):
  - 목표 범주 8개는 그대로다.
  - 목표의 휴대형 하위 종류를 별칭으로 추가했다: `table/desk/floor/bedside lamp` → lamp, `handbag` → bag, `coffee mug` → mug, `storage box` → box.
  - 고정 조명(천장·벽 조명, 샹들리에, light fixture)과 머리명사가 목표가 아닌 복합어(`electric box`, `pen cup`, `towel bar`)는 제외한다.
  - 설정 파일은 layout의 `targets` 필드로 고른다. 기본값은 v0.1이라 v3 계열 결과가 그대로 재현된다.
  - v0.1로 만든 v4 catalog·masked 산출물은 `outputs/hm3d_v4/superseded/targets_v0.1/`에 있다.
- **v4 결과(목표 v0.2):** 세 장면 모두 전 단계를 마쳤다. masked batch 사용량은 입력 약 9.1만, 출력 약 2만 토큰이다.

  | 장면 | 영역 | Symbolic(GT 있는 조합 기준) |
  |---|---|---|
  | 00800 | 4개 | 14개 중 11개 발견 |
  | 00808 | 5개 | 12개 중 10개 발견. 가장 큰 영역 A의 lamp·pillow·towel 3개 실패 |
  | 00803 | 14개 | GT 있는 조합 없음. 새로 목표가 된 `table lamp` 2개가 NavMesh가 없는 Room `_2`에 있어 범위 밖 |
- **자동 제출:** `run_scene_pipeline.py --submit`은 LLM batch를 직접 제출하고 완료될 때까지 기다린 뒤 이어서 진행한다. 사용자가 데이터셋 파이프라인 batch 제출을 상시 승인했다(2026-09-26).
