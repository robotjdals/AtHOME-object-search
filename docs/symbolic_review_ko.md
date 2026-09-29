# A/B Symbolic 탐색 검증

현재 단계는 단일 장면의 Masking 그래프·격자 지도를 공통 탐색 코드에 연결하는 개발 검증이다. 학습 데이터 생성이나 로봇 가시성 검증은 아직 완료되지 않았다.

## 구성

- `src/athome/search/session.py`: 공통 후보 선택과 상태 갱신. 완료된 Location만 Visited에 넣는다. 주변 Standalone을 자동 방문 처리하지 않고, 이동 실패 Location을 영구 제외하지 않는다.
- `src/athome/execution/command.py`: 실제 실행기의 제한된 재시도마저 실패하면 명령을 PAUSED로 전환한다. 재개 전 원인을 확인한다.
- `src/athome/scene_graph/query.py`: Room 바닥 객체의 중앙 높이 중앙값을 기준으로 높이 필터링한다. `room.floor_z_m`으로 명시할 수 있다. 일반 사용 시 근거가 없으면 높이 필터를 생략하고, 이 실행기는 Room별 근거가 없으면 중단한다.
- `src/athome/symbolic/environment.py`: 공통 Dijkstra 경로로 이동하고, Goal 도착 후 실행기와 같은 heading 순서로 주입된 observer를 호출해 관측을 만든다.
- `src/athome/symbolic/habitat_observer.py`: Habitat-Sim semantic·depth 카메라. 인스턴스별 가시 픽셀 수를 센다.
- `src/athome/symbolic/surface.py`: 영역 NavMesh 삼각형에서 Goal XY의 바닥 Z를 보간한다(카메라 높이 기준).
- `src/athome/data/hm3d/semantic_mesh.py`: semantic GLB에서 인스턴스별 삼각형·AABB를 읽는다. `scripts/review_corridor_surfaces.py`와 같은 파싱 규칙이다.
- `scripts/run_symbolic_review.py`: A/B × 목표별 독립 에피소드. MinCost 정책으로 동작하며 GT가 없는 세 조합은 기본적으로 건너뛴다. `--include-absent`로 음성 에피소드도 실행한다.
- `configs/data/symbolic_observation.json`: 카메라·가시성 기준(잠정값).

## 실행

저장소: `/home/min/projects/AtHOME`
환경: `athome-data` (Habitat-Sim 0.3.3, GPU EGL 렌더링)

```bash
conda activate athome-data
cd /home/min/projects/AtHOME
python -m pip install -e .
python -m pytest -q
python scripts/run_symbolic_review.py \
  --max-steps 30 --include-absent \
  --output outputs/hm3d/wcojb4TFT35/symbolic_habitat_render.include_absent.review.json
```

기존 결과 파일을 덮어쓰지 않는다. 재실행하려면 다른 결과 이름을 사용한다. pytest가 없으면 `python -m pip install pytest`로 설치 후 검사한다.

입력은 기존 `component_masked_graphs.review`, `component_graphs.review`, `component_grids/f440d0f045c4_5cm`(grid·mesh·metadata), GT catalog, 원본 semantic GLB/TXT다. 기존 생성/검증 스크립트를 재사용해 원본 해시, 마스킹, 참조, 격자, Goal 후보와 시작 셀을 검증한다. mesh.npz는 component 보고서의 삼각형 ID와 metadata Z 범위로 검증한다.

## 관측 모델

기본은 `wall_los_2d`다(2026-09-26 결정). 제안서 6.2절은 Symbolic 환경을 "실제 로봇과 센서를 물리적으로 시뮬레이션하지 않는" 환경으로, 발견을 "GT 위치와 관측 거리 조건"으로 정의한다. 거리 조건만 쓰면 벽 너머 물체가 발견되므로(A `box_1033`, `towel_253`), 다중 물체 탐색 연구의 격자 센서 모델(Wandzel et al., ICRA 2019, pomdp_py: 벽 등 정적 구조물 격자 + 제한 거리 + 가림)을 따른다.

- 관측: Goal 도착 후 한 번. 로봇이 60° 간격 6방향을 회전 관측하므로(간격 < D435i RGB 수평 시야 69.4°, 2026-09-28 4방향에서 변경) 360° 원형 센서로 본다. 이동 중에는 관측하지 않는다. 아래 검증의 "4방향 렌더"는 변경 전 실행기 기준이다.
- 거리: 목표 footprint 내부 3×3 표본점(1/4, 1/2, 3/4 지점)까지 3.0 m 이내(D435i 권장 범위).
- 가림: 스캔 당시 상태의 불투명 구조물(벽, 계단 벽, 문짝) semantic mesh를 층 높이 대역(바닥 +0.3~+1.8 m)으로 잘라 평면 기하로 쓴다. 문도 포함한다. NavMesh도 스캔 당시 문 상태로 만들어졌고, A/B 욕실 사이 닫힌 문(`door_261`)이 두 영역을 나눈다. 유리(창, 샤워 도어, 미닫이문, glass)는 막지 않는다.
- 광선 판정: 표본점까지의 선분에서 첫 가림 지점이 표본점보다 2 cm 넘게 앞이면 가려진 것이다(표준 ray casting 허용 오차). 벽에 붙은 물체는 방 쪽에서 보이고 벽 반대편 물체는 가려진다.
- 격자로 벽을 칠하는 방식은 버렸다. 벽 반대편에 붙은 물체의 bbox가 벽 칸과 겹쳐 보이는 오탐이 있었다(격자 이산화 오차, Wandzel 문서에도 명시된 한계).
- 관측 대상: 영역과 같은 층(floor_environments) 방의 목표 범주 인스턴스. 결과에 `found_object_in_component`로 영역 밖 여부를 기록한다.
- 검출기 오류, 높이·시야각은 모델링하지 않는다.
- 알려진 한계: 벽을 한 평면으로 합치므로 일부 높이에만 뚫린 곳(벽 안쪽 수납 공간, 부분 창)은 막힌 것으로 본다(`box_279`).

### 검증 결과 (2026-09-26, `observation_model_comparison.height_scope.review.json`)

모든 Goal 후보 자세(A 1,884 + B 2,365) × 같은 층·3 m 이내 목표 인스턴스, 높이 범위(1.5 m 이하) 안의 24,440쌍.

| 비교 | 일치율 | Cohen κ |
|---|---:|---:|
| `wall_los_2d` vs 12방향 렌더 | 92.6% | 0.848 |
| 4방향 렌더(실행기) vs 12방향 렌더 | 93.9% | 0.875 |
| `wall_los_2d` vs 4방향 렌더 | 88.2% | 0.756 |

κ 0.81 이상은 "거의 완전한 일치"(Landis & Koch 1977)다. 남은 차이: 2D만 보이는 쌍의 64%가 1 m 이내(고정 각도 카메라 화면 밖으로 벗어나는 가까운 높은/낮은 물체), 렌더만 보이는 쌍의 46%가 2.5 m 초과(Habitat depth는 광축 거리라 3 m 경계에서 더 멀리 셈). 높이 범위 적용 전에는 천장·벽 조명 때문에 81.5%(κ 0.60)였다.

`habitat_render`(Habitat semantic 렌더링, 실행기 4방향)는 검증용 기준이다. 카메라 높이는 실측 전 0.88 m 임시값이다. `scripts/compare_observation_models.py`가 모든 Goal 후보 자세 × 같은 층·3 m 이내 목표 인스턴스 쌍에서 두 모델의 일치율을 잰다. 2D 모델과 비교하는 대상은 12방향 회전 렌더링(ObjectNav의 oracle-visibility, Batra et al. 2020)이다. 4방향 렌더링과의 차이는 시야각 틈의 효과로 따로 보고한다.

- GT는 평가 환경에만 전달한다. 정책은 Masking된 그래프와 이동 비용만 사용한다.
- 관측 stamp는 시뮬레이션 tick이다. Goal 후보가 없는 Location은 따로 기록하며, 후보 없음이 물리적 접근 불가능의 증명은 아니다.
- 기준 바닥 높이는 Masking 전 영역 그래프에서 한 번 구해 모든 목표에 고정한다.
- `evaluator_*`, GT 개수, 정답 ID가 포함된 결과 JSON은 평가용이다. 정책 프롬프트나 SFT 입력으로 그대로 사용하지 않는다.

## 다음 단계

- 그래프 bbox 품질: 그래프 객체 967개 중 104개(10.8%)는 semantic mesh가 bbox 밖으로 5 cm 넘게 나간다. 관계 계산·Workspace·높이 필터에 영향을 주므로 bbox 출처를 semantic mesh로 바꿀지 검토한다. mesh에는 떨어진 조각이 섞인 인스턴스도 있다(예: `book_60`).
- Standalone Location 규칙, Goal 후보 없는 Location, max_steps 기준을 정한다.
- 시작점 다양화·건물 단위 Split·Search State 생성·Teacher 라벨링으로 진행한다.
