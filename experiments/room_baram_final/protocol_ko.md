# 최종 실험 절차: 학습된 플래너 vs 학습 안 된 플래너 (room_baram)

## 목적
위치를 모르는 물체(unknown)를 찾을 때, 증류 학습한 Qwen3-4B 플래너(SFT 어댑터)가 학습하지 않은 같은 모델(기본 Qwen3-4B)보다 효율적으로 찾는지 같은 조건에서 비교한다.
두 플래너의 차이는 다음 장소를 고르는 모델 하나뿐이다(같은 프롬프트, 씬그래프, 목표 지점, 주행, 인식, 발견 판정).

## 실험 전에 고정할 것
- 코드: 실험 직전 커밋하고 커밋 번호를 아래에 적는다. 실험 중에는 코드와 설정을 바꾸지 않는다.
- 설정: `configs/robot/room_baram.yaml` (명령당 시간 예산 300초, 인식 신뢰도 0.35, CLIP 5위, 목표 거리 가구 1.0 m / standalone 2.0 m).
- 그래프와 지도: `outputs/room_baram/graph.json`, `room_baram_demo.yaml`.
- 플래너 서버: 랩 서버 vLLM. 어댑터 v1(room, workspace, search_location)과 기본 모델 qwen3-4b가 같은 서버에 있다.
- 검출기: 실험 내내 한 곳만 쓴다(랩 GPU 또는 `detector_host:=local`). 시간 지표가 검출 속도에 따라 달라진다.
- 환경: 같은 조명. 실험 중 사람이 공간과 카메라 시야에 들어가지 않는다.
- unknown 물체(노트북, 책, 식물)는 집 안에 하나씩만 둔다. 쓰지 않는 물체는 공간 밖으로 치운다.
- known 물체(그래프에 있는 물체)는 지도 만들 때 자리 그대로 둔다.

기록: 커밋 ______ / 검출기 ______ / 날짜·시간 ______

## 실행 순서 (ABBA, `run_order.csv`; 이 폴더의 파일은 실험 전 계획, 기록은 outputs/final_experiment/ 사본에)
| 블록 | 플래너 | 시나리오 |
|---|---|---|
| 1 | trained (학습된 어댑터) | 1-10 |
| 2 | base (학습 안 된 모델) | 1-10 |
| 3 | base | 11-20 |
| 4 | trained | 11-20 |

블록이 바뀔 때만 탐색을 다시 켠다:

    ros2 launch athome_bringup realtime_search.launch.py robot_config:=$PWD/configs/robot/room_baram.yaml rviz:=true console_host:=0.0.0.0 planner_variant:=trained
    ros2 launch athome_bringup realtime_search.launch.py robot_config:=$PWD/configs/robot/room_baram.yaml rviz:=true console_host:=0.0.0.0 planner_variant:=base

시작 로그에 `planner llm (trained)` 또는 `planner llm (base)`가 나오는지 확인한다.

## 한 회 진행
1. `run_order.csv`의 물체를 정해진 책상 위, 카메라에서 보이는 쪽에 둔다(매번 비슷한 자리).
2. 로봇을 시작 위치(start_x, start_y)로 옮기고 위치 추정이 맞는지 RViz에서 확인한다.
3. 콘솔에 명령을 입력하고(예: "노트북 찾아줘") 확인 창의 이름을 보고 시작한다.
4. 끝날 때까지 손대지 않는다. 취소는 안전 문제일 때만 한다.
5. 결과(발견, 못 찾음, 시간 초과)는 로그에 남는다. `run_order.csv`의 valid 칸에 O, 무효면 X와 이유를 적는다.

## 판정 기준 (실험 전에 확정, 결과를 본 뒤 바꾸지 않음)
- 성공: 로봇이 발견을 선언했고 발견 사진의 박스 안에 목표가 있다.
- 실패:
  - 오검출: 발견을 선언했는데 사진의 물체가 목표가 아니다.
  - 탐색 소진: 모든 탐색 위치를 돌고 못 찾았다.
  - 시간 초과: 300초를 넘겼다.
  - 시스템 중단: 주행 실패, 프로그램 오류로 더 진행할 수 없다.
  - 운영자 취소: 이유와 관계없이 실패.
- 무효(바로 다시 시행): 시스템과 무관한 원인만. 사람이 시야에 들어옴, 물체를 잘못된 책상에 둠, 배터리 문제. 결과를 보기 전에 판단한다.
- 사람 개입(일시정지 후 재개 등)은 성공이어도 기록하고, 개입 없는 성공률을 따로 보고한다.

## 지표
- 성공률(95% Wilson 신뢰구간), SPL(최단 경로 1-30 m, 모든 시나리오가 해당), Search Steps(이동과 관측을 마친 탐색 위치 수), 이동 거리, 시간.
- 같은 시나리오끼리 짝지어 비교한다(시나리오별 방문 수와 이동 거리의 차이, 부호 검정 또는 Wilcoxon).
- 시나리오의 typical / atypical / neutral 구분은 실험 전에 정한 것이다(노트북·책: 사무실·침실·거실 책상 typical, 주방 책상 atypical; 식물: neutral).

## 기록 위치
- 명령·방문·결과: `outputs/room_baram/visits/<시작 시각>/events.jsonl`
- 플래너 질의: 같은 폴더 `decisions.jsonl`
- 방향별 사진: 같은 폴더 `cmd*_step*_h*.jpg`, 발견 사진: `outputs/room_baram/found/`
- MPPI 주행 기록: 문상훈 님 컨테이너 `results/real_*.csv` (이동 거리 계산)
