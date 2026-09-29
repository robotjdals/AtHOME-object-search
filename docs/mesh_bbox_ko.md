# Semantic mesh bbox 재구축 (v2)

## 이유

기존 그래프 bbox는 Habitat OBB에서 만든 AABB다. semantic mesh와 비교하면 그래프 객체 967개 중 87개가 mesh를 5 cm 넘게 놓친다(예: `chair_199` 등받이 46 cm 누락, `lamp_201` 상단 누락). bbox는 Child 관계·Workspace 위치·높이 필터·Goal 후보에 쓰이므로 mesh 기준으로 다시 만든다. 기존 `outputs/hm3d/`, `outputs/teacher/`는 변경하지 않고 새 버전 폴더에 생성한다.

## 규칙 (`src/athome/data/hm3d/mesh_bbox.py`, `semantic_mesh_aabb_v2`)

1. 인스턴스 삼각형을 꼭짓점 간격 0.10 m 이내 연결로 조각 분리.
2. 표면적 5% 미만 조각(annotation 잡티) 제거.
3. 벽·바닥·천장은 남은 조각을 모두 유지한다. 가구에 가려 한 면이 크게 끊기는 것이 정상이다.
4. 그 밖의 객체는 강체로 본다. 0.30 m 이내 조각끼리 한 묶음(재구성 구멍으로 끊긴 좌석·등받이 등)으로 보고 면적이 가장 큰 묶음만 남긴다. 더 떨어진 조각은 칠 번짐이나 이웃 인스턴스다(예: `window_189`의 1.25 m 떨어진 조각은 `window_190` 위치).
5. 남은 삼각형의 AABB.

사람 판단 없이 결정된다. 다부품 유지 29개·분리 조각 제거 10개는 `review_mesh_bboxes.py`의 3D 화면으로 확인할 수 있다. v1(단일 규칙 + 수동 검토 39개)은 제출 전에 폐기했다.

## 생성 순서

```bash
S=wcojb4TFT35; V=outputs/hm3d_meshbbox_v2; T=outputs/teacher_meshbbox_v2
python scripts/build_mesh_bbox_annotations.py --annotations outputs/hm3d/$S.annotations.json \
  --output $V/$S.annotations.json --audit $V/$S.mesh_bbox.audit.json
python scripts/review_mesh_bboxes.py --annotations outputs/hm3d/$S.annotations.json \
  --audit $V/$S.mesh_bbox.audit.json --output $V/$S.mesh_bbox.review.html
python -m athome.data.hm3d.converter --input $V/$S.annotations.json --output $V/$S.room_object_map.json
python -m athome.data.hm3d.coordinates --input $V/$S.room_object_map.json --output $V/$S.room_object_map.z_up.json
python -m athome.scene_graph.geometric_relations --input $V/$S.room_object_map.z_up.json \
  --output $V/$S.geometric_relations.json --distance 0.5 --height 0.1 --overlap 0.5
python -m athome.scene_graph.semantic_inputs --scene $V/$S.room_object_map.z_up.json \
  --geometry $V/$S.geometric_relations.json --output $V/$S.semantic_inputs.json
python scripts/make_semantic_batch.py --input $V/$S.semantic_inputs.json --scene-id $S --output $T/$S.semantic.batch.jsonl
python scripts/validate_semantic_batch.py --input $V/$S.semantic_inputs.json --batch $T/$S.semantic.batch.jsonl
python scripts/compact_semantic_batch.py --input $T/$S.semantic.batch.jsonl --output $T/$S.semantic.compact.batch.jsonl
# 제출(외부 전송·비용): python scripts/submit_batch.py --input $T/$S.semantic.compact.batch.jsonl
# 수집: python scripts/collect_semantic_batch.py --state $T/$S.semantic.compact.batch.state.json \
#         --labels-output $T/$S.semantic_labels.review.json
```

재현성: 같은 명령으로 기존 입력에서 만든 compact batch는 이미 제출된 `outputs/teacher/wcojb4TFT35.semantic.compact.batch.jsonl`과 바이트 단위로 같다.

이후 단계(모두 스크립트화, 기존 산출물로 재현 검증 완료):

```bash
python scripts/build_hm3d_workspace_graph.py --semantic-inputs $V/$S.semantic_inputs.json \
  --labels $T/$S.semantic_labels.review.json --reviewed-output $T/$S.semantic_labels.reviewed.json \
  --graph-output $V/$S.workspace_graph.json
python scripts/build_target_catalog.py --graph $V/$S.workspace_graph.json \
  --config configs/data/pilot_targets.json --scene-id $S --output $V/$S.target_catalog.json
python scripts/build_masked_inputs.py --semantic-inputs $V/$S.semantic_inputs.json \
  --config configs/data/pilot_targets.json --input-dir $V/$S/masked_inputs --audit-dir $V/$S/masking_audit
python scripts/make_masked_semantic_batch.py --masked-dir $V/$S/masked_inputs \
  --labels $T/$S.semantic_labels.reviewed.json --baseline-batch $T/$S.semantic.compact.batch.jsonl \
  --batch-output $T/$S.masked.semantic.batch.jsonl --manifest-output $T/$S.masked.semantic.manifest.json
# 제출: python scripts/submit_batch.py --input $T/$S.masked.semantic.batch.jsonl
python scripts/collect_masked_semantic_batch.py --state $T/$S.masked.semantic.batch.state.json \
  --masked-dir $V/$S/masked_inputs --labels-dir $T/$S.masked_labels
python scripts/build_masked_graphs.py --input-dir $V/$S/masked_inputs --label-dir $T/$S.masked_labels \
  --output-dir $V/$S/masked_graphs --catalog $V/$S.target_catalog.json
```

재현 검증: 기존 입력으로 실행하면 `workspace_graph.v2.json`(rooms/workspaces/objects), `target_catalog.json`, `masked_inputs_v2`·`masking_audit_v2`, masked batch·manifest, `masked_graphs`·`*.reviewed.json`이 기존 파일과 같다. 수동 검토였던 `tray_553` 제외는 공통 Source 정책(`src/athome/scene_graph/source_review.py`)으로 같은 결과가 나온다.

## 현재 결과

- mesh가 있는 인스턴스 1020개. 그래프 객체 집합은 967개로 동일하다. 기존에 bbox가 없어 제외된 38개(책 28개 등)는 mesh에도 삼각형이 없다.
- OBB bbox가 mesh를 5 cm 넘게 놓친 객체 79개. 잡티 제거 60개, 다부품 유지 29개, 분리 조각 제거 10개.
- unmasked LLM batch `batch_6ab75bc74bc481908271778ec72f4458` 완료(17/17). Workspace 58 → 52.
- LLM 라벨 불안정: `_0`의 선반 5개, `_15`의 `bookshelf_157`이 Source에서 빠졌다. 해당 객체 입력은 거의 또는 완전히 같고 같은 Room의 다른 객체만 바뀌었다. temperature 0에서도 문맥 변화에 선택이 뒤집힌다. 단일 응답을 그대로 교사 라벨로 쓰기 어렵다.

## 남은 단계

1. LLM 라벨 안정화 방식 결정(아래 보고 참조) 후 unmasked 라벨 확정.
2. Target catalog·Masking 입력 → masked batch 제출·수집 → Masking 그래프.
3. 층 분리·A/B Room 후보·객체 소속을 새 그래프로 다시 만든다. 계단 삼각형 선택은 NavMesh 기준이라 그대로 쓴다. 복도 객체 소속 결정은 다시 검토한다.
4. Goal 후보·경로·Symbolic 검증 재실행.
