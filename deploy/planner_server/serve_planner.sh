#!/bin/bash
# AtHOME LLM Planner 서버 (vLLM)
#   내부 포트 6007 (외부 주소·포트는 서버 환경에 따라 다름, 저장소에 적지 않음)
#   사용법: /data/llm/serve_planner.sh          (백그라운드 실행)
#           /data/llm/serve_planner.sh stop     (중지)
#   어댑터: /data/adapters/<room|search_location|workspace>/current 가 있으면 자동 등록
#   옵션 환경변수: ENFORCE_EAGER=1 (CUDA graph 끄기), GPU_UTIL (기본 0.95),
#                 MODEL_DIR (기본 /data/models/Qwen3-4B, 비교 테스트용. 서빙 이름은 qwen3-4b 유지)
source /data/llm/env.sh
LOG=/data/llm/logs/vllm.log
PIDF=/data/llm/vllm.pid

if [ "$1" = "stop" ]; then
  [ -f $PIDF ] && kill $(cat $PIDF) 2>/dev/null && rm -f $PIDF && echo "stopped" || echo "not running"
  exit 0
fi
if [ -f $PIDF ] && kill -0 $(cat $PIDF) 2>/dev/null; then echo "already running (pid $(cat $PIDF))"; exit 1; fi

ARGS=("${MODEL_DIR:-/data/models/Qwen3-4B}"
  --served-model-name qwen3-4b
  --host 0.0.0.0 --port 6007
  --api-key "$(cat /data/llm/api_key)"
  --max-model-len 4096
  --max-num-seqs 4
  --gpu-memory-utilization "${GPU_UTIL:-0.95}"
  --enable-prefix-caching)
[ "${ENFORCE_EAGER:-0}" = "1" ] && ARGS+=(--enforce-eager)
# Instruct-2507처럼 max_position_embeddings가 큰 모델은 그대로 띄우면 KV 캐시가 모자라 기동 실패 → 4만으로 제한
MPE=$(python3 -c "import json;print(json.load(open('${MODEL_DIR:-/data/models/Qwen3-4B}/config.json'))['max_position_embeddings'])")
[ "$MPE" -gt 40960 ] && ARGS+=(--hf-overrides '{"max_position_embeddings": 40960}')
echo "model: ${MODEL_DIR:-/data/models/Qwen3-4B}"

MODS=(); MAXR=0
for n in room search_location workspace; do
  d=/data/adapters/$n/current
  if [ -f $d/adapter_config.json ]; then
    MODS+=("$n=$(readlink -f $d)")
    r=$(python3 -c "import json;print(json.load(open('$d/adapter_config.json'))['r'])")
    [ "$r" -gt "$MAXR" ] && MAXR=$r
  fi
done
# 평가용 추가 어댑터: /data/adapters/eval/<서빙 이름>/ (로봇은 쓰지 않음). GPU 슬롯은 3개로 두고
# 나머지는 CPU에 두었다가 요청 시 교체(--max-cpu-loras) → 메모리는 어댑터 3개일 때와 같음.
for d in /data/adapters/eval/*/; do
  [ -f "$d/adapter_config.json" ] || continue
  n=$(basename "$d")
  MODS+=("$n=$(readlink -f "$d")")
  r=$(python3 -c "import json;print(json.load(open('$d/adapter_config.json'))['r'])")
  [ "$r" -gt "$MAXR" ] && MAXR=$r
done
SLOTS=${#MODS[@]}; [ "$SLOTS" -gt 3 ] && SLOTS=3
if [ ${#MODS[@]} -gt 0 ]; then
  # LoRA를 켜면 KV 캐시가 모자라서(0.27GiB) CUDA graph를 끔 → KV 1.04GiB 확보 (2026-09-28 rank16 x3 실측)
  [ "${ENFORCE_EAGER:-0}" = "1" ] || ARGS+=(--enforce-eager)
  ARGS+=(--enable-lora --max-loras $SLOTS --max-cpu-loras ${#MODS[@]} --max-lora-rank $MAXR --lora-modules "${MODS[@]}")
  echo "LoRA: ${MODS[*]} (max rank $MAXR)"
else
  echo "LoRA: 없음 (기본 모델만)"
fi

nohup /data/llm/.venv/bin/vllm serve "${ARGS[@]}" > $LOG 2>&1 &
echo $! > $PIDF
echo "started pid $(cat $PIDF), log: $LOG"
