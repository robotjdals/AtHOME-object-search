# AtHOME LLM 서버 환경: 캐시·설치를 모두 정해진 폴더 안에 둠.
# 기본값은 이전 서버 배치(/data/llm, /data/.cache). 다른 배치는 serve.conf에서 바꿈:
#   CACHE_ROOT     캐시 (uv, pip, Hugging Face, vLLM, triton)
#   TOOLS_DIR      uv 실행 파일(bin/)과 uv가 설치한 Python(python/)
#   ISOLATED_HOME  설정하면 HOME·XDG_CONFIG_HOME·TMPDIR를 이 폴더로 돌림
#                  (공유 서버에서 root 홈이나 /tmp에 아무것도 쓰지 않기 위해)
: "${CACHE_ROOT:=/data/.cache}"
: "${TOOLS_DIR:=/data/llm}"
export UV_INSTALL_DIR=$TOOLS_DIR/bin
export UV_CACHE_DIR=$CACHE_ROOT/uv
export UV_PYTHON_INSTALL_DIR=$TOOLS_DIR/python
export PIP_CACHE_DIR=$CACHE_ROOT/pip
export HF_HOME=$CACHE_ROOT/huggingface
export VLLM_CACHE_ROOT=$CACHE_ROOT/vllm
export XDG_CACHE_HOME=$CACHE_ROOT
export TRITON_CACHE_DIR=$CACHE_ROOT/triton
export PATH=$TOOLS_DIR/bin:$PATH
if [ -n "${ISOLATED_HOME:-}" ]; then
  mkdir -p "$ISOLATED_HOME/.config" "$ISOLATED_HOME/tmp"
  export HOME=$ISOLATED_HOME XDG_CONFIG_HOME=$ISOLATED_HOME/.config TMPDIR=$ISOLATED_HOME/tmp
fi
# vLLM 사용 통계 전송·기록 끔
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
