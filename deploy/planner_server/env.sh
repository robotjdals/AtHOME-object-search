# AtHOME LLM 서버 환경 (모든 캐시/설치를 /data 안에 둠)
export UV_INSTALL_DIR=/data/llm/bin
export UV_CACHE_DIR=/data/.cache/uv
export UV_PYTHON_INSTALL_DIR=/data/llm/python
export PIP_CACHE_DIR=/data/.cache/pip
export HF_HOME=/data/.cache/huggingface
export VLLM_CACHE_ROOT=/data/.cache/vllm
export XDG_CACHE_HOME=/data/.cache
export TRITON_CACHE_DIR=/data/.cache/triton
export PATH=/data/llm/bin:$PATH
