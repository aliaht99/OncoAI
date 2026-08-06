#!/usr/bin/env bash
# OncoAI — one-command launcher
set -euo pipefail
cd "$(dirname "$0")"

if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "/opt/anaconda3/envs/aicd/bin/python" ]; then
    PY="/opt/anaconda3/envs/aicd/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

if ! "$PY" -c "import streamlit" 2>/dev/null; then
    echo "✗ Streamlit not installed in $PY"
    echo "   $PY -m pip install -r requirements.txt"
    exit 1
fi

echo "▶ Python: $PY"
exec "$PY" -m streamlit run app.py \
    --server.port "${PORT:-8501}" \
    --server.address 127.0.0.1 \
    --browser.gatherUsageStats false
