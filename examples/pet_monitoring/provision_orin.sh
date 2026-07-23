#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Provision a device to run the pet-monitoring pack with REAL on-device
# inference, then smoke-test it. Written for an NVIDIA Jetson Orin (the
# reference edge target), but works on any arm64 / x86-64 Linux box.
#
#   bash examples/pet_monitoring/provision_orin.sh
#
# Idempotent: safe to re-run. Installs Ollama, pulls the model, installs the
# Convilyn edge SDK, and runs "Doudou's day" against the local model.
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${CONVILYN_EDGE_MODEL:-qwen3:4b}"
OLLAMA_URL="${CONVILYN_EDGE_MODEL_URL:-http://localhost:11434}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "$HERE/../.." && pwd)"   # examples/pet_monitoring -> sdk/edge-python

echo "==> 1/4  Install Ollama (the on-device inference server)"
if command -v ollama >/dev/null 2>&1; then
  echo "         already installed: $(ollama --version 2>/dev/null | head -n1 || true)"
else
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> 2/4  Start Ollama + pull the model ($MODEL)"
if ! curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "         starting 'ollama serve' in the background (log: /tmp/ollama.log)"
  ( ollama serve >/tmp/ollama.log 2>&1 & )
  # wait (bounded) for the server to answer
  for _ in $(seq 1 30); do
    curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
fi
ollama pull "$MODEL"

echo "==> 3/4  Install the Convilyn edge SDK"
python3 -m pip install --upgrade convilyn-edge

echo "==> 4/4  Smoke test: run 'Doudou's day' against the local model"
cd "$SDK_ROOT"
CONVILYN_EDGE_MODEL_URL="$OLLAMA_URL" CONVILYN_EDGE_MODEL="$MODEL" \
  python3 -m examples.pet_monitoring.run_e2e

cat <<EOF

Provisioned. Run the demo any time:

  cd $SDK_ROOT
  CONVILYN_EDGE_MODEL_URL=$OLLAMA_URL python3 -m examples.pet_monitoring.run_e2e

(Omit CONVILYN_EDGE_MODEL_URL to run the offline stub — no model needed.)
EOF
