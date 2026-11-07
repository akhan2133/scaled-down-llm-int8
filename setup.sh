#!/bin/bash
set -e

VENV_DIR="/tmp/${USER}-llm-int8-venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install -r requirements.txt

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "To activate the environment, run:"
echo "source $VENV_DIR/bin/activate"