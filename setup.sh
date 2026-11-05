#!/bin/bash
set -e

# Create a temporary environment if needed
if [ ! -f /tmp/aka306-llm-int8-venv/bin/activate ]; then
    python3 -m venv /tmp/aka306-llm-int8-venv
fi

# Activate the environment
source /tmp/aka306-llm-int8-venv/bin/activate

# Install project dependencies
python -m pip install -r requirements.txt

# Verify PyTorch GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
