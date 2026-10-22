# LLM-int8
Optimizing a Scaled Down LLM.int8() Implementation for RTX 2000 GPUs

CMPT 479 Fall 2026
Based on: https://www.alphaxiv.org/pdf/2208.07339

# Quickstart
```
# First time only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build and run
cmake --build build -j
./build/llm_int8

# Or, for convenience
cmake --build build --target run -j
```