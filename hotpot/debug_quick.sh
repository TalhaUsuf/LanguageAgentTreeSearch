#!/bin/bash

# Quick debug script for LATS workflow
# This mirrors lats.sh but with reduced parameters for faster debugging

echo "🐛 LATS Quick Debug Script"
echo "========================="

# Set environment variables
export LLM=custom
export OPENAI_API_BASE=http://69.48.159.10:30000/v1
export OPENAI_MODEL=llama-3.1-70b

echo "🌐 Environment:"
echo "   LLM: $LLM"
echo "   API_BASE: $OPENAI_API_BASE"
echo "   MODEL: $OPENAI_MODEL"

# Create logs directory if it doesn't exist
mkdir -p logs

echo ""
echo "🚀 Running debug version with reduced parameters..."
echo ""

python run.py \
    --backend "llama-3.1-70b" \
    --task_start_index 0 \
    --task_end_index 2 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --algorithm lats \
    --temperature 1.0 \
    --iterations 10 \
    --log logs/debug_quick.log \
    ${@}

echo ""
echo "🏁 Debug run completed!"
echo "📋 Check logs/debug_quick.log for detailed output"
