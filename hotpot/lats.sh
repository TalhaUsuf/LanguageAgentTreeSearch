python run.py \
    --backend meta-llama/Meta-Llama-3.1-70B-Instruct \
    --task_start_index 0 \
    --task_end_index 100 \
    --n_generate_sample 5 \
    --n_evaluate_sample 1 \
    --prompt_sample cot \
    --algorithm lats \
    --temperature 1.0 \
    --iterations 30 \
    --log logs/tot_10k.log \
    ${@}
