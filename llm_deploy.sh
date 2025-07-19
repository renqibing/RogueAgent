# MODEL_NAME="/mnt/hwfile/trustai/models/Meta-Llama-3-8B-Instruct"
MODEL_NAME="/mnt/hwfile/trustai/models/Llama-3-8B-Lexi-Uncensored"
srun -p AI4Good_S --gres=gpu:1 -J oasis-rqb python -m vllm.entrypoints.openai.api_server --model $MODEL_NAME --dtype auto --port 8500 --served-model-name 'llama-3' --trust-remote-code --disable-log-stats