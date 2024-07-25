#!/bin/bash
#SBATCH --output=cache/logs/rollout/%j.log
#SBATCH --error=cache/logs/rollout/%j.log
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32Gb
#SBATCH -c 6
#SBATCH --gres=gpu:a100l:1
#SBATCH --nodes=1
#SBATCH --partition=unkillable


MODEL_NAME="lmsys/vicuna-13b-v1.5"
TEMPLATE_JSON="prm_pipeline/rollout_generator/templates/example-20231031b1.json"
DATASET_NAME="data/liar-new-filtered-20231031a2-openai"

# Set BASE URL
# export WEBSERVER_URL=

export HUGGINGFACE_HUB_CACHE="$SCRATCH/.cache/"
export TRANSFORMERS_CACHE="$SCRATCH/.cache"
module load python/3.10
module load cuda/11.1/cudnn/8.1
source env/bin/activate

# Get port number for vLLM server
BASE_PORT=50000
VLLM_PORT=$((BASE_PORT + SLURM_JOB_ID % 1000))
export OPENAI_API_BASE=http://127.0.0.1:${VLLM_PORT}/v1
export OPENAI_API_KEY=EMPTY
echo OPENAI_API_BASE: ${OPENAI_API_BASE}

# Launch vLLM in background
source ~/vllm/env/bin/activate
python3 -m vllm.entrypoints.openai.api_server --model=${MODEL_NAME} --port=${VLLM_PORT} > /dev/null 2>&1  &

# Loop and check for the HTTP listener
while true; do
    # Try to connect to the listener; if successful, break out of the loop
    curl --max-time 5 http://localhost:${VLLM_PORT} > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        break
    fi
    # Wait for a short time before trying again
    sleep 5
done

# Launch AI preference generator
source env/bin/activate

python3 -m prm_pipeline.rollout_generator.worker  \
 --model_name="lmsys/vicuna-13b-v1.5"  \
 --template_json=${TEMPLATE_JSON}  \
 --dataset_name=${DATASET_NAME}  \
 --num_processes=128  \
 --num_repeats=8