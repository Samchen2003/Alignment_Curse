# nohup bash run_text_to_audio_transfer.sh > adv-rene-gpt.log 2>&1 &

MODEL="gpt"
METHOD="renellm"

API_URL="http://192.168.13.91:8000/v1/chat/completions"

if [[ "${MODEL}" == "qwen2.5-omni-7B" ]]; then
    API_URL="http://192.168.13.91:8000/v1/chat/completions"
elif [[ "${MODEL}" == "qwen2.5-omni-3B" ]]; then
    API_URL="http://192.168.13.91:8009/v1/chat/completions"
elif [[ "${MODEL}" == "qwen3-omni" ]]; then
    API_URL="http://192.168.13.91:8900/v1/chat/completions"
fi



# 8000: qwen2.5-omni-7B http://192.168.13.91:8000/v1/chat/completions
# 8009: qwen2.5-omni-3B  http://192.168.13.91:8009/v1/chat/completions
# 8900: qwen3-omni  http://192.168.13.91:8900/v1/chat/completions


TEXT_JSON="/homes/55/samchen/ReNeLLM/rebuttal/advbench_result/${METHOD}-${MODEL}.json"
# TEXT_JSON="/homes/55/samchen/AutoDAN-Turbo/result/autodan-gpt.json"
AUDIO_DIR="/storage3/samchen/results/rebuttal/${METHOD}_${MODEL}_audio"
OUTPUT_JSON="/homes/55/samchen/ReNeLLM/rebuttal/advbench_result/${METHOD}-${MODEL}-final.json"

echo "Running TTS"
python -u generate_audio_naive.py \
  --json_path "${TEXT_JSON}" \
  --output_dir "${AUDIO_DIR}" 

echo "TTS Done. Running audio eval"

if [[ "${MODEL}" == *"gpt"* ]]; then
    echo "Detected GPT model in name ('${MODEL}'), converting audio into WAV format"
    
    NEW_AUDIO_DIR="${AUDIO_DIR}_new"
    mkdir -p "${NEW_AUDIO_DIR}"

    # Loop through WAV files properly
    for f in "${AUDIO_DIR}"/*.wav; do
        base=$(basename "$f" .wav)
        ffmpeg -y -i "$f" \
            -ac 1 -ar 16000 -sample_fmt s16 \
            "${NEW_AUDIO_DIR}/${base}.wav"
    done

    AUDIO_DIR="${NEW_AUDIO_DIR}"
fi

python -u eval_audio_dir.py \
  --audio_dir "${AUDIO_DIR}" \
  --text_json "${TEXT_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --model_name "${MODEL}" \
  --api_url "${API_URL}" 
 
echo "Audio eval done"
