# The Alignment Curse: Cross-Modality Jailbreak Transfer in Omni-Models

### :book: Table of Contents
- [Installation](#installation)
- [Usage](#usage)

<a name="installation"></a>
## :hammer: Installation

Make sure you have Conda installed (Anaconda or Miniconda)

~~~bash
conda env create -f environment.yml
pip install -r requirements.txt
pip install flash-attn --no-build-isolation --no-cache-dir
pip install git+https://github.com/dsbowen/strong_reject.git@main 
~~~



<a name="usage"></a>
## :rocket: Usage

Note that we use vllm service for Qwen2.5-Omni-7B, Qwen2.5-Omni-3B, and Qwen3-Omni. Please start the vllm service of these models according to their official documents.
For GPT models we use the official API, please prepare your OPENAI_API_KEY.
For InteractiveOmni we use the official Transformer usage, please download the model checkpoint.

```bash
export OPENAI_API_KEY="<your-api-key>"
```

### Naive Attack

```bash
cd naive
# For Qwen models using vllm:
python eval_naive_text.py --input_json <path-to-json> --output_json <output-json> --api_url <url-for-vllm> --model_name qwen
# For gpt model:
python eval_naive_text.py --input_json <path-to-json> --output_json <output_json> --api_url <url_for_vllm> --model_name gpt
# For InteractiveOmni:
python eval_naive_text.py --input_json <path-to-json> --output_json <output_json> --api_url <ckpt-path> --model_name io
```

### ReNeLLM Attack

```bash
cd ReNeLLM
# For Qwen models using vllm:
python renellm_omni.py --data_path <path-to-json> --save_suffix <save-suffix> --attack_model qwen --api_url <url_for_vllm>
# For gpt model:
python renellm_omni.py --data_path <path-to-json> --save_suffix <save-suffix> --attack_model gpt
# For InteractiveOmni:
python renellm_omni.py --data_path <path-to-json> --save_suffix <save-suffix> --attack_model io --api_url <ckpt-path>
```

### PAP Attack

```bash
cd PAP
# For Qwen models using vllm:
python eval_pap.py  --dataset_path <path-to-json>    --qwen_url <url-for-vllm>  --output_json  <output-json>  --model_name qwen
# For gpt model:
python eval_pap.py  --dataset_path <path-to-json>    --output_json <output-json>  --model_name gpt
# For InteractiveOmni:
python eval_pap.py  --dataset_path <path-to-json>    --qwen_url <ckpt-path>     --output_json  <output_json>  --model_name io
```

### AutoDAN-Turbo Attack

```bash
cd AutoDAN-Turbo
cd llm
git clone https://github.com/chujiezheng/chat_templates.git
cd ..
# For Qwen models using vllm:
python test.py --input_file <path-to-json> --target_model qwen  --output_file <output_json>  --api_url <url-for-vllm>
# For gpt model:
python test.py --input_file <path-to-json> --target_model gpt  --output_file <output_json>
# For InteractiveOmni:
python test.py --input_file <path-to-json> --target_model interactiveomni  --output_file <output_json>  --api_url <ckpt-path>
```

### VJ Attack

```bash
cd VJ
# For Qwen models using vllm:
python qwen.py --dataset <path-to-json> --prompt-audio-dir <audio-files-dir> --output-dir <output-dir> --model <model-name> --api-url <url-for-vllm>
# For gpt model:
python gpt.py --dataset <path-to-json> --original-audio-dir <audio-files-dir> --output-dir <output-dir>  --model <model-name>
# For InteractiveOmni:
python io.py --dataset <path-to-json> --model-path <ckpt-path> --prompt-audio-dir <audio-files-dir> --output-dir <output-dir>  --model <model-name>
```

### Speech Editing Attack
You can use different TTS engines to generate different variations of the audio and pass in directory name and variation name.
```bash
cd SE
# For Qwen models using vllm:
python qwen.py --goal-target-path <path-to-json>  --model-name <model-name>  --api-url <url-for-vllm>     --variations-base <directory-for-audio-variations>  --variant-names <names-of-audio-variations>
# For gpt model:
python gpt.py --goal-target-path <path-to-json> --variations-base <directory-for-audio-variations>  --variant-names <names-of-audio-variations>
# For InteractiveOmni:
python io.py --goal-target-path <path-to-json> --model-name io  --model-path <ckpt-path>    --variations-base <directory-for-audio-variations>  --variant-names <names-of-audio-variations>
```

### Text Transferred Audio Attack

```bash
python generate_audio.py --json_path <text-attack-result-json> --output_json <output-json> --output_dir <output-dir-for-audio>
python eval_audio_json.py --text_json <output-json-from-audio-generation> --output_json <final-output-json> --api_url <vllm-api-for-qwen-or-interactiveomni-ckpt>  --model_name <model_name>
```

### KL Estimation

```bash
# First extract representations following official Qwen/InteractiveOmni Transformer implementation and save the representations in an npz file.
python kl_estimate.py --path <path-to-embedding.npz>  
```

