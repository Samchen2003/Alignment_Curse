# The Alignment Curse: Cross-Modality Jailbreak Transfer in Omni-Models

### :book: Table of Contents
- [Installation](#installation)
- [Usage](#usage)

<a name="installation"></a>
## :hammer: Installation

Make sure you have Conda installed (Anaconda or Miniconda)

~~~bash
conda env create -f environment.yml
conda activate alignmentcurse
~~~



<a name="usage"></a>
## :rocket: Usage

Note that we use vllm service for Qwen2.5-Omni-7B, Qwen2.5-Omni-3B, and Qwen3-Omni. Please start the vllm service of these models according to their official documents.
For GPT models we use the official API, please prepare your OPENAI_API_KEY.
For InteractiveOmni we use the official Transformer usage, please download the model checkpoint.

### Naive Attack

```bash
cd naive
# For Qwen models using vllm:
python eval_naive_text.py --input_json <path-to-json> --output_json <output-json> --api_url <url-for-vllm> --model_name qwen
# For gpt model:
export OPENAI_API_KEY="<your-api-key>"
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
export OPENAI_API_KEY="<your-api-key>"
python renellm_omni.py --data_path <path-to-json> --save_suffix <save-suffix> --attack_model gpt
# For InteractiveOmni:
python renellm_omni.py --data_path <path-to-json> --save_suffix <save-suffix> --attack_model io --api_url <ckpt-path>
```

### PAP Attack

```bash
cd PAP
# For Qwen models using vllm:
python eval_pap.py  --qwen_url <url-for-vllm>  --output_json  <output-json>  --model_name qwen
# For gpt model:
export OPENAI_API_KEY="<your-api-key>"
python eval_pap.py   --output_json <output-json>  --model_name gpt
# For InteractiveOmni:
python eval_pap.py  --qwen_url <ckpt-path>     --output_json  <output_json>  --model_name io
```

### AutoDAN-Turbo Attack

```bash
cd AutoDAN-Turbo
# For Qwen models using vllm:
python test.py --target_model qwen  --output_file <output_json>  --api_url <url-for-vllm>
# For gpt model:
export OPENAI_API_KEY="<your-api-key>"
python test.py --target_model gpt  --output_file <output_json>
# For InteractiveOmni:
python test.py --target_model interactiveomni  --output_file <output_json>  --api_url <ckpt-path>
```

