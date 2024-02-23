# LLoVi

This is official implementation for paper: [A Simple LLM Framework for Long-Range Video Question-Answering](https://arxiv.org/pdf/2312.17235.pdf).

## **Installation**

**Install environment.**

Python 3.8 or above is required.

```bash
git clone git@github.com:CeeZh/LLoVi.git
cd LLoVi

python3 -m venv llovi_env
source activate llovi_env/bin/activate
pip install openai
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas
pip install transformers
pip install accelerate
```

**Download dataset annotations and extracted captions.**

Download data.zip from https://drive.google.com/file/d/13M10CB5ePPVlycn754_ff3CwnpPtDfJA/view?usp=drive_link

```bash
unzip data.zip
```

We provide extracted captions for **EgoSchema**, **NeXT-QA**, **NeXT-GQA** and **IntentQA** at `./data`. It also contains dataset annotations.

We used [LaViLa](https://arxiv.org/pdf/2212.04501.pdf) base model to extract EgoSchema captions at 1 FPS, [LLaVA](https://llava-vl.github.io/) ([llava-hf/llava-1.5-13b-hf](https://huggingface.co/llava-hf/llava-1.5-13b-hf)) to extract captions for other datasets at 0.5 FPS. 

Note that LaViLa is trained on Ego4D, which has overlap with EgoSchema. To avoid data leakage, we trained LaViLa using videos that are not in EgoSchema. You can download the model from [this link](https://drive.google.com/file/d/1AZ5I4eTUAUBX31rL8jLB00vNrlFGWiv8/view?usp=drive_link).

**Download experiment results.**

Download output.zip from https://drive.google.com/file/d/1d7a-FuQzdfQ7ZAzU5Y8HJpog1gm_sye_/view?usp=drive_link

```bash
unzip output.zip
```

The result files are generated by running the commands in next sections. Note that the result files will be detected, so the commands in next sections will not run directly. Add `--start_from_scratch` to run the commmands one more time.

## EgoSchema

### Captioner

```bash
# LaViLa
python main.py --output_base_path output/egoschema --output_filename standard_qa.json --api_key YOUR_OPENAI_KEY

# BLIP-2
python main.py --data_path data/egoschema/blip2_fullset.json --output_base_path output/egoschema --output_filename standard_qa_blip2.json --api_key YOUR_OPENAI_KEY
```

| Captioner | LLM | Prompt | Accuracy |
| --- | --- | --- | --- |
| LaViLa | gpt-3.5-turbo | standard | 51.2 |
| BLIP-2 | gpt-3.5-turbo | standard | 47.4 |

### LLM

```bash
# gpt-3.5-turbo
python main.py --output_base_path output/egoschema --output_filename standard_qa.json --api_key YOUR_OPENAI_KEY

# gpt-3.5-turbo-1106
python main.py --model gpt-3.5-turbo-1106 --output_base_path output/egoschema --output_filename standard_qa_1106.json --api_key YOUR_OPENAI_KEY

# llama2 70B
python main.py --model gpt-3.5-turbo-1106 --output_base_path output/egoschema --output_filename llama.json

# gpt-4 (please run gpt-3.5-turbo first as backup, otherwise please disable --backup_pred_path)
python main.py --model gpt-4 --backup_pred_path output/egoschema/standard_qa.json --output_base_path output/egoschema --output_filename standard_qa_gpt4.json --api_key YOUR_OPENAI_KEY

# gpt-4-1106
python main.py --model gpt-4-1106-preview --output_base_path output/egoschema --output_filename standard_qa_gpt4_1106.json --api_key YOUR_OPENAI_KEY
```

| Captioner | LLM | Prompt | Accuracy |
| --- | --- | --- | --- |
| LaViLa | gpt-3.5-turbo | standard | 51.2 |
| LaViLa | gpt-3.5-turbo-1106 | standard | 55.2 |
| LaViLa | https://huggingface.co/meta-llama/Llama-2-70b-chat-hf | standard | 55.4 |
| LaViLa | gpt-4 | standard | 59.0 |
| LaViLa | gpt-4-1106-preview | standard | 61.2 |

### Prompting

```bash
# standard
## gpt-3.5-turbo
python main.py --output_base_path output/egoschema --output_filename standard_qa.json --api_key YOUR_OPENAI_KEY
## gpt-3.5-turbo-1106
python main.py --model gpt-3.5-turbo-1106 --output_base_path output/egoschema --output_filename standard_qa_1106.json --api_key YOUR_OPENAI_KEY

# zero-shot CoT
python main.py --prompt_type qa_zs-cot --output_base_path output/egoschema --output_filename cot.json --api_key YOUR_OPENAI_KEY

# (C, Q) —> S
## gpt-3.5-turbo
### Step 1. generate summary for each example. 
python main.py --task sum --prompt_type sum_q --num_words_in_sum 500 --temperature 1.0 --output_base_path output/egoschema --output_filename sum_q_500.json --api_key YOUR_OPENAI_KEY
### Step 2. feed the summary (instead of raw captions) to the LLM. 
python main.py --prompt_type qa_sum --data_path output/egoschema/sum_q_500_data.json --output_base_path output/egoschema --output_filename qa_sum_q_500.json --api_key YOUR_OPENAI_KEY
## gpt-3.5-turbo-1106
### Step 1. generate summary for each example. 
python main.py --model gpt-3.5-turbo-1106 --task sum --prompt_type sum_q --num_words_in_sum 500 --temperature 1.0 --output_base_path output/egoschema --output_filename sum_q_500_1106.json --api_key YOUR_OPENAI_KEY
### Step 2. feed the summary (instead of raw captions) to the LLM. 
python main.py --model gpt-3.5-turbo-1106 --prompt_type qa_sum --data_path output/egoschema/sum_q_500_1106_data.json --output_base_path output/egoschema --output_filename qa_sum_q_500_1106.json --api_key YOUR_OPENAI_KEY
```

| Captioner | LLM | Prompt | Accuracy |
| --- | --- | --- | --- |
| LaViLa | gpt-3.5-turbo | standard | 51.2 |
| LaViLa | gpt-3.5-turbo | zero-shot CoT | 55.2 |
| LaViLa | gpt-3.5-turbo | (C, Q) —> S | 57.4 |
| LaViLa | gpt-3.5-turbo-1106 | standard | 55.2 |
| LaViLa | gpt-3.5-turbo-1106 | (C, Q) —> S | 58.8 |

### Few shot

```bash
# standard
python main.py --fewshot_example_path data/egoschema/few_shot_6.json --backup_pred_path output/egoschema/standard_qa.json --prompt_type qa_standard_fewshot --output_base_path output/egoschema --output_filename fewshot.json --api_key YOUR_OPENAI_KEY

# (C, Q) --> S
### Step  1. generate summary for each example. 
python main.py --task sum --prompt_type sum_q --num_words_in_sum 500 --temperature 1.0 --output_base_path output/egoschema --output_filename sum_q_500.json --api_key YOUR_OPENAI_KEY
### (Optional) Step 2. QA without few-shot examples. Use the result as backup predictions. Otherwise, please disable backup_pred_path in the next command. 
python main.py --prompt_type qa_sum --data_path output/egoschema/sum_q_500_data.json --output_base_path output/egoschema --output_filename qa_sum_q_500.json --api_key YOUR_OPENAI_KEY
### Step 3. QA with few-shot examples. 
python main.py --prompt_type qa_standard_fewshot --fewshot_example_path data/egoschema/few_shot_6.json --backup_pred_path output/egoschema/qa_sum_q_500.json --data_path output/egoschema/sum_q_500_data.json --output_base_path output/egoschema --output_filename fewshot_sum.json --api_key YOUR_OPENAI_KEY
```

| Captioner | LLM | Prompt | Accuracy |
| --- | --- | --- | --- |
| LaViLa | gpt-3.5-turbo | standard | 57.6 |
| LaViLa | gpt-3.5-turbo | (C, Q) —> S | 60.2 |

### Accuracy on different categories

```bash
python eval.py \
--function eval_egoschema_cats \
data_path output/egoschema/standard_qa.json \
cats_path data/egoschema/categories.json
```

| Category | Percentage | Accuracy |
| --- | --- | --- |
| Purpose/Goal Identification | 49.2 | 50.4 |
| Tools and Materials Usage | 21.8 | 55.0 |
| Key Action/Moment Detection | 21.6 | 43.5 |
| Action Sequence Analysis | 18.2 | 50.5 |
| Character Interaction | 9.4 | 63.8 |

## NeXT-QA

```bash
python main.py \
--dataset nextqa \
--data_path data/nextqa/llava1.5_fps1.json \
--fps 0.5 \
--anno_path data/nextqa/val.csv \
--duration_path data/nextqa/durations.json \
--prompt_type qa_next \
--model gpt-4-1106-preview \
--output_base_path output/nextqa \
--output_filename gpt4_llava.json \
--api_key YOUR_OPENAI_KEY
```

Accuracy: 

| Why  | How | Bef&Aft | When | Cnt | Loc | Other | Acc_C | Acc_T | Acc_D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 70.95 | 65.45 | 54.69 | 70.14 | 58.76 | 82.71 | 78.36 | 69.51 | 61.04 | 75.55 |

## Intent-QA

```bash
python main.py \
--dataset intentqa \
--data_path data/nextqa/llava1.5_fps1.json \
--fps 0.5 \
--anno_path data/intentqa/test.csv \
--duration_path data/nextqa/durations.json \
--prompt_type qa_next \
--model gpt-4-1106-preview \
--output_base_path output/intentqa \
--output_filename gpt4_llava.json \
--api_key YOUR_OPENAI_KEY
```

| Why | How | Bef&Aft | Total |
| --- | --- | --- | --- |
| 68.40 | 67.41 | 51.05 | 63.96 |

## NeXT-GQA

```bash
# Step 1. QA
python main.py \
--dataset nextgqa \
--data_path data/nextqa/llava1.5_fps1.json \
--fps 0.5 \
--anno_path data/nextgqa/test.csv \
--duration_path data/nextqa/durations.json \
--prompt_type qa_next \
--model gpt-4-1106-preview \
--output_base_path output/nextgqa \
--output_filename gpt4_llava.json \
--api_key YOUR_OPENAI_KEY

# Step 2. Grounding
python main.py \
--dataset nextgqa \
--data_path data/nextqa/llava1.5_fps1.json \
--fps 0.5 \
--anno_path data/nextgqa/test.csv \
--duration_path data/nextqa/durations.json \
--nextgqa_gt_ground_path data/nextgqa/gsub_test.json \
--nextgqa_pred_qa_path output/nextgqa/gpt4_llava.json \
--prompt_type gqa \
--task gqa \
--model gpt-4-1106-preview \
--output_base_path output/nextgqa \
--output_filename gpt4_llava_grounding.json \
--save_info
--api_key YOUR_OPENAI_KEY
```

| Acc&GQA | mIoP | TIoP@0.3 | TIoP@0.5 | mIoU | TIoU@0.3 | TIoU@0.5 |
| --- | --- | --- | --- | --- | --- | --- |
| 24.3 | 37.3 | 45.0 | 36.9 | 20.0 | 29.1 | 15.3 |

## Debug

```bash
--save_info: save more information, e.g. token usage, detailed prompts, etc.
--num_examples_to_run: how many examples to run. -1 (default) to run all.
--start_from_scratch: ignore existing output files. Start from scratch.
```

# Citation
If you find this repository useful for your research, please consider citing our work:
```
@misc{zhang2023simple,
      title={A Simple LLM Framework for Long-Range Video Question-Answering}, 
      author={Ce Zhang and Taixi Lu and Md Mohaiminul Islam and Ziyang Wang and Shoubin Yu and Mohit Bansal and Gedas Bertasius},
      year={2023},
      eprint={2312.17235},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
