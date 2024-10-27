import openai
from openai import OpenAI
from transformers import AutoTokenizer
import torch
import transformers
from prompts import identity
import pdb
from pprint import pprint
import time


def get_model(args):
    model_name, temperature, max_new_tokens = args.model, args.temperature, args.max_new_tokens
    if 'gpt' in model_name:
        model = GPT(args.api_key, model_name, temperature)
        return model
    elif 'Llama-2' in model_name:
        return LLaMA2(model_name, temperature, max_new_tokens)
    elif 'Llama-3' in model_name:
        return LLaMA3(model_name, temperature, max_new_tokens) 


class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, api_key, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)

    def get_response(self, **kwargs):
        try:
            res = self.client.chat.completions.create(**kwargs)
            return res
        except openai.APIConnectionError as e:
            print('APIConnectionError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.RateLimitError as e:
            print('RateLimitError')
            time.sleep(10)
            return self.get_response(**kwargs)
        except openai.APITimeoutError as e:
            print('APITimeoutError')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.BadRequestError as e:
            print('BadRequestError')
            kwargs['messages'] = [{
                "role": "user", "content": "Randomly return one letter from A, B, C, D."
            }]
            return self.get_response(**kwargs)

    def forward(self, head, prompts):
        messages = [
            {"role": "system", "content": head}
        ]
        info = {}
        for i, prompt in enumerate(prompts):
            messages.append(
                {"role": "user", "content": prompt}
            )
            response = self.get_response(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )
            info = dict(response.usage)  # completion_tokens, prompt_tokens, total_tokens
            info['response'] = messages[-1]["content"]
            info['message'] = messages
        return self.post_process_fn(info['response']), info


class LLaMA2(Model):
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = "[PAD]"
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            tokenizer=tokenizer,
            temperature=temperature
        )

    def forward(self, head, prompts):
        prompt = prompts[0]
        sequences = self.pipeline(
            prompt,
            do_sample=False,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
        )
        response = sequences[0]['generated_text']  # str
        info = {
            'message': prompt,
            'response': response
        }
        return self.post_process_fn(info['response']), info


class LLaMA3(Model):
    def __init__(self, model_name, temperature, max_new_tokens):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.pipeline = transformers.pipeline(
            "text-generation", 
            model=model_name, 
            model_kwargs={"torch_dtype": torch.float16}, 
            device_map="auto",
        )

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def forward(self, head, prompts):
        prompt = prompts[0]
        messages = [
            {"role": "system", "content": head},
            {"role": "user", "content": prompt}
        ]
        sequences = self.pipeline(
            messages,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=False,
            temperature=self.temperature,
        )
        response = sequences[0]["generated_text"][-1]["content"]
        info = {
            'message': prompt,
            'response': response
        }
        return self.post_process_fn(info['response']), info
