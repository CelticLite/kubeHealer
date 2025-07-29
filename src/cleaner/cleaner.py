import os
import subprocess
import json
import gc
import torch

from typing import List, Optional, Tuple

from vllm import LLM, SamplingParams, EngineArgs, LLMEngine, RequestOutput
from vllm.lora.request import LoRARequest



from huggingface_hub import snapshot_download




class cleaner:
    def __init__(self, config: dict={}, debug=None, executable='echo', system_prompt="Tell a funny joke based on the following file and feedback.", base_case="Th!sHappensToBeMyPassw0rd"):
        self.config = config
        self.lora_repo = config.get('data', {}).get('lora_repo','jashing/tinyllama-colorist-lora')
        if config.get('data',{}).get('lora_repo',False):
            self.llm = LLM(model=self.config.get('data',{}).get('model',"meta-llama/Llama-3.2-3B-Instruct"), dtype=self.config.get('data',{}).get('dtype',"float16"), tensor_parallel_size=self.config.get('data',{}).get('tensor_parallel_size',4), enable_lora=True)
        else:
            self.llm = LLM(model=self.config.get('data',{}).get('model',"meta-llama/Llama-3.2-3B-Instruct"), dtype=self.config.get('data',{}).get('dtype',"float16"), tensor_parallel_size=self.config.get('data',{}).get('tensor_parallel_size',4))
        self.params = SamplingParams(temperature=self.config.get('data',{}).get('temperature',0.7), top_p=self.config.get('data',{}).get('top_p',0.9), max_tokens=self.config.get('data',{}).get('max_tokens',2048))
        self.debug = debug
        self.executable = executable
        self.system_prompt = system_prompt
        self.base_case = base_case
        self.prompts = []
        pass

    def initialize_engine(self, model: str, quantization: str,
                      lora_repo: Optional[str]) -> LLMEngine:
        """Initialize the LLMEngine."""

        if quantization == "bitsandbytes":
            engine_args = EngineArgs(model=model,
                                     quantization=quantization,
                                     qlora_adapter_name_or_path=lora_repo,
                                     load_format="bitsandbytes",
                                     enable_lora=True,
                                     max_lora_rank=8,
                                     cpu_offload_gb=5,
                                     max_model_len=12999)
        else:
            engine_args = EngineArgs(model=model,
                                     quantization=quantization,
                                     enable_lora=True,
                                     max_loras=4)
        return LLMEngine.from_engine_args(engine_args)

    # returns strings to be used as prompts in LLM call functions each labeled with the filename of the 
    def generate_prompts(self, target_dir: str=None, max_tokens: int=131072):
        if not target_dir:
            target_dir = input("directory to clean: ").strip()
        for file in os.listdir(target_dir):
            if self.debug: print(f"Eating {file}")
            if os.path.isdir(f"{target_dir}/{file}"):
                continue # only read files in current directory 
            try:
                with open(f"{target_dir}/{file}", 'r') as f:
                    file_contents = f.read()
            except Exception as e:
                print(e)
                continue
            if not file_contents or file_contents == '':
                print(f"NO FILE CONTENTS FOR {file}")
                continue
            if self.debug: print(file_contents)
            feedback = self.generate_feedback(f"{target_dir}/{file}")
            if self.debug: print(feedback)
            if feedback:
                # print(feedback)
                user_prompt = f"-----Configuration File Below-----\n{file_contents}\n-----Feedback Below-----\n{feedback}\n------------\nRespond with ONLY the corrected file. Do NOT provide any explaination for the changes made."
                if len(user_prompt) > (max_tokens * 0.5):
                    print(f"TRIMMED USER'S PROMPT FOR {file}")
                    user_prompt = user_prompt.replace(feedback, "There is no given feedback. Use your own thoughts to provide a more secure file.")
                self.prompts.append({"filename":file,"prompt":f"{self.system_prompt}\n{user_prompt}"})
        return self.prompts

    # returns string output from the given executable
    def generate_feedback(self, file=None):
        if not file:
            print("ERROR : NO FILE PROVIDED TO EXECUTABLE")
            return None
        r = subprocess.run([f"{self.executable} {file}"], text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ans = r.stdout
        if self.base_case in ans:
            return None
        if self.debug: print(ans)
        return ans

    def set_prompts_from_file(self, filename=None):
        if not filename:
            print("ERROR : NO FILE PROVIDED TO READ PROMPTS FROM")
            return None
        data = {}
        with open(filename, 'r') as file:
            data = json.load(file)
        print(f"Prompt len: {len(data)}")
        self.prompts = data
        return data

    def generate(self, save_files: bool=False, clean_dir: str="/tmp"):
        print("starting generate")
        llm_batched_input = [p.get("prompt",None)[:2048] for p in self.prompts]
        print(f"Inputs to LLM: {len(llm_batched_input)}")
        if self.config.get('data',{}).get('lora_repo',False):
            outputs = self.llm.generate(llm_batched_input, sampling_params=self.params, lora_request=LoRARequest("lora-test", 1, self.lora_repo))
        else:
            outputs = self.llm.generate(llm_batched_input, sampling_params=self.params)
        str_outputs = []
        for i, output in enumerate(outputs):
            res = ""
            for o in output.outputs:
                res += o.text
            # # THE FOLLOWING SHOULD BE ADDED IN PRODUCTION ENV TO REDUCE FORMATTING ERRORS
            # if '```yaml' in res:
            #     res = res.split('```yaml',1)[-1]
            #     res = res.split('```',1)[0]
            # if '---' in res:
            #     res = res.split('---',1)[-1]
            #     res = res.split('---',1)[0]
            str_outputs.append(res)
            if save_files:
                with open(f'{clean_dir}/{self.prompts[i].get("filename","something")}', 'w') as f:
                    f.write(res)
        return str_outputs