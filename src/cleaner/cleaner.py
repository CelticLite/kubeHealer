import os
import subprocess

from vllm import LLM, SamplingParams


class cleaner:
    def __init__(self, config: dict={}, debug=None, executable='echo', system_prompt: str="Tell a funny joke based on the following file and feedback.", base_case: str="Th!sHappensToBeMyPassw0rd"):
        self.config = config
        self.llm = LLM(model=self.config.get('data',{}).get('model',"meta-llama/Llama-3.2-3B-Instruct"), dtype=self.config.get('data',{}).get('dtype',"float16"), tensor_parallel_size=self.config.get('data',{}).get('tensor_parallel_size',4))
        self.params = SamplingParams(temperature=self.config.get('data',{}).get('temperature',0.7), top_p=self.config.get('data',{}).get('top_p',0.9), max_tokens=self.config.get('data',{}).get('max_tokens',2048))
        self.debug = debug
        self.executable = executable
        self.system_prompt = system_prompt
        self.base_case = base_case
        pass

    # returns strings to be used as prompts in LLM call functions each labeled with the filename of the 
    def generate_prompts(self, target_dir: str=None):
        prompts = []
        if not target_dir:
            target_dir = input("directory to clean: ").strip()
        for file in os.listdir(target_dir):
            if self.debug: print(f"Eating {file}")
            if os.path.isdir(f"{target_dir}/{file}"):
                continue # only read files in current directory 
            try:
                with open(path, 'r') as f:
                    file_contents = f.read()
            except Exception as e:
                continue
            if not file_contents or file_contents == '':
                continue
            if self.debug: print(file_contents)
            feedback = generate_feedback(f"{target_dir}/{file}")
            if self.debug: print(feedback)
            if feedback:
                # print(feedback)
                user_prompt = f"-----Configuration File Below-----\n{file_contents}\n-----Feedback Below-----\n{feedback}\n------------\nRespond with ONLY the corrected file. Do NOT provide any explaination for the changes made."
                prompts.append({"filename":file,"prompt":f"{self.system_prompt}\n{user_prompt}"})
        return prompts

    # returns string output from the given executable
    def generate_feedback(self, file=None):
        if not file:
            print("ERROR : NO FILE PROVIDED TO KUBE LINTER")
            return None
        r = subprocess.run([f"{self.executable} {file}"], text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ans = r.stdout
        if self.base_case in ans:
            return None
        if self.debug: print(ans)
        return ans

    def generate(prompts, save: bool=False, clean_dir: str="/tmp"):
        llm_batched_input = [p.get("prompt",None) for p in prompts]
        outputs = self.llm.generate(llm_batched_input, sampling_params=self.params)
        str_outputs = []
        for i, output in enumerate(outputs):
            res = ""
            for o in output.outputs:
                res += o.text
            str_outputs.append(res)
        if save:
            with open(f'{clean_dir}/{prompts[i].get("filename","something")}', 'w') as f:
                f.write(res)
        return str_outputs