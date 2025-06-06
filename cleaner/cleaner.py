import os
import subprocess

from vllm import LLM, SamplingParams


class cleaner:
    def __init__(self, config: dict={}, debug=None, executable='echo'):
        self.llm = None
        self.params = None
        self.config = config
        self.start_vllm(config)
        self.debug = debug
        self.executable = executable
        pass

    def start_vllm(self, config):
        self.llm = LLM(model=config.get('data',{}).get('model',"meta-llama/Llama-3.2-3B-Instruct"), dtype=config.get('data',{}).get('dtype',"float16"), tensor_parallel_size=config.get('data',{}).get('tensor_parallel_size',4))
        self.params = SamplingParams(temperature=config.get('data',{}).get('temperature',0.7), top_p=config.get('data',{}).get('top_p',0.9), max_tokens=config.get('data',{}).get('max_tokens',2048))

    def generate_prompts(self, target_dir: str=None):
        prompts = []
        if not target_dir:
            target_dir = input("directory to clean: ").strip()
        for file in os.listdir(target_dir):
            if self.debug: print(f"Eating {file}")
            if os.path.isdir(f"{target_dir}/{file}"):
                continue # only read files in current directory 
            file_contents = read_file(f"{target_dir}/{file}")
            try:
                with open(path, 'r') as f:
                    file_contents = f.read()
            except Exception as e:
                continue
            if not file_contents or file_contents == '':
            	continue
            if self.debug: print(file_contents)
            feedback = kube_lint(f"{target_dir}/{file}")
            if self.debug: print(feedback)
            if feedback:
                # print(feedback)
                system_prompt = "You are an expert Kubernetes configuration optimizer. Your task is to ingest a Kubernetes YAML configuration file along with feedback on issues found in the file. Based on the feedback, you must generate a corrected Kubernetes configuration file in proper YAML format."
                user_prompt = f"-----Configuration File Below-----\n{file_contents}\n-----Feedback Below-----\n{feedback}\n------------\nRespond with ONLY the corrected YAML Kubernetes configuration file. Do NOT provide any explaination for the changes made."
                prompts.append({"filename":file,"prompt":f"{system_prompt}\n{user_prompt}"})
        return prompts

    def kube_lint(self, file=None):
        if not file:
            print("ERROR : NO FILE PROVIDED TO KUBE LINTER")
            return None
        r = subprocess.run([f"{self.executable} {file}"], text=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        ans = r.stdout
        if "No lint errors found!" in ans:
            return None
        if self.debug: print(ans)
        return ans