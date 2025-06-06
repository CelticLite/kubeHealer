import argparse
import json

from cleaner import Cleaner

def process_args():
    parser = argparse.ArgumentParser(description ='Fix errors found by a linter')
    parser.add_argument('-t','--target', help ='target directory containing dirty yaml', default=None)
    parser.add_argument('--debug', help ='run in debug mode', action="store_true")
    parser.add_argument('-c','--clean', help ='output directory containing cleaned yaml', default=None)
    return parser.parse_args()

def build_options(file=None):
    if file:
        with open(file, 'r') as f:
            opts = json.load(f)
    else:
        opts = {'version' : '1', 'data' : {'model' : 'meta-llama/Llama-3.2-3B-Instruct', 'dtype': 'float16', 'tensor_parallel_size': 4}}
    return opts

def main():
    wrong = 0
    right = 0 

    args = parse_args()
    if args.options:
        opts = build_options(file=args.options)
    else:
        opts = build_options()
    c = Cleaner(config=opts, debug=args.debug, executable="kube-linter lint")
    if args.debug: print(f"SamplingParams set: {c.params}")
    prompts = c.generate_prompts(target_dir=args.target)
    try:
        if args.debug: print(prompts)
        llm_batched_input = [p.get("prompt",None) for p in prompts]
        outputs = c.llm.generate(llm_batched_input, sampling_params=c.params)
        print("All outputs collected")
    except Exception as e:
        print(e)
        outputs = []
        pass
    if not args.clean:
        clean_dir = input("output directory to write new yaml to: ").strip()
    else:
        clean_dir = args.clean
    for i, output in enumerate(outputs):
        prompt = output.prompt
        res = ""
        for o in output.outputs:
            res += o.text
        if '```yml' in res:
            res = res.split('```yml')[1]
            res = res.split('```')[0]
        if '```yaml' in res:
            res = res.split('```yaml')[1]
            res = res.split('```')[0]
        with open(f'{clean_dir}/{prompts[i].get("filename","something")}', 'w') as f:
            f.write(res)
        still_wrong = c.kube_lint(f'{clean_dir}/{prompts[i].get("filename","something")}')
        if still_wrong:
            print(f'File: {prompts[i].get("filename","something")} still has errors')
            wrong += 1
        else:
            right += 1
    print(f"Guessed {right} correct fixes!")
    print(f"Missed {wrong} corrections.")

if __name__ == '__main__':
    main()