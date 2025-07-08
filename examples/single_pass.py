import argparse
import json

from src.cleaner import cleaner

def process_args():
    parser = argparse.ArgumentParser(description ='Fix errors found by a linter')
    parser.add_argument('-t','--target', help ='target directory containing dirty yaml', default=None)
    parser.add_argument('-c','--config', help ='json options file', default=None)
    parser.add_argument('--debug', help ='run in debug mode', action="store_true")
    parser.add_argument('-o','--output', help ='output directory containing cleaned yaml', default=None)
    parser.add_argument('-p','--prompts', help ='prompts to use for batched inference', default=None)
    return parser.parse_args()
def build_options(file=None):
    if file:
        with open(file, 'r') as f:
            opts = json.load(f)
    else:
        opts = {'version' : '1', 'data' : {'model' : 'meta-llama/Llama-3.2-3B-Instruct', 'dtype': 'float16', 'tensor_parallel_size': 4}}
    return opts

def save_to_file(d, filename):
    with open(filename, 'w') as file:
        file.write(json.dumps(d))

def main():
    wrong = 0
    right = 0 

    args = process_args()
    if args.config:
        opts = build_options(file=args.config)
    else:
        opts = build_options()
    c = cleaner.cleaner(config=opts, debug=args.debug, executable="kube-linter lint", system_prompt="You are an expert Kubernetes configuration optimizer. Your task is to ingest a Kubernetes YAML configuration file along with feedback on issues found in the file. Based on the feedback, you must generate a corrected Kubernetes configuration file in proper YAML format.", base_case="No lint errors found")
    if args.debug: print(f"SamplingParams set: {c.params}")
    if args.prompts:
        prompts = c.set_prompts_from_file(args.prompts)
    else:
        prompts = c.generate_prompts(target_dir=args.target)
        save_to_file(prompts, 'prompts.json')
    if not args.output:
        clean_dir = input("output directory to write new yaml to: ").strip()
    else:
        clean_dir = args.output
    try:
        if args.debug: print(prompts)
        outputs = c.generate(prompts, clean_dir=clean_dir)
        print("All outputs collected")
    except Exception as e:
        print(e)
        outputs = []
        pass
    for i, output in enumerate(outputs):
        still_wrong = c.generate_feedback(f'{clean_dir}/{prompts[i].get("filename","something")}')
        if still_wrong:
            print(f'File: {prompts[i].get("filename","something")} still has errors')
            wrong += 1
        else:
            right += 1
    print(f"Guessed {right} correct fixes!")
    print(f"Missed {wrong} corrections.")

if __name__ == '__main__':
    main()
