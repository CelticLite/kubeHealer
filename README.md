# Kubernetes File Cleaner

A Python library for automating the process of cleaning and formatting Kubernetes configuration files using large language models (LLMs).

## Features  

- Generates text prompts for LLMs to correct Kubernetes configuration files  
- Uses `vllm` library for efficient LLM interactions  
- Supports customizable model, data type, and tensor parallel size configurations  
- Integrates with external tools (e.g., `kube-lint`) for analysis and feedback  

## Installation

To install this library, run:  
```bash  
pip install git+https://github.com/celticlite/kubeHealer.git  
```  
Alternatively, you can clone the repository manually and install it using pip:  
```bash  
git clone https://github.com/celticlite/kubeHealer.git  
cd kubeHealer  
pip install .  
```  

## Usage  

To use this library in your Python program, simply import the `cleaner` class:  
```python
import cleaner
```
Create an instance of the `cleaner` class: 
```python
config = {'data': {'model': 'meta-llama/Llama-3.2-3B-Instruct', 'dtype': 'float16'}}
cleaner_instance = cleaner.cleaner(config=config)
```
Example using the `generate_prompts` method to generate text prompts for LLMs to correct Kubernetes configuration files:
```python
prompts = c.generate_prompts(target_dir=args.target)
```
Each prompt is a dictionary containing the file name, system/user prompts, and any relevant feedback.

**Requirements**

* Python 3.8+
* `vllm` library (version 1.0+)
* External tool (e.g., `kube-lint`) for analysis and feedback


**Contributing**

Feel free to contribute to this project by opening issues or pull requests. We welcome any suggestions for improvements, bug fixes, or new features!

**License**
