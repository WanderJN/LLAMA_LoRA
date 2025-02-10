import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

base_model_path = "./Llama2_7b_hf"
lora_model_path = "./Llama2_7b_finetuned"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoPeftModelForCausalLM.from_pretrained(lora_model_path, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16)

model = model.merge_and_unload()
output_dir = "./Llama2_7b_merged"
model.save_pretrained(output_dir)



