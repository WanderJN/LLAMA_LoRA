import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


base_model_path = "./Llama2_7b_hf"
merged_model_path = "./Llama2_7b_merged"
tokenizer = LlamaTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token_id = 0
# model = LlamaForCausalLM.from_pretrained(base_model_path, load_in_8bit=False, device_map={"": "cuda:0"}, torch_dtype=torch.float16)
model = LlamaForCausalLM.from_pretrained(merged_model_path, load_in_8bit=False, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16)

model.eval()

# 多个测试提示
test_prompts = [
    "Translate Chinese to English:\nInput:我想八点半去火车站\nOutput:",
    "Translate English to Chinese:\nInput:At the end of the 16th century, about five to seven million people spoke English.\nOutput:",
    "Translate Chinese to English:\nInput:今天的天气真好，我想出去玩\nOutput:",
    "Translate English to Chinese:\nInput:The community is a big home for me, I know everyone here, it is a paradise.\nOutput:",
    "Translate English to Chinese:\nInput:These new settlers enriched the English language and especially its vocabulary.\nOutput:"
]

tokenizer.padding_side = "left"
batch_inputs = tokenizer(test_prompts, return_tensors='pt', padding='longest', truncation=True).to("cuda")

with torch.no_grad():
    batch_outputs = model.generate(**batch_inputs, max_new_tokens=512)

# 解码并打印每个提示的响应
for i, output in enumerate(batch_outputs):
    response = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{test_prompts[i]}\n{response}\n=======================================")

