import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from lora import LoraLayer, inject_lora

base_model_path = "e:\Codes\llama\Llama2_7b_hf"
tokenizer = LlamaTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
tokenizer.pad_token_id = 0
model = LlamaForCausalLM.from_pretrained(base_model_path, load_in_8bit=False, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 先向基础模型里注入lora参数
for name, layer in model.named_modules():
    name_cols = name.split('.')
    filter_names = ["q_proj", "v_proj"]
    if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
        inject_lora(model, name, layer, device)


# 加载lora的参数信息到model中
lora_state = torch.load("e:\Codes\llama\my_lora_finetune\lora_state/lora.pt")
model.load_state_dict(lora_state, strict=False)


# 将lora权重合并到主模型上
for name, layer in model.named_modules():
    name_cols = name.split('.')

    # 当前module为LoraLayer时，需要进行替换回来
    if isinstance(layer, LoraLayer):
        children = name_cols[:-1]
        cur_layer = model
        # 找到layer的父module，为了后续通过setattr替换当前module
        for child in children:
            cur_layer = getattr(cur_layer, child)
        
        # 通过lora_a和lora_b计算lora的权重，并以相加的形式更新raw_linear
        lora_weight = (layer.lora_a @ layer.lora_b) * layer.alpha / layer.r
        layer.raw_linear.weight = nn.Parameter(layer.raw_linear.weight.add(lora_weight.T)).to(device)   # 注意这里的lora_weight要转置
        
        # 最后通过setattr更新主模型的权重
        setattr(cur_layer, name_cols[-1], layer.raw_linear)

# 对合并之后的模型，还可以按照model.save()进行保存

# 开始模型评估
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
batch_inputs = tokenizer(test_prompts, return_tensors='pt', padding='longest', truncation=True).to(device)

with torch.no_grad():
    batch_outputs = model.generate(**batch_inputs, max_new_tokens=512)

# 解码并打印每个提示的响应
for i, output in enumerate(batch_outputs):
    response = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{test_prompts[i]}\n{response}\n=======================================")

# 结果输出
'''
Translate Chinese to English:
Input:我想八点半去火车站
Output:
Translate Chinese to English:
Input:我想八点半去火车站
Output:I want to leave the train station at eight-thirty.
=======================================
Translate English to Chinese:
Input:At the end of the 16th century, about five to seven million people spoke English.
Output:
Translate English to Chinese:
Input:At the end of the 16th century, about five to seven million people spoke English.
Output:在16世纪末，约五到七百万人能说英语。
=======================================
Translate Chinese to English:
Input:今天的天气真好，我想出去玩
Output:
Translate Chinese to English:
Input:今天的天气真好，我想出去玩
Output:The weather is good today, I want to go out and play.
=======================================
Translate English to Chinese:
Input:The community is a big home for me, I know everyone here, it is a paradise.
Output:
Translate English to Chinese:
Input:The community is a big home for me, I know everyone here, it is a paradise.
Output:我在这里认识了很多人，我感到这里是我家，是我的天堂。
=======================================
Translate English to Chinese:
Input:These new settlers enriched the English language and especially its vocabulary.
Output:
Translate English to Chinese:
Input:These new settlers enriched the English language and especially its vocabulary.
Output:这些新的移民增加了英语的词汇，特别是其词汇。
=======================================
'''