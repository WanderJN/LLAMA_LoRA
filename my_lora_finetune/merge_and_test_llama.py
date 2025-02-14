import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer
from lora import LoraLayer, inject_lora
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import jieba, re

# nltk.download('punkt')    # 如果没下载的话需要下载nltk数据
# nltk.download('punkt_tab')


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
    "Translate Chinese to English:\nInput:机器翻译是一个很有挑战的任务。\nOutput:",
    "Translate English to Chinese:\nInput:At the end of the 16th century, about five to seven million people spoke English.\nOutput:",
    "Translate Chinese to English:\nInput:今天的天气真好，我想出去玩。\nOutput:",
    "Translate English to Chinese:\nInput:The community is a big home for me, I know everyone here, it is a paradise.\nOutput:",
    "Translate English to Chinese:\nInput:These new settlers enriched the English language and especially its vocabulary.\nOutput:"
]

# 参考文本
reference_texts = [ 
    "Machine translation is a very challenging task.",
    "在16世纪末，约有五百万到七百万人说英语。", 
    "The weather today is really nice, I want to go out and play.",
    "这个社区对我来说就像家，我认识这里的每个人，这里是一个天堂。",
    "这些新的定居者丰富了英语语言，特别是它的词汇。" 
]

tokenizer.padding_side = "left"
batch_inputs = tokenizer(test_prompts, return_tensors='pt', padding='longest', truncation=True).to(device)

with torch.no_grad():
    batch_outputs = model.generate(**batch_inputs, max_new_tokens=512)

# 解码并打印每个提示的响应
for i, output in enumerate(batch_outputs):
    response = tokenizer.decode(output, skip_special_tokens=True)
    print(f"{response}\n" + '='*100)

# 结果输出
'''
Translate Chinese to English:
Input:机器翻译是一个很有挑战的任务。
Output:Machine translation is a challenging task.
====================================================================================================
Translate English to Chinese:
Input:At the end of the 16th century, about five to seven million people spoke English.
Output:16世纪末，英语的使用人数约为五到七百万。
====================================================================================================
Translate Chinese to English:
Input:今天的天气真好，我想出去玩
Output:Today's weather is good, I want to go out to play
====================================================================================================
Translate English to Chinese:
Input:The community is a big home for me, I know everyone here, it is a paradise.
Output:我在这里的团队是我家，我们是一家人，它是我的天堂。
====================================================================================================
Translate English to Chinese:
Input:These new settlers enriched the English language and especially its vocabulary.
Output:随着新的移民的到来，英语的词汇质量得到了增强。
====================================================================================================
'''

# 判断是否为中文句子
def is_chinese(text):
    # 匹配中文字符
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# 计算BLEU分数
def calculate_bleu(reference, generated):
    # 定义一个平滑函数
    smoothing = SmoothingFunction().method1  # You can try other methods as well

    # 中文用jieba.cut分词，英文用nltk.word_tokenize分词
    if is_chinese(generated):
        reference_tokens = list(jieba.cut(reference))
        generated_tokens = list(jieba.cut(generated))
    else:
        reference_tokens = nltk.word_tokenize(reference.lower())
        generated_tokens = nltk.word_tokenize(generated.lower())

    # Calculate BLEU
    return sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)

# 解码并计算BLEU得分
bleu_socre_list = []
for i in range(len(batch_outputs)):
    response = tokenizer.decode(batch_outputs[i], skip_special_tokens=True).split('Output:')[1]
    reference = reference_texts[i]

    # 计算BLEU得分
    bleu_socre = calculate_bleu(reference, response)
    bleu_socre_list.append(bleu_socre)

    print(f'bleu: {bleu_socre:.4f} |' ,response, '|', reference)
    print('='*100)

average_bleu = sum(bleu_socre_list) / len(bleu_socre_list)
print(f'Average bleu:{average_bleu:.4f}')

'''
bleu: 1.0000 | Machine translation is a very challenging task. | Machine translation is a very challenging task.
====================================================================================================
bleu: 0.1064 | 16世纪中期，约五至七百万人会说英语。 | 在16世纪末，约有五百万到七百万人说英语。
====================================================================================================
bleu: 0.6379 | The weather is nice today, I want to go out and play. | The weather today is really nice, I want to go out and play.
====================================================================================================
bleu: 0.2258 | 我在这里是个大家庭，我知道每个人，这里是天堂。 | 这个社区对我来说就像家，我认识这里的每个人，这里是一个天堂。
====================================================================================================
bleu: 0.0385 | 这些新移民对英语的语法和词汇大大润色了。 | 这些新的定居者丰富了英语语言，特别是它的词汇。
====================================================================================================
Average bleu:0.4017
'''
