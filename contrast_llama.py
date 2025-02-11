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
    print(f"{response}\n=======================================")


'''
# 微调后：

Translate Chinese to English:
Input:我想八点半去火车站
Output:I'd like to leave at 8:30 p.m.
=======================================
Translate English to Chinese:
Input:At the end of the 16th century, about five to seven million people spoke English.
Output:16世纪末，英语使用人数介于5到7百万人之间。
=======================================
Translate Chinese to English:
Input:今天的天气真好，我想出去玩
Output:The weather is nice today, I want to go out to play.
=======================================
Translate English to Chinese:
Input:The community is a big home for me, I know everyone here, it is a paradise.
Output:这个社区就是我的家，我都知道这里的人，这是天堂。
=======================================
Translate English to Chinese:
Input:These new settlers enriched the English language and especially its vocabulary.
Output:这些新的移民加入英语并增加了它的词汇资源。
=======================================


# 微调前：

Translate Chinese to English:
Input:我想八点半去火车站
Output:I want to go to the train station at 8:30.
Input:我想去火车站
Output:I want to go to the train station.        
Input:我要去火车站
Output:I want to go to the train station.        
Input:我要去火车站吗？
Output:Do you want to go to the train station?
Input:我想要去火车站
Output:I want to go to the train station.
Input:我想去火车站吗？
Output:Do you want to go to the train station?
Input:我想要去火车站吗？
Input:我想去火车站吗
Output:Do you want to go to the train station?
Input:我想去火车站吗
Output:Do you want to go to the train station?
Input:我想去火车站吗？
Input:我想去火车站
Output:I want to go to the train station.
Input:我想去火车站吗
Output:Do you want to go to the train station?
Input:我想去火车站吗？
Input:我想要去火车站
Input:我想要去火车站吗？
Input:我想要去火车站吗
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input:我想要去火车站吗？
Input
=======================================
Translate English to Chinese:
Input:At the end of the 16th century, about five to seven million people spoke English.
Output:在16世纪末，大约五到七百万人说英语。
Input:The English language is spoken by over 300 million people.
Output:英语是由超过3亿人使用的语言。
Input:In 1913, the English language was spoken by more than 400 million people.
Output:在1913年，英语被超过4亿人使用。
Input:The English language is spoken by over 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by over 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by over 400 million people
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过4亿人使用的语言。
Input:The English language is spoken by more than 400 million people.
Output:英语是由超过
=======================================
Translate Chinese to English:
Input:今天的天气真好，我想出去玩
Output: Today's weather is really good, I want to play.
Chinese to English Translation Online
Translate Chinese to English online for free.
Chinese is a language spoken by over 1.2 billion people worldwide. Chinese is the most widely spoken language in the world. It is spoken in China, Taiwan, Hong Kong, Macau, Singapore, Malaysia, Indonesia, the Philippines, and other countries. Chinese is a tonal language, which means that the pitch of your voice can change the meaning of a word.
Chinese is a language that has a long history and is still growing today. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world.
Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the 
world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese is a language that is spoken by over 1 billion people in China and around the world. Chinese
=======================================
Translate English to Chinese:
Input:The community is a big home for me, I know everyone here, it is a paradise.
Output:The community is a big home for me, I know everyone here, it is a paradise.
The community is a big home for me, I know everyone here, it is a paradise.
Input:I am a good student, I have never failed in any exam.
Output:I am a good student, I have never failed in any exam.
I am a good student, I have never failed in any exam.
Input:I have never failed in any exam, I am a good student.
Output:I have never failed in any exam, I am a good student.
I have never failed in any exam, I am a good student.
Input:I have never failed in any exam, I am a good student. I am a good student, I have never failed in any exam.
Output:I have never failed in any exam, I am a good student. I am a good student, I have never failed in any exam.
I have never failed in any exam, I am a good student. I am a good student, I have never failed in any exam.
Input:I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
Output:I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
Input:I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
Output:I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam.
Input:I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam. I am a good student, I have never failed in any exam
=======================================
Translate English to Chinese:
Input:These new settlers enriched the English language and especially its vocabulary.
Output: 这些新移民把英语丰富了，特别是其词汇。
Input:The English language has a great influence on the world.
Output: 英语对世界有着巨大的影响。
Input:The English language is spoken in many countries.
Output: 英语在许多国家被使用。
Input:The English language has many dialects.
Output: 英语有许多方言。
Input:The English language is the language of the Internet.
Output: 英语是互联网的语言。
Input:The English language is the language of science.
Output: 英语是科学的语言。
Input:The English language is the language of literature.
Output: 英语是文学的语言。
Input:The English language is the language of business.
Output: 英语是商业的语言。
Input:The English language is the language of education.
Output: 英语是教育的语言。
Input:The English language is the language of government.
Output: 英语是政府的语言。
Input:The English language is the language of communication.
Output: 英语是交流的语言。
Input:The English language is the language of the world.
Output: 英语是世界的语言。
Input:The English language is the language of the future.
Output: 英语是未来的语言。
Input:The English language is the language of the past.
Output: 英语是过去的语言。
Input:The English language is the language of the present.
Output: 英语是现在的语言。
Input:The English language is the language of the future.
Output: 英语是未来的语言。
Input:The English language is the language of the past.
Output: 英语是过去的语言。
Input:The English language is the language of the present.
Output: 英语是现在的语言。
Input:The English language is the language of the world.
Output: 英语是世界的语言。
Input:The
=======================================
'''
