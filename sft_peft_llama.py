import json
import random
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer

max_train_counts = 100000
max_test_counts = 10000

output_dir = "./output"

train_file_path = "./translation2019zh/translation2019zh_train.json"
train_data = []
val_file_path = "./translation2019zh/translation2019zh_valid.json"
val_data = []

count = 0
with open(train_file_path, "r", encoding="utf-8") as file:
    for line in file:
        count += 1
        if count > max_train_counts:
            break
        one_data = json.loads(line.strip())
        if random.random() > 0.5:
            train_data.append({'text': "Translate English to Chinese:\nInput:" + one_data['english'] + "\nOutput:" + one_data['chinese'] + '</s>'})
        else:
            train_data.append({'text': "Translate Chinese to English:\nInput:" + one_data['chinese'] + "\nOutput:" + one_data['english'] + '</s>'})

count = 0
with open(val_file_path, "r", encoding="utf-8") as file:
    for line in file:
        count += 1
        if count > max_test_counts:
            break
        one_data = json.loads(line.strip())
        if random.random() > 0.5:
            val_data.append({'text': "Translate English to Chinese:\nInput:" + one_data['english'] + "\nOutput:" + one_data['chinese'] + '</s>'})
        else:
            val_data.append({'text': "Translate Chinese to English:\nInput:" + one_data['chinese'] + "\nOutput:" + one_data['english'] + '</s>'})

# print(train_data[0:5])
keys = train_data[0].keys()
train_dataset = Dataset.from_dict({key: [dic[key] for dic in train_data] for key in keys})
val_dataset = Dataset.from_dict({key: [dic[key] for dic in val_data] for key in keys})


# 配置lora参数
peft_config = LoraConfig(r=4,
                         lora_alpha=8,
                         target_modules=['q_proj', 'v_proj'],
                         lora_dropout=0.05,
                         task_type=TaskType.CAUSAL_LM)

# 配置训练参数
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    optim='adamw_torch',
    learning_rate=10e-4,
    eval_steps=50,
    save_steps=100,
    logging_steps=20,
    evaluation_strategy='steps',
    group_by_length=False,
    max_steps=200,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    bf16=True,
    lr_scheduler_type='cosine',
    warmup_steps=100
)

model_name = './Llama2_7b_hf'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": "cuda:0"}  # 显式指定GPU设备
)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token_id = 0
tokenizer.padding_side = 'right'


model.enable_input_require_grads()
# 在这里为模型注入lora微调的层
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model.config.use_cache=False

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field='text',
    peft_config=peft_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments
)

trainer.train()
trainer.model.save_pretrained(output_dir)
