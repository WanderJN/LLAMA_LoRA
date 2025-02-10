import os, sys
from tqdm.auto import tqdm
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, get_cosine_schedule_with_warmup
from lora import inject_lora
from textDataset import CausalLMTextDataset, DatasetCollator
from config import *
from torch.cuda.amp import autocast


if __name__ == "__main__":
    base_model_path = r"e:\Codes\llama\Llama2_7b_hf"
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'

    model = LlamaForCausalLM.from_pretrained(base_model_path, device_map={"": "cuda:0"}, torch_dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.config.use_cache=False
    model.gradient_checkpointing_enable()  # 启用梯度检查点以节省显存
    model.enable_input_require_grads()


    # 向原始模型中注入lora层
    for name, layer in model.named_modules():
        name_cols = name.split(".")
        
        filter_names = ["q_proj","v_proj"]
        if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
            inject_lora(model, name, layer, device)
    
    # 冻结非lora的参数
    for name, param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a', 'lora_b']:
            param.requires_grad = False
        else:
            param.requires_grad = True

       # 准备数据集
    train_file_path = r"e:\Codes\llama\translation2019zh/translation2019zh_train.json"
    train_data = []
    count = 0
    max_train_counts = 10000
    with open(train_file_path, "r", encoding="utf-8") as file:
        for line in file:
            count+= 1
            if count > max_train_counts:
                break
            one_data = json.loads(line.strip())
            if random.random() > 0.5:
                train_data.append("Translate English to Chinese:\nInput:" + one_data['english'] + "\nOutput:" + one_data['chinese'] + '</s>')
            else:
                train_data.append("Translate Chinese to English:\nInput:" + one_data['chinese'] + "\nOutput:" + one_data['english'] + '</s>')

    train_dataset = CausalLMTextDataset(tokenizer, train_data)

    # 创建数据处理器
    collator = DatasetCollator(tokenizer=tokenizer)

    # 创建DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collator  # 使用你定义的数据处理器
    )

    # 优化器只更新requires_grad为True的参数
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=10e-4)
    # 创建调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=MAX_TRAIN_STEPS
    )

    model.train()
    count = 0

    for batch in tqdm(train_dataloader, desc=f'Training batch', total=len(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}    # 此时的batch包含input_idx, attention_mask, labels

        optimizer.zero_grad()
        with autocast(dtype=torch.bfloat16):  # 明确指定使用bfloat16
            outputs = model(**batch)
            loss = outputs.loss
        
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], MAX_GRAD_NORM)

        optimizer.step()

        # 更新学习率
        scheduler.step()

        if count % 2 == 0:
            print(f"Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']:.8f}")
        
        '''
        # 每间隔xx次打印lora_a和lora_b的参数与梯度
        if count % 3 == 0:
            s = 1
            # 打印指定参数的梯度
            for name, param in model.named_parameters():
                if 'lora_a' in name or 'lora_b' in name:
                    print(f"Parameter {name}:")
                    print(param.data)  # 打印参数值
                    if param.grad is not None:
                        print(f"Gradient of {name}:")
                        print(param.grad)  # 打印整个梯度张量
                        # 或者打印梯度的一些统计信息，如均值、标准差等
                        print(f"Mean gradient of {name}: {param.grad.mean().item()}")
                        print(f"Std of gradient of {name}: {param.grad.std().item()}")
                    else:
                        print(f"No gradient for {name}")
                    # 只打印一次a和一次b
                    if s >= 2:
                        break
                    s += 1
        '''
        count += 1

        # 如果达到了预定的步数，则提前结束训练
        if count >= MAX_TRAIN_STEPS:
            print(f"Reached {MAX_TRAIN_STEPS} steps. Stopping training.")
            break  # 跳出批次循环

    # 将训练的lora_a和lora_b的参数保存下来，为了后续merge模型
    lora_state = {}
    for name, param in model.named_parameters():
        name_cols = name.split('.')
        filter_names = ['lora_a', 'lora_b']
        if any(n==name_cols[-1] for n in filter_names):
            lora_state[name] = param     # 通过字典进行记录
    
    temp_file = "./lora_state/lora.pt.tmp"
    output_file = "./lora_state/lora.pt"
    # 提取目录部分
    directory = os.path.dirname(temp_file)

    # 检查目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(lora_state, temp_file)
    os.replace(temp_file, output_file)

