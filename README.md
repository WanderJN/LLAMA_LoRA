# 基于peft库实现LoRA微调LLAMA模型
1.获取训练数据集，`translation2019zh`，来源百度飞桨https://aistudio.baidu.com/datasetdetail/209041 ；<br>
2.基于peft库实现注入lora层，并利用stf库进行模型训练，详见`sft_peft_llama.py`；<br>
3.合并lora层，获得最终微调后的模型，详见`merge_llama.py`；<br>
4.利用测试数据进行测试，对照微调前后输出结果的差异，详见`contrast_llama.py`。<br>

# 自己实现LoRA微调LLAMA模型（my_lora_finetune文件夹）
1.准备数据集和数据集加载器，详见`textDataset.py`；<br>
2.准备lora层实现的类，以及为模型某层更改并注入lora层的函数，详见`lora.py`；<br>
3.为模型注入lora，加载数据集，训练模型，保存lora层的参数，详见`lora_finetune.py`；<br>
4.读取原模型，注入训练好的lora参数，合并模型，并对模型进行测试，详见`merge_and_test_llama.py`；<br>

## 训练过程
![image](https://github.com/user-attachments/assets/941a5ad8-3384-495a-8ec6-951eadc03304)
## 结果展示
![image](https://github.com/user-attachments/assets/ab411cc4-c6f5-472a-b765-7618d190db3a)
