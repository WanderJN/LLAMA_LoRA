import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer

class CausalLMTextDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts
        # self.encodings = tokenizer(texts, truncation=True, padding='do_not_pad', max_length=max_length, return_tensors='pt')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # encodings是包含input_ids和attention_mask的字典，格式是[batch, seq_len]
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='do_not_pad', max_length=self.max_length, return_tensors='pt')
        item = encoding
        # Shift labels by one position to the left and fill the last position with -100 (ignore_index)
        item['labels'] = item['input_ids'].clone()
        item['labels'][item['labels'] == self.tokenizer.pad_token_id] = -100
        # 因为在llama模型内部计算loss时已经处理了移位问题，因此这里不需要再处理
        # item['labels'] = torch.roll(item['labels'], shifts=-1, dims=0)  # 将labels整体向左移动一位
        # item['labels'][-1] = -100   # 最后一位补0
        return item

class DatasetCollator:
    def __init__(self, tokenizer, IGNORE_INDEX=-100):
        self.IGNORE_INDEX = IGNORE_INDEX
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids_list = []
        label_list = []
        input_len_list = []

        for feature in features:
            input_ids, label = feature['input_ids'], feature['labels']
            input_ids_list.append(input_ids)
            label_list.append(label)
            input_len_list.append(input_ids.shape[1])   # [1, seq_len]
        
        max_input_len = max(input_len_list)

        final_input_ids = torch.concat([
            torch.concat([torch.full((1, max_input_len - input_len_list[index]), self.tokenizer.pad_token_id), value], axis=1)
            for index, value in enumerate(input_ids_list)
        ])
        final_label = torch.concat([
            torch.concat([torch.full((1, max_input_len - input_len_list[index]), self.tokenizer.pad_token_id), value], axis=1)
            for index, value in enumerate(label_list)
        ])
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.tokenizer.pad_token_id] = 0

        return {
            "input_ids": final_input_ids,
            "attention_mask": attention_mask,
            "labels": final_label
        }



if __name__ == "__main__":
    # 示例文本列表
    texts = [
        "Translate English to Chinese:\nInput:" + 'english' + "\nOutput:" + 'chinese' + '</s>',
        "今天天气很好！",
        "添加更多文本..."
    ]

    # 初始化分词器
    tokenizer = LlamaTokenizer.from_pretrained("E:\Codes\llama\Llama2_7b_hf", trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    # 创建数据集实例
    dataset = CausalLMTextDataset(tokenizer, texts)

    # 创建数据处理器
    collator = DatasetCollator(tokenizer=tokenizer)

    # 创建DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=collator  # 使用你定义的数据处理器
    )

    for batch in dataloader:
        print(batch)