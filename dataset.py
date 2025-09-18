import json

import torch
from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    def __init__(self, data_path,tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, data_path):
        samples = []
        with open(data_path, 'r', encoding = "utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # assume each line is json data format
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # We need to prepare Y as the next token of X, for generative training
        # Y is one token ahead of X
        self.tokenizer(
            # assume each line is json format and includes key of "text"
            str(sample['text']),
            max_length = self.max_length,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        input_ids = encoding.input_ids.squeeze()
        # mask out tokens that is not real content, we will ignore them in loss calculation
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # X won't inlcude the last token, Y won't include the first token
        X = torch.tensor(input_ids[:-1], dtype = torch.long)
        Y = torch.tensor(input_ids[1:], dtype = torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype = torch.long)
        return X, Y, loss_mask
