import json

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
        #
