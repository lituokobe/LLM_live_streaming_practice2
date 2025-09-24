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
        encoding = self.tokenizer(
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

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens = False).input_ids
        # we manually check the data to confirm the bos
        # we make "assistant" as part of the bos token, so that the model can better learn the response generation
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens = False).input_ids

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding = "utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # assume each line is json data format
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)
    
    def _create_chat_prompt(self, conversations):
        # Create conversation based on the chat format
        # In the data, each conversation is a list of 2 dicts, each dict has two keys: "role" and "content"
        # first dict has role of "user", second dict has role of "assistant"
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i%2==0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        
        # apply_chat_template will add the special tokens.
        # it will return a long string including Q and A
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = False
        )
    
    def _generate_loss_mask(self, input_ids):
        # the content of assistant is the only part that should be used to compute loss
        loss_mask = [0]*len(input_ids) # initialize the mask to all zeros
        i = 0
        # iterate through input_ids to find the positions of bos and eos tokens
        while i < len(input_ids):
            # find which one is bos token
            if input_ids[i:i+len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id) # locate the start position
                end = start
                # locate the end position
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # mark the positions between start and end as 1 in loss_mask
                for j in range(start + 1, min(end+len(self.eos_id)+1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


    def __getitem__(self, index):
        sample = self.samples[index]
        # craete prompt based on the conversation style data
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id]*(self.max_length - len(input_ids)) # padding or cutting to max_length

        # create loss mask, indicating which tokens should be used to compute loss and which should be ignored
        loss_mask = self._generate_loss_mask(input_ids)

        # prepare X and Y, Y is one token ahead of X
        X = torch.tensor(input_ids[:-1], dtype = torch.long)
        Y = torch.tensor(input_ids[1:], dtype = torch.long)
        # for pretraining, every token is considered as next token which we care
        # but in this Q&A SFT task, only the tokens in the response part are important
        loss_mask = torch.tensor(loss_mask[1:], dtype = torch.long)

        return X, Y, loss_mask

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer("<|im_start|>assistant>", add_special_tokens = False).input_ids
        self.eos_id = tokenizer("<im_end>", add_special_tokens = False).input_ids
        with open(file_path, 'r', encoding = "utf-8") as f:
            self.data = []
            for line in f:
                line = line.strip()
                obj = json.loads(line)
                self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index] # one line of data, including chose and rejected
        chosen = item["chosen"] #yw
        rejected = item["rejected"] #yl
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize = False, add_generation_prompt = False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize = False, add_generation_prompt = False
        )
        # Convert the prompt to input_ids
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation = True, max_length = self.max_length, padding = 'max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation = True, max_length = self.max_length, padding = 'max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)

        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype = torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype = torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype = torch.long)

        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype = torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype = torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype = torch.long)

        return {
            'x_chosen':x_chosen,
            'y_chosen':y_chosen,
            'mask_chosen':mask_chosen,
            'x_rejected':x_rejected,
            'y_rejected':y_rejected,
            'mask_rejected':mask_rejected
        }

    def _generate_loss_mask(self, input_ids):
        # input_ids: question+answer(chosen) or question+answer(rejected)
        loss_mask = [0]*len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i+len(self.bos_id)] == self.bos_id: # matching <|im_start|>assistant
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end+len(self.eos_id)] == self.eos_id: #matching <|im_end|>
                        break
                    end += 1
                for j in range(start + 1, min(end+len(self.eos_id)+1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


