import torch

# Simulated special tokens
bos_id = [101]
eos_id = [102]
max_length = 20

def _generate_loss_mask(input_ids, bos_id, eos_id, max_length):
    loss_mask = [0]*len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i+len(bos_id)] == bos_id:  # found BOS
            start = i + len(bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end+len(eos_id)] == eos_id:  # found EOS
                    break
                end += 1
            # notice start+1 here
            for j in range(start + 1, min(end+len(eos_id)+1, max_length)):
                loss_mask[j] = 1
            i = end + len(eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask

# Fake sequence:
# X [bos] A B C D [eos] Y
input_ids = [999, 101, 11, 12, 13, 14, 102, 888]

mask = _generate_loss_mask(input_ids, bos_id, eos_id, max_length)

print("input_ids:", input_ids)
print("loss_mask:", mask)
for idx, (tok, m) in enumerate(zip(input_ids, mask)):
    print(f"pos {idx:2d} | token {tok:3d} | mask {m}")

print(type(input_ids[3:4]))