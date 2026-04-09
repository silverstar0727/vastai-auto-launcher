import torch


def get_num_valid_tokens(item_indexes):
    """
    # Args
        item_indexes : (N, T)
    # Returns
        num_valid_tokens : (N, 1)
            padding을 제외한 token 숫자
    """
    num_tokens = item_indexes.size()[-1]
    num_pad_tokens = torch.sum(item_indexes == 0, dim=1)
    num_valid_tokens = (num_tokens - num_pad_tokens).unsqueeze(dim=1)
    return num_valid_tokens
