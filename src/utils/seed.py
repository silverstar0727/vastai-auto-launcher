import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


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
