import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Function that sets the seed for reproducibility across various libraries.
    Call this at the top of every training and preprocessing script.
    
    :param seed: The seed value to set for reproducibility
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False