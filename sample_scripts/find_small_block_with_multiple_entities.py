from typing import Dict, Tuple, List
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
import math

from s2and.consts import PREPROCESSED_DATA_DIR
from s2and.data import S2BlocksDataset, Signature
import logging


DATA_HOME_DIR = "/work/pi_mccallum_umass_edu/pragyaprakas_umass_edu/prob-ent-resolution/data"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_blockwise_metadata():
    train_signatures_pkl = f"{PREPROCESSED_DATA_DIR}/pubmed/seed1/train_signatures.pkl"
    blockwise_metadata: Dict[str, List[Signature]]
    with open(train_signatures_pkl,"rb") as _pkl_file:
        blockwise_metadata = pickle.load(_pkl_file)

    print("Total num of blocks:", len(blockwise_metadata.keys()))
    return blockwise_metadata

if __name__=='__main__':
    blockwise_metadata = read_blockwise_metadata()
    for idx, block in enumerate(blockwise_metadata):
        block_id, sign = block
        print(block_id, sign.signature)