# coding: utf-8
"""
Defining global constants
"""
# sign segmentation constants
OUT_SIGN_TOKEN, OUT_ID = "<out>", 0
IN_SIGN_TOKEN, IN_ID = "<in>", 1
BEGIN_TOKEN, BEGIN_ID = "<begin>", 2

output_classes = {0: OUT_SIGN_TOKEN, 1: IN_SIGN_TOKEN, 2: BEGIN_TOKEN}
bslcp_classes = {0: "sign", 1: "boundary"}

# special tokens
PAD_TOKEN, PAD_ID = "<pad>", 0
UNK_TOKEN, UNK_ID = "<unk>", 1

special_tokens = {PAD_TOKEN: 0, UNK_TOKEN: 1}

# BOS_TOKEN, BOS_ID = "<s>", 2
# EOS_TOKEN, EOS_ID = "</s>", 3
