import torch
from typing import *
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


def load_model(model_ckpt, quantization=None):
    device_map = "auto"
    q_lora = True
   
    model = AutoModel.from_pretrained(
        model_ckpt,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
        #--------8 bit-----------------------------------------------------------
        load_in_8bit= True if quantization =="8bit" else False,
        #--------4 bit-----------------------------------------------------------
        load_in_4bit= True if quantization =="4bit" else False,
        bnb_4bit_use_double_quant=True if quantization =="4bit" else False,
        bnb_4bit_quant_type="nf4" if quantization =="4bit" else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16 if quantization =="4bit" else torch.float32,
        ) if q_lora else None
    )

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        print(f"pass unk_token_id {tokenizer.unk_token_id} to pad_token_id")
        tokenizer.pad_token_id = tokenizer.unk_token_id
    print(f'memory usage of model: {model.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
    return model, tokenizer
