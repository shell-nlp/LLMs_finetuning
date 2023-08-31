import torch
from typing import *
from transformers import AutoModel,BitsAndBytesConfig
from transformers.utils.logging import get_logger
logger = get_logger("模型")
def load_model(args):
    device_map = "auto"
    logger.info("开始加载模型......")
    if args.q_lora:
        logger.info("正在使用 QLora......")
        logger.info(f"量化为：{args.quantization}")
    model = AutoModel.from_pretrained(
        args.model_ckpt,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
        #--------8 bit-----------------------------------------------------------
        load_in_8bit= True if args.quantization =="8bit" else False,
        #--------4 bit-----------------------------------------------------------
        load_in_4bit= True if args.quantization =="4bit" else False,
        bnb_4bit_use_double_quant=True if args.quantization =="4bit" else False,
        bnb_4bit_quant_type="nf4" if args.quantization =="4bit" else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.quantization =="4bit" else torch.float32,
        ) if args.q_lora else None
    )
    logger.info("模型加载完成......")
    print(f'memory usage of model: {model.get_memory_footprint() / (1024 * 1024 * 1024):.2} GB')
    return model
