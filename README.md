# LLms_finetuning

致力于用简洁的代码实现对现有大模型（chatGLM1/2、BaiChuan、QWen、Internlm、Llama等）的训练、推理和部署。

仓库正在初步构件中......
## 支持模型
- [x] chatGLM1/2
- [x] Internlm
- [ ] BaiChuan
- [ ] QWen
- [ ] Llama
## 支持训练方式
- [x] 8bit QLora
- [x] 4bit QLora
- [ ] Lora
- [ ] Pre-traing
## 微调方式

1. 修改 train.py文件中的FinetuneArguments 和 TrainingArguments
   
2. 运行train.py文件
   
如下：
```bash
python train.py
```
