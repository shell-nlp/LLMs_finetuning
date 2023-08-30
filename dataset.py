import re
from datasets import load_dataset
from transformers import AutoTokenizer


def get_dataset(data_file, tokenizer):
   
    def generate_and_tokenize_prompt(data_point):
        model_inputs = {}
        instruction = data_point['instruction']
        input_text = data_point["input"]
        target_text = data_point['output']

        prompt = instruction +"\n"+ input_text
        prompt_ids = tokenizer.encode(prompt,add_special_tokens=True)

        target_text = tokenizer.encode(target_text,add_special_tokens=False)
        prompt_len = len(prompt_ids)
        input_ids = prompt_ids + target_text + [tokenizer.eos_token_id]
        labels = [tokenizer.pad_token_id] * prompt_len+target_text + [tokenizer.eos_token_id]
        attention_mask = [1] * len(labels)
        assert len(input_ids) == len(labels) and len(input_ids)==len(attention_mask),"input_ids 和 labels 长度不一致！"
        model_inputs["input_ids"]=input_ids
        model_inputs["labels"]=labels
        model_inputs["attention_mask"]=attention_mask
        return model_inputs

    data = load_dataset("json", data_files=data_file)["train"]
    data = data.map(generate_and_tokenize_prompt, num_proc=8,remove_columns=list(data.features))
    return data


if __name__ == "__main__":
    check_point = r"E:\models\THUDM\chatglm2-6b-int4"
    tokenizer = AutoTokenizer.from_pretrained(check_point, trust_remote_code=True)
    ds = get_dataset("./data/example_data.json", tokenizer)
    print(ds[1])