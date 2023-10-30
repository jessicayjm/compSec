from transformers import AutoTokenizer, AutoModel


llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(llama2_path)
model = AutoModel.from_pretrained(llama2_path, device_map='balanced')
print(model)