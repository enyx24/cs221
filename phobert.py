from transformers import AutoModel, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModel.from_pretrained(
    "vinai/phobert-base-v2",
    trust_remote_code=True,
    use_safetensors=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

inputs = tokenizer("Tôi yêu tiếng Việt", return_tensors="pt").to(device)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
