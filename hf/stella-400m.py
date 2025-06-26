import os
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize

query_prompt = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
queries = [query_prompt + query for query in queries]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

# The path of your model after cloning it
model_dir = "NovaSearch/stella_en_400M_v5"

vector_dim = 1024
vector_linear_directory = f"2_Dense_{vector_dim}"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).cuda().eval()
# you can also use this model without the features of `use_memory_efficient_attention` and `unpad_inputs`. It can be worked in CPU.
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,use_memory_efficient_attention=False,unpad_inputs=False).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
vector_linear_dict = {
    k.replace("linear.", ""): v for k, v in
    torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin")).items()
}
vector_linear.load_state_dict(vector_linear_dict)
vector_linear.cuda()

# Embed the queries
with torch.no_grad():
    input_data = tokenizer(queries, padding="longest", truncation=True, max_length=512, return_tensors="pt")
    input_data = {k: v.cuda() for k, v in input_data.items()}
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    query_vectors = normalize(vector_linear(query_vectors).cpu().numpy())

# Embed the documents
with torch.no_grad():
    input_data = tokenizer(docs, padding="longest", truncation=True, max_length=512, return_tensors="pt")
    input_data = {k: v.cuda() for k, v in input_data.items()}
    attention_mask = input_data["attention_mask"]
    last_hidden_state = model(**input_data)[0]
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    docs_vectors = normalize(vector_linear(docs_vectors).cpu().numpy())

print(query_vectors.shape, docs_vectors.shape)
# (2, 1024) (2, 1024)

similarities = query_vectors @ docs_vectors.T
print(similarities)
# [[0.8397531  0.29900077]
#  [0.32818374 0.80954516]]

