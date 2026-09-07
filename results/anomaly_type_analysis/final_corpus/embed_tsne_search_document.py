import json,os,sys,numpy as np,torch
os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HOME']='${HF_HOME}'
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
S='.'
pairs=json.load(open(f'{S}/types_final_anom.json')); texts=[t for _,t in pairs]
tok=AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True)
model=AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True,torch_dtype=torch.float32).cuda().eval()
embs=[]
for i in range(0,len(texts),64):
    b=[f"search_document: {t}" for t in texts[i:i+64]]
    inp=tok(b,return_tensors='pt',padding=True,truncation=True,max_length=512).to('cuda')
    with torch.no_grad(): embs.append(model(**inp).last_hidden_state.mean(1).cpu().numpy())
E=np.vstack(embs); np.save(f'{S}/types_final_embeddings.npy',E); print('embedded',E.shape,flush=True)
xy=TSNE(n_components=2,random_state=42,perplexity=30,max_iter=1000).fit_transform(E); np.save(f'{S}/types_final_2d.npy',xy); print('tsne done',xy.shape)
