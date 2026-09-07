import os,json,numpy as np,torch,urllib.request
os.environ['HF_HUB_OFFLINE']='1'; os.environ['HF_HOME']='${HF_HOME}'
from transformers import AutoTokenizer, AutoModel
S='.'
tok=AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True)
model=AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v2-moe',trust_remote_code=True,torch_dtype=torch.float32).cuda().eval()
def emb1(w):  # exact server path: search_query prefix, masked mean pooling, L2 normalise, single text
    inp=tok(f"search_query: {w}",return_tensors='pt',truncation=True,max_length=512).to('cuda')
    with torch.no_grad(): o=model(**inp)[0]
    m=inp['attention_mask'].unsqueeze(-1); e=(o*m).sum(1)/m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(e,p=2,dim=1)[0].cpu().numpy()
def server(a,b):
    r=urllib.request.Request('http://127.0.0.1:5200/embed_similarity',data=json.dumps({'text1':a,'text2':b}).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(r,timeout=30))['similarity']
def bin_(c): return 1.0 if c>=.9 else .9 if c>=.8 else .7 if c>=.7 else .5 if c>=.55 else .2 if c>=.4 else 0.0
# 1. verify local replica against the server
for a,b in (('scratch','banana'),('scratch','crack'),('scratch','scratches'),('missing parts','missing component')):
    print(f'check {a}/{b}: local {float(emb1(a)@emb1(b)):.4f} server {server(a,b):.4f}')
# 2. synonym / near-miss / wrong-family / control table
cands={'scratch':['scratches','a scratch','surface scratch','scratched surface','scuff mark','abrasion','crack','pit','chip','contamination','missing parts','banana'],
       'missing parts':['missing part','missing component','absent piece','incomplete assembly','broken','deformation','scratch','contamination','banana'],
       'contamination':['contaminant','foreign objects','dirt','stain','dust particle','scratch','missing parts','banana'],
       'pit':['pits','pitting','small hole','dent','chip','scratch','deformation','banana']}
out={}
for g,cs in cands.items():
    eg=emb1(g); out[g]=[(c,float(eg@emb1(c))) for c in cs]; print(f'\ngold <type> = {g}')
    for c,s in out[g]: print(f'   {c:20s} cosine {s:.2f} -> type score {bin_(s):.1f}')
json.dump(out,open(f'{S}/syn_table_query.json','w'),indent=1)
# 3. corpus: the 36 distinct final strings, cosine matrix among the top 10 + t-SNE of all 7237 with the server path
pairs=json.load(open(f'{S}/types_final_anom.json')); t=[s.strip().lower() for _,s in pairs]
from collections import Counter; cnt=Counter(t); uniq=list(cnt); U={u:emb1(u) for u in uniq}
names=[n for n,_ in cnt.most_common(10)]; M=[[float(U[a]@U[b]) for b in names] for a in names]
print('\ntop-10 cosine matrix (search_query path)'); print(names); print(np.round(np.array(M),2))
off=[M[i][j] for i in range(10) for j in range(10) if i<j]; print('off-diag min %.2f median %.2f max %.2f'%(min(off),float(np.median(off)),max(off)),'bins',Counter(bin_(c) for c in off))
for a,b in (('missing parts','missing part'),('missing parts','missing component'),('foreign objects','foreign object'),('chip','chipping')):
    if a in U and b in U: print(f'synonym {a}/{b}: %.2f -> %.1f'%(float(U[a]@U[b]),bin_(float(U[a]@U[b]))))
json.dump({'names':names,'M':M,'uniq':uniq},open(f'{S}/type_cosine_table_query.json','w'))
E=np.stack([U[s] for s in t]); np.save(f'{S}/types_final_embeddings_query.npy',E)
from sklearn.manifold import TSNE
xy=TSNE(n_components=2,random_state=42,perplexity=30,max_iter=1000).fit_transform(E); np.save(f'{S}/types_final_2d_query.npy',xy); print('tsne done')
