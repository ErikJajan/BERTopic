"""Dos ajustes BERTopic, métricas descriptivas y figuras verificables."""
import os
os.environ['HF_HOME'] = str(__import__('pathlib').Path(__file__).resolve().parents[1]/'models/cache')
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
import requests, zipfile, io

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'results'
df = pd.read_csv(ROOT/'data/segments.csv')
model_id = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
revision=json.loads((ROOT/'data/embedding_metadata.json').read_text())['sha']
encoder = SentenceTransformer(str(ROOT/'models/encoder'), device='cpu') if (ROOT/'models/encoder/model.safetensors').exists() else SentenceTransformer(model_id, revision=revision, device='cpu')
# Subdividir ventanas a nivel de palabras hasta caber en el límite real del tokenizer.
def split_fit(text):
    if len(encoder.tokenizer.encode(text)) <= encoder.max_seq_length:
        return [text]
    words=text.split(); mid=len(words)//2
    return split_fit(' '.join(words[:mid]))+split_fit(' '.join(words[mid:]))
rows=[]
for r in df.to_dict('records'):
    for k, text in enumerate(split_fit(r['text'])):
        rows.append(dict(r, segment_id=r['segment_id']+f'_P{k}',text=text,words=len(text.split())))
df=pd.DataFrame(rows)
df.to_csv(out/'model_segments.csv',index=False)
docs=df.text.tolist()
cache=out/'embeddings.npy'
fingerprint=hashlib.sha256(json.dumps({'docs':docs,'model':model_id,'revision':revision},ensure_ascii=False).encode('utf-8')).hexdigest()
fingerprint_file=out/'embeddings.sha256'
if cache.exists() and fingerprint_file.exists() and fingerprint_file.read_text().strip()==fingerprint:
    emb=np.load(cache)
    assert len(emb)==len(docs)
else:
    emb=encoder.encode(docs,batch_size=32,show_progress_bar=True,normalize_embeddings=True)
    np.save(cache,emb)
    fingerprint_file.write_text(fingerprint,encoding='ascii')
stopfile=ROOT/'data/stopwords_es.txt'
if not stopfile.exists():
    r=requests.get('https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip',timeout=60)
    r.raise_for_status()
    stopfile.write_bytes(zipfile.ZipFile(io.BytesIO(r.content)).read('stopwords/spanish'))
stop=stopfile.read_text(encoding='utf-8').splitlines()
metrics=[]; assignments={}; models={}
for name, size, samples, grams in [('A',15,5,(1,1)),('B',35,10,(1,2))]:
    start=time.time()
    m=BERTopic(embedding_model=encoder,
        umap_model=UMAP(n_neighbors=15,n_components=5,min_dist=0,metric='cosine',random_state=42,n_jobs=1),
        hdbscan_model=HDBSCAN(min_cluster_size=size,min_samples=samples,metric='euclidean',cluster_selection_method='eom',prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=stop,ngram_range=grams,min_df=2),
        nr_topics=None,calculate_probabilities=False,verbose=True)
    topics,probs=m.fit_transform(docs,emb)
    t=np.array(topics); assignments[name]=t; models[name]=m
    m.get_topic_info().to_json(out/f'topic_info_{name}.json',orient='records',force_ascii=False,indent=2)
    m.get_topic_info().to_csv(out/f'topic_info_{name}.csv',index=False)
    df.assign(topic=t,probability=probs).to_csv(out/f'assignments_{name}.csv',index=False)
    reps={}
    for topic in sorted(map(int,set(t)-{-1})):
        # Los tres documentos elegidos por BERTopic usando c-TF-IDF.
        reps[str(topic)]=[]
        for text in m.get_representative_docs(topic)[:3]:
            row=df[(df.text==text)&(t==topic)].iloc[0]
            reps[str(topic)].append({'id':row.segment_id,'movie':row.movie,'text':text})
    (out/f'representatives_{name}.json').write_text(json.dumps(reps,ensure_ascii=False,indent=2),encoding='utf-8')
    mask=t!=-1; k=len(set(t[mask]))
    words=[w for topic in range(k) for w,_ in m.get_topic(topic)[:10]]
    sil=float(silhouette_score(emb[mask],t[mask],metric='cosine',sample_size=min(2000,int(mask.sum())),random_state=42)) if k>1 else None
    purity=sum(df.loc[t==topic,'movie'].value_counts().iloc[0] for topic in range(k))/mask.sum()
    metrics.append({'configuration':name,'min_cluster_size':size,'min_samples':samples,'ngram_range':grams,
                    'topics':k,'outliers':int((~mask).sum()),'outlier_pct':float(100*(~mask).mean()),
                    'diversity_top10':len(set(words))/len(words),'silhouette_cosine':sil,'movie_purity':float(purity),'seconds':time.time()-start})
    m.visualize_barchart(top_n_topics=min(12,k),n_words=8).write_html(out/f'barchart_{name}.html',include_plotlyjs=True)
    m.visualize_heatmap().write_html(out/f'heatmap_{name}.html',include_plotlyjs=True)
    m.save(str(out/f'model_{name}'),serialization='safetensors',save_ctfidf=True,save_embedding_model=model_id)
    print(metrics[-1],flush=True)
common=(assignments['A']!=-1)&(assignments['B']!=-1)
summary={'metrics':metrics,'segments':len(df),'reviews':df.review_id.nunique(),'movies':df.movie.nunique(),
         'encoder':model_id,'max_seq_length':encoder.max_seq_length,'max_observed_tokens':max(len(encoder.tokenizer.encode(x)) for x in docs),
         'ari_all':float(adjusted_rand_score(assignments['A'],assignments['B'])),
         'ari_common_assigned':float(adjusted_rand_score(assignments['A'][common],assignments['B'][common])),
         'common_assigned':int(common.sum())}
(out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.crosstab(assignments['A'],assignments['B']).to_csv(out/'configuration_crosstab.csv')
print(json.dumps(summary,indent=2),flush=True)
