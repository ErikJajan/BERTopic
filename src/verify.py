"""Verifica que las tablas, conteos y ejemplos entregados se correspondan."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]; out=root/'results'
m=json.loads((out/'metrics.json').read_text())
segments=pd.read_csv(out/'model_segments.csv')
emb=np.load(out/'embeddings.npy')
assert emb.shape==(len(segments),384)
assert np.isfinite(emb).all()
assert segments.segment_id.is_unique
assert len(segments)==m['segments']
assert m['max_observed_tokens']<=m['max_seq_length']
for metric in m['metrics']:
    cfg=metric['configuration']
    a=pd.read_csv(out/f'assignments_{cfg}.csv')
    t=pd.read_csv(out/f'topic_info_{cfg}.csv')
    r=json.loads((out/f'representatives_{cfg}.json').read_text(encoding='utf-8'))
    assert a.segment_id.tolist()==segments.segment_id.tolist()
    assert a.text.tolist()==segments.text.tolist()
    assert t.Count.sum()==len(a)
    assert int((a.topic==-1).sum())==metric['outliers']
    assert abs(100*(a.topic==-1).mean()-metric['outlier_pct'])<1e-8
    for topic, examples in r.items():
        assert len(examples)==3
        for e in examples:
            match=a[a.segment_id==e['id']].iloc[0]
            assert match.topic==int(topic)
            assert match.text==e['text']
    assert set(t.Topic)-{-1}==set(map(int,r))
print('PASS: embeddings, segmentos, conteos, outliers y tres ejemplos por tópico en A/B.')
