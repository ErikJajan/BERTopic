"""Descarga pública y selección reproducible de reseñas españolas."""
import hashlib, html, json, re, zipfile
from pathlib import Path
import pandas as pd
import requests
from langdetect import DetectorFactory, detect_langs

ROOT = Path(__file__).resolve().parents[1]
DetectorFactory.seed = 42
raw = ROOT / 'data/raw'
raw.mkdir(parents=True, exist_ok=True)
csv = raw / 'letterboxd_250movie_reviews.csv'
if not csv.exists():
    if not (raw/'letterboxd.zip').exists():
        r = requests.get('https://www.kaggle.com/api/v1/datasets/download/riyosha/letterboxd-movie-reviews-90000', timeout=180)
        r.raise_for_status()
        (raw/'letterboxd.zip').write_bytes(r.content)
    with zipfile.ZipFile(raw/'letterboxd.zip') as z:
        csv.write_bytes(z.read('letterboxd_250movie_reviews.csv'))
d = pd.read_csv(csv)
stats = {'raw_rows': len(d), 'raw_movies': d.Movie.nunique(), 'sha256_csv': hashlib.sha256(csv.read_bytes()).hexdigest()}
def clean(s):
    s = html.unescape(str(s))
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'https?://\S+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
d['review_id'] = ['R%05d' % i for i in range(len(d))]
d['text'] = d.Review.fillna('').map(clean)
d = d[d.text.str.len() > 0].copy()
stats['nonempty'] = len(d)
d['norm'] = d.text.str.lower()
d = d.drop_duplicates('norm').copy()
stats['unique_reviews'] = len(d)
# Prefiltro conservador para reducir detección costosa. No es un censo del español.
p = re.compile(r'\b(?:el|la|los|las|una|pero|muy|película|pelicula|porque|también|tambien|más|esta|esto|sus|del|que)\b', re.I)
d = d[d.text.map(lambda x: len(p.findall(x)) >= 4)].copy()
stats['language_candidates'] = len(d)
def spanish(s):
    try:
        return next((x.prob for x in detect_langs(s) if x.lang == 'es'), 0.)
    except Exception:
        return 0.
d['es_probability'] = d.text.map(spanish)
d = d[d.es_probability >= .90].copy()
stats['spanish_reviews'] = len(d)
rows = []
for r in d.itertuples():
    # Ventanas de hasta 80 palabras, sin superposición; adjuntar cola corta.
    words = r.text.split()
    if len(words) < 20:
        continue
    chunks = [words[i:i+80] for i in range(0, len(words), 80)]
    if len(chunks) > 1 and len(chunks[-1]) < 20:
        tail = chunks.pop()
        chunks[-1].extend(tail)
    for j, chunk in enumerate(chunks):
        rows.append({'segment_id': f'{r.review_id}_S{j:02d}', 'review_id': r.review_id, 'movie': r.Movie,
                     'date': r.Date, 'text': ' '.join(chunk), 'words': len(chunk), 'es_probability': r.es_probability})
seg = pd.DataFrame(rows)
stats['eligible_segments'] = len(seg)
stats['eligible_reviews'] = seg.review_id.nunique()
# Límite fijo por coste local; muestreo aleatorio sobre segmentos.
if len(seg) > 3000:
    seg = seg.sample(3000, random_state=42)
seg = seg.sort_values('segment_id').reset_index(drop=True)
stats.update({'modeled_segments': len(seg), 'modeled_reviews': seg.review_id.nunique(), 'modeled_movies': seg.movie.nunique(),
              'words': seg.words.describe().to_dict(), 'top_movies': seg.movie.value_counts().head(10).to_dict()})
(ROOT/'data').mkdir(exist_ok=True)
d.drop(columns=['Review','norm']).to_csv(ROOT/'data/spanish_reviews.csv', index=False)
seg.to_csv(ROOT/'data/segments.csv', index=False)
(ROOT/'results/segmentation.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(stats, indent=2, ensure_ascii=False))
