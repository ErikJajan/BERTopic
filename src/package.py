"""Empaqueta la entrega sin el entorno virtual ni los pesos descargables del encoder."""
from pathlib import Path
import hashlib, json, zipfile
ROOT=Path(__file__).resolve().parents[1]
paths=[ROOT/'README.md',ROOT/'requirements.txt',ROOT/'requirements-lock.txt']
for folder in ['src','results','data','output/pdf']:
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts:
            continue
        if folder=='data' and 'raw' in p.parts and p.name not in ['letterboxd.zip','kaggle_metadata.json']:
            continue
        paths.append(p)
paths=sorted(set(paths))
manifest={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}
manifest_path=ROOT/'MANIFEST.json'
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
target=ROOT/'output/entrega_letterboxd_bertopic.zip'
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths+[manifest_path]:
        z.write(p,'letterboxd_bertopic/'+str(p.relative_to(ROOT)).replace('\\','/'))
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    assert any(x.endswith('reporte_letterboxd.pdf') for x in z.namelist())
print(f'Entrega: {target} | {target.stat().st_size/1024**2:.1f} MiB | {len(paths)+1} archivos')
