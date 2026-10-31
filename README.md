# Tópicos en reseñas de Letterboxd en español

Proyecto académico ejecutado con datos públicos reales. El PDF, las tablas y las
asignaciones se generan a partir de la misma ejecución; no se simulan resultados.

## Fuente

Riyosha, *Letterboxd Movie Reviews (~90,000)*, versión 1, actualizado el
5 de septiembre de 2024:
https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000

Los metadatos de Kaggle declaran CC0. Esta declaración del publicador no verifica
la titularidad de cada reseña. El corpus se utiliza para análisis académico local;
se conserva la atribución al dataset y no se publica un espejo en Internet.
El archivo original no proporciona autor ni URL individual de las reseñas.
Los identificadores Rxxxxx corresponden a la posición del registro original.

## Reproducir

Python 3.13 en Windows (CPU). Entorno y versiones reales en
`requirements-lock.txt`. Para una instalación limpia basta con los paquetes
directos de `requirements.txt`; el lock incluye paquetes heredados del entorno
del equipo y sirve como registro, no como instalación portable.

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe src/prepare.py
./.venv/Scripts/python.exe src/model.py
./.venv/Scripts/python.exe src/catalogue.py
./.venv/Scripts/python.exe src/report.py
./.venv/Scripts/python.exe src/verify.py
```

La primera ejecución requiere Internet para descargar el corpus, el modelo y las
stopwords. `prepare.py` descarga el CSV si falta. `model.py` usa la copia local
`models/encoder` si existe; de lo contrario descarga el modelo de Hugging Face
con la revisión registrada en `data/embedding_metadata.json`.

Semillas: 42 en detección de idioma, muestreo y UMAP. UMAP usa un solo hilo.
La reproducibilidad exacta se limita a versiones y plataforma equivalentes.
Las dos configuraciones usan los mismos segmentos y embeddings.

## Archivos

- `output/pdf/reporte_letterboxd.pdf`: reporte final en español.
- `data/segments.csv`: ventanas antes de ajustar el límite del tokenizer.
- `results/model_segments.csv`: segmentos efectivamente modelados.
- `results/segmentation.json`: embudo de selección y estadísticas.
- `results/metrics.json`: comparación numérica de A y B.
- `results/topic_info_A.csv`, `topic_info_B.csv`: salida de `get_topic_info`.
- `results/assignments_A.csv`, `assignments_B.csv`: asignación por segmento.
- `results/representatives_A.json`, `representatives_B.json`: tres ejemplos por tópico.
- `results/topic_info_named_A.csv`, `topic_info_named_B.csv`: tablas con nombres propios.
- `results/catalogue_A.md`, `catalogue_B.md`: los 58 tópicos con nombres y ejemplos completos.
- `results/interpretation.json`: interpretación cualitativa de los resultados de B.
- `results/*html`: visualizaciones interactivas, utilizables sin conexión.
- `src/`: adquisición, segmentación, modelado y construcción del reporte.

El reporte enlaza el código incluido en la entrega. No se creó un repositorio
público; para entregar una URL pública se puede subir este paquete a un repositorio
propio y sustituir el enlace local del PDF.

## Decisiones

Se trabaja con español detectado automáticamente, sin traducir reseñas. Se exige
un mínimo de 20 palabras por reseña. Las ventanas de 80 palabras no se solapan;
una cola de menos de 20 palabras se une a la ventana anterior. Se muestrean como
máximo 3,000 ventanas y después se subdividen las que exceden el límite real del
tokenizer. Se evita confundir la palabra «segmento» con una oración completa.

Los outliers se conservan con etiqueta -1 y no se fuerzan a un tópico. Las
probabilidades HDBSCAN no se interpretan como probabilidades de verdad temática.
No se ejecuta el bonus de API de LLM: no hay gasto ni tokens de API contratados.
La IA asistente ayuda con programación, interpretación y nombres de tópicos;
esto se declara en el PDF y no constituye una comparación experimental con LLM.

La imagen de segmentación del enunciado requiere acceso a Canvas y no se pudo
consultar. Se usa el criterio explícito y reproducible descrito arriba.
