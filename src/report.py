"""Reporte PDF a partir de resultados reales. Ejecutar después de model.py."""
from pathlib import Path
import json, re, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
S=json.loads((R/'segmentation.json').read_text(encoding='utf-8'))
M=json.loads((R/'metrics.json').read_text(encoding='utf-8'))
N=json.loads((R/'interpretation.json').read_text(encoding='utf-8'))
cfg=N['selected']; D=pd.read_csv(R/f'assignments_{cfg}.csv')
T=json.loads((R/f'topic_info_{cfg}.json').read_text(encoding='utf-8'))
E=json.loads((R/f'representatives_{cfg}.json').read_text(encoding='utf-8'))
labels={int(k):v['name'] for k,v in N['topics'].items()}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
topic_rows=[r for r in T if r['Topic']!=-1]
fig,ax=plt.subplots(figsize=(10,5.7))
ids=[r['Topic'] for r in topic_rows]
ax.barh([f'{i}. {labels[i]}' for i in ids][::-1],[r['Count'] for r in topic_rows][::-1],color='#167D9A')
ax.set_xlabel('Número de segmentos'); ax.set_title('Distribución de tópicos (excluye -1)',loc='left',weight='bold')
fig.tight_layout(); fig.savefig(R/'topic_sizes.png',dpi=180); plt.close(fig)
emb=np.load(R/'embeddings.npy'); centroids=np.array([emb[D.topic.to_numpy()==i].mean(axis=0) for i in ids])
sim=cosine_similarity(centroids)
fig,ax=plt.subplots(figsize=(8,6.2)); im=ax.imshow(sim,vmin=0,vmax=1,cmap='YlGnBu')
ax.set_xticks(range(len(ids)),ids); ax.set_yticks(range(len(ids)),ids)
ax.set_xlabel('ID de tópico'); ax.set_ylabel('ID de tópico'); ax.set_title('Similitud coseno entre centroides de embeddings',loc='left',weight='bold')
if len(ids)<=15:
    for i in range(len(ids)):
        for j in range(len(ids)):
            ax.text(j,i,f'{sim[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if sim[i,j]>.65 else '#162B3A')
fig.colorbar(im,ax=ax,shrink=.8); fig.tight_layout(); fig.savefig(R/'heatmap_centroids.png',dpi=180); plt.close(fig)
# Barras c-TF-IDF: no son frecuencias de palabras.
terms={int(r['Topic']):r['Representation'][:6] for r in topic_rows}
saved_topics=json.loads((R/f'model_{cfg}'/'topics.json').read_text(encoding='utf-8'))['topic_representations']
cols=2; n=len(ids); rows=(n+1)//2
fig,axes=plt.subplots(rows,cols,figsize=(11,2.5*rows),squeeze=False)
for ax,i in zip(axes.flat,ids):
    pairs=saved_topics[str(i)][:6]
    ax.barh([x[0] for x in pairs][::-1],[x[1] for x in pairs][::-1],color='#167D9A')
    ax.set_title(f'{i}. {labels[i]}',fontsize=10); ax.tick_params(axis='y',labelsize=9)
for ax in list(axes.flat)[len(ids):]: ax.set_visible(False)
fig.tight_layout(); fig.savefig(R/'ctfidf.png',dpi=160); plt.close(fig)

font=Path('C:/Windows/Fonts/arial.ttf'); bold=Path('C:/Windows/Fonts/arialbd.ttf')
if not font.exists():
    from matplotlib.font_manager import findfont, FontProperties
    font=Path(findfont('DejaVu Sans')); bold=Path(findfont(FontProperties(family='DejaVu Sans',weight='bold')))
pdfmetrics.registerFont(TTFont('Body',str(font))); pdfmetrics.registerFont(TTFont('BodyBold',str(bold)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='Body',boldItalic='BodyBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Main',fontName='Body',fontSize=10.5,leading=15,spaceAfter=9,textColor=colors.HexColor('#233746')))
styles.add(ParagraphStyle(name='TitleX',fontName='BodyBold',fontSize=29,leading=34,spaceAfter=18,textColor=colors.HexColor('#123849')))
styles.add(ParagraphStyle(name='H',fontName='BodyBold',fontSize=17,leading=21,spaceBefore=12,spaceAfter=12,textColor=colors.HexColor('#126A82')))
styles.add(ParagraphStyle(name='Sub',fontName='BodyBold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor('#123849')))
styles.add(ParagraphStyle(name='SmallX',fontName='Body',fontSize=8,leading=11,spaceAfter=6,textColor=colors.HexColor('#334A57')))
styles.add(ParagraphStyle(name='QuoteX',fontName='Body',fontSize=9.3,leading=13,leftIndent=12,rightIndent=6,spaceAfter=12,borderPadding=7,backColor=colors.HexColor('#F0F5F7')))
story=[]
def p(s,style='Main'): return Paragraph(s,styles[style])
def add(s,style='Main'): story.append(p(s,style))
def heading(s): add(s,'H')
def page(): story.append(PageBreak())
def table(data,widths):
    rows=[[p(escape(str(x)),'SmallX') for x in row] for row in data]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDEDF2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#167D9A')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F7F8')]),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(t)
def img(name,w=480):
    from PIL import Image as PILImage
    with PILImage.open(R/name) as im: h=w*im.height/im.width
    story.append(Image(str(R/name),width=w,height=h))

add('ANÁLISIS DE TEXTO · BERTopic','Sub'); story.append(Spacer(1,26))
add('¿De qué hablan las reseñas de Letterboxd?','TitleX')
add('Modelado de tópicos en reseñas en español','H')
add('Reporte reproducible | 30 de octubre de 2026')
add(f"<b>{M['segments']:,} segmentos · {M['reviews']:,} reseñas · {M['movies']} películas</b>")
story.append(Spacer(1,18))
add(N['abstract'])
add('Pregunta de investigación','Sub')
add('¿Qué temas emergen en las reseñas en español de una muestra de películas muy valoradas, y cómo cambian al exigir grupos de mayor tamaño?')
add('Entrega y trazabilidad','Sub')
add('Se incluyen corpus, identificadores de origen, asignaciones, tablas de get_topic_info, ejemplos completos, visualizaciones interactivas y código. Todos los conteos proceden de la ejecución guardada; los nombres e interpretaciones son una lectura analítica asistida por IA.')
add('<link href="../../src/model.py" color="#126A82">Código de modelado (archivo incluido)</link> · <link href="../../README.md" color="#126A82">Guía y reproducción</link>')
page()
heading('1. Corpus y segmentación')
add('Se utilizó el dataset de Riyosha, <i>Letterboxd Movie Reviews (~90,000)</i>, versión 1, actualizado el 5 de septiembre de 2024 [1]. Contiene reseñas y valoraciones de 250 películas del listado superior de Letterboxd en la captura. No se extrajeron datos de cuentas ni se utilizó una API de pago.')
add('Los metadatos del publicador declaran CC0; esto no verifica los derechos de cada autor. El archivo no incluye nombres de usuario ni enlaces individuales. Se conservan Movie, fecha y posición original como trazabilidad; los IDs Rxxxxx permiten volver al CSV. La fecha de descarga fue el 30 de octubre de 2026.')
table([['Etapa','Registros'],['CSV original',S['raw_rows']],['Texto no vacío',S['nonempty']],['Reseñas únicas (texto normalizado)',S['unique_reviews']],['Candidatas por vocabulario español',S['language_candidates']],['Español con puntuación ≥ 0.90',S['spanish_reviews']],['Reseñas con ≥ 20 palabras',S['eligible_reviews']],['Ventanas elegibles',S['eligible_segments']],['Ventanas muestreadas (semilla 42)',S['modeled_segments']],['Segmentos finales por límite del tokenizer',M['segments']]], [370,110])
add('La normalización decodifica HTML y elimina etiquetas, URLs y espacios repetidos; conserva acentos y puntuación para embeddings. Los duplicados se detectan ignorando mayúsculas. El prefiltro exige cuatro apariciones de un vocabulario español explícito; después langdetect selecciona es con puntuación ≥ 0.90 y semilla 42. Esa puntuación no equivale a una probabilidad calibrada de idioma correcto.', 'SmallX')
page()
heading('2. Criterio de segmentación y alcance')
add('La reseña es la unidad de origen. Se excluyen reseñas de menos de 20 palabras, con poco contexto. Las demás se dividen en ventanas consecutivas de hasta 80 palabras, sin solapamiento. Una cola de menos de 20 palabras se adjunta a la anterior (máximo 99). Se muestrean 3,000 ventanas al azar por coste computacional, antes de ajustar el tokenizer.')
add(f"Las ventanas que exceden {M['max_seq_length']} tokens, incluidos los especiales, se subdividen recursivamente por palabras. Por eso el resultado final tiene {M['segments']:,} segmentos. El máximo observado fue {M['max_observed_tokens']} tokens: no se recortó silenciosamente el texto al calcular embeddings.")
add('Este criterio admite reseñas cortas y largas con una regla común, pero puede cortar oraciones y separar una negación de su contexto. Los fragmentos no son observaciones independientes: una reseña extensa aporta varios. El muestreo por ventanas favorece reseñas largas; las frecuencias describen fragmentos de esta muestra, no porcentajes de usuarios ni de toda la plataforma.')
w=D.words.describe()
table([['Estadística final','Valor'],['Reseñas distintas',M['reviews']],['Películas distintas',M['movies']],['Palabras por segmento: media',f'{w["mean"]:.1f}'],['Mediana',f'{w["50%"]:.0f}'],['Mínimo / máximo',f'{w["min"]:.0f} / {w["max"]:.0f}'],['Cuartiles 25% / 75%',f'{w["25%"]:.0f} / {w["75%"]:.0f}']], [370,110])
add('Sesgos y restricciones','Sub')
add('El corpus favorece cine apreciado y reseñas disponibles en la captura de 2024. El filtro español puede omitir textos con pocas palabras funcionales, variación dialectal o mezcla de idiomas, y no se validó contra un conjunto de etiquetas humanas. No se tradujeron reseñas. No se infiere nacionalidad a partir del idioma. La imagen del enunciado en Canvas no fue accesible; se utiliza el criterio explícito anterior.')
page()
heading('3. Metodología')
add('Se eligió <b>sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</b> [2]. Produce vectores de 384 dimensiones y admite español entre sus idiomas. Es una opción multilingüe compacta, adecuada para ejecutar localmente y agrupar similitud semántica sin entrenar un clasificador. No es un modelo específico de cine ni demuestra ser superior a todos los modelos españoles; no se realizó un benchmark de embeddings.')
add('Se conservó el contexto original para el encoder y se normalizaron los vectores. El flujo fue embeddings → UMAP → HDBSCAN → CountVectorizer → c-TF-IDF. UMAP reduce a cinco dimensiones con distancia coseno; HDBSCAN busca grupos densos. c-TF-IDF caracteriza cada grupo mediante términos distintivos [3]. Las stopwords españolas de NLTK se aplican exclusivamente al vectorizador y su copia se incluye en la entrega [5]. No se eliminaron nombres de películas: interesa comprobar su influencia.')
table([['Parámetro','A: exploratoria','B: grupos mayores'],['UMAP n_neighbors / n_components','15 / 5','15 / 5'],['UMAP min_dist / metric','0 / cosine','0 / cosine'],['UMAP random_state / n_jobs','42 / 1','42 / 1'],['HDBSCAN min_cluster_size','15','35'],['HDBSCAN min_samples','5','10'],['HDBSCAN metric / método','euclidean / eom','euclidean / eom'],['CountVectorizer ngram_range','(1, 1)','(1, 2)'],['CountVectorizer min_df','2','2'],['nr_topics','None','None']], [220,130,130])
add('A permite recuperar grupos pequeños; B exige mayor soporte y añade bigramas para representar expresiones. Se cambian varios parámetros como comparación exploratoria, por lo que no se atribuye un efecto causal a uno solo. ngram_range modifica la representación léxica; UMAP/HDBSCAN determinan los grupos. No se fuerza un número final con nr_topics [4]. Se comparten exactamente los mismos embeddings y segmentos.')
page()
heading('4. Comparación de configuraciones')
a,b=M['metrics']
table([['Medida','A','B'],['Tópicos (sin -1)',a['topics'],b['topics']],['Outliers',a['outliers'],b['outliers']],['Outliers (%)',f'{a["outlier_pct"]:.2f}',f'{b["outlier_pct"]:.2f}'],['Silhouette coseno',f'{a["silhouette_cosine"]:.3f}',f'{b["silhouette_cosine"]:.3f}'],['Diversidad top-10',f'{a["diversity_top10"]:.3f}',f'{b["diversity_top10"]:.3f}'],['Concentración por película',f'{a["movie_purity"]:.3f}',f'{b["movie_purity"]:.3f}']], [260,110,110])
add('Silhouette se calcula en embeddings originales normalizados, solo con segmentos asignados, sobre hasta 2,000 observaciones con semilla 42; no mide coherencia humana ni compara la misma submuestra si cambia la cobertura. Diversidad es términos únicos / (10 × tópicos): los bigramas pueden elevarla mecánicamente. Concentración es la suma, por tópico, del número de segmentos de su película más frecuente, dividida entre todos los asignados; un valor alto sugiere agrupación por título.', 'SmallX')
add(f"ARI entre particiones: {M['ari_all']:.3f} incluyendo -1 como etiqueta, y {M['ari_common_assigned']:.3f} sobre los {M['common_assigned']:,} segmentos asignados en ambas. ARI mide concordancia, no calidad; el primer valor trata los outliers como un grupo aunque no constituyan un tópico.")
add(N['comparison'])
add('Selección para el análisis detallado','Sub'); add(N['selection_reason'])
add('No hay etiquetas verdaderas ni evaluación humana independiente. Las métricas permiten describir compromisos; no sustentan una afirmación de exactitud temática o superioridad general.')
page()
heading(f'5. Tabla de tópicos: configuración {cfg}')
add('Resumen de get_topic_info con nombres interpretativos propios. El CSV incluido conserva las columnas completas y los nombres automáticos. Los porcentajes usan todos los segmentos como denominador, incluido -1.')
tab=[['ID','Nombre interpretativo','n','%','Términos c-TF-IDF']]
for r in T:
    i=r['Topic']; tab.append([i,labels.get(i,'Sin grupo denso (outliers)'),r['Count'],f'{100*r["Count"]/len(D):.1f}',', '.join(r['Representation'][:5])])
table(tab,[28,175,42,36,199])
add('Los IDs son locales al ajuste y no tienen significado ordinal. El nombre interpretativo se asignó después del modelado al leer términos y ejemplos; no intervino en el clustering.', 'SmallX')
page()
heading('6. Visualizaciones e interpretación')
img('topic_sizes.png')
add('Figura 1. Tamaño de los tópicos, excluyendo los outliers. Un segmento puede reflejar más de un tema, pero aquí tiene una sola etiqueta.', 'SmallX')
add(N['size_interpretation'])
page()
heading('7. Relaciones entre tópicos')
img('heatmap_centroids.png',440)
add('Figura 2. Similitud coseno entre el promedio de embeddings de cada tópico. La diagonal vale uno por construcción. Esta figura usa embeddings originales, no distancias UMAP ni una escala de causalidad.', 'SmallX')
add(N['heatmap_interpretation'])
page()
heading('8. Términos que caracterizan los grupos')
# En muchas categorías se parten los paneles en páginas para mantener legibilidad.
from PIL import Image as PILImage
with PILImage.open(R/'ctfidf.png') as full:
    panelrows=(len(ids)+1)//2
    for start in range(0,panelrows,3):
        if start: page(); heading('Términos c-TF-IDF (continuación)')
        crop=full.crop((0,int(full.height*start/panelrows),full.width,int(full.height*min(start+3,panelrows)/panelrows)))
        path=R/f'ctfidf_part_{start}.png'; crop.save(path); img(path.name)
add('Figura 3. Seis términos con mayor peso c-TF-IDF por tópico. El peso representa distintividad dentro del corpus; no es probabilidad del tópico ni conteo bruto.', 'SmallX')
add(N['terms_interpretation'])
page()
heading('9. Segmentos representativos y lectura crítica')
add('Se muestran tres segmentos seleccionados por get_representative_docs para cada tópico. Son fragmentos textuales reales, no paráfrasis ni ejemplos inventados. Los IDs permiten localizar el texto completo en las asignaciones; las películas se identifican por el slug del dataset. Los extractos se limitan a 55 palabras para lectura; [...] indica recorte editorial.')
for ix,i in enumerate(ids):
    if ix:
        page()
    heading(f'Tópico {i}. {labels[i]}')
    row=next(r for r in topic_rows if r['Topic']==i)
    add(f"<b>{row['Count']} segmentos ({100*row['Count']/len(D):.1f}% del corpus).</b> Términos: {escape(', '.join(row['Representation'][:10]))}.")
    add(N['topics'][str(i)]['interpretation'])
    for j,e in enumerate(E[str(i)][:3],1):
        add(f"Ejemplo {j} · {escape(e['id'])} · {escape(e['movie'])}",'SmallX')
        words=e['text'].split(); excerpt=' '.join(words[:55])+(' [...]' if len(words)>55 else '')
        add(escape(excerpt),'QuoteX')
    add('Alcance del nombre','Sub'); add(N['topics'][str(i)]['limitation'])
page()
heading('10. Outliers: qué significan y qué se hizo')
metric=next(x for x in M['metrics'] if x['configuration']==cfg)
add(f"La configuración {cfg} dejó {metric['outliers']:,} segmentos en -1 ({metric['outlier_pct']:.2f}%). Se conservaron en las asignaciones y en los denominadores del reporte. No se borraron, ni se reasignaron automáticamente. -1 indica que HDBSCAN no encontró pertenencia suficientemente densa; no significa reseña inválida, idioma incorrecto o ausencia de contenido.")
add(N['outliers'])
for r in D[D.topic==-1].sample(min(3,int((D.topic==-1).sum())),random_state=42).itertuples():
    add(f'{escape(r.segment_id)} · {escape(r.movie)}','SmallX')
    words=r.text.split(); add(escape(' '.join(words[:45])+(' [...]' if len(words)>45 else '')),'QuoteX')
add('BERTopic ofrece reducción de outliers por probabilidades, distribuciones o similitud [6]. Se optó por conservarlos porque forzar cobertura podría ocultar la heterogeneidad. Una extensión sería comparar reasignaciones con umbral de similitud y validación manual, informando por separado cobertura original y posterior.')
page()
heading('11. Discusión y conclusiones')
for s in N['discussion']: add(s)
add('Limitaciones y siguientes pasos','Sub')
add('Además del sesgo de películas bien valoradas y del filtrado de idioma, hay dependencia entre segmentos de una misma reseña, posible repetición de citas y un solo encoder. La deduplicación exacta no elimina paráfrasis. No se evaluó estabilidad con otras semillas ni con un corpus externo. La precisión de los nombres se basa en lectura asistida, sin anotadores independientes. Una mejora sería muestrear por reseña, segmentar por oraciones dentro del límite de tokens y comparar modelos de embeddings con una evaluación humana ciega.')
add('Bonus opcional y uso de IA','Sub')
add('No se ejecutó la comparación BERTopic contra un LLM por API. Gasto de API del proyecto: 0; tokens facturados por una API contratada para este experimento: 0. No se configuró un límite en una cuenta porque no se usó ninguna. La asistencia de Codex en este chat no se contabiliza como experimento de API ni se presentan sus etiquetas como un benchmark.')
add('Se utilizó IA generativa para localizar fuentes, escribir y depurar código, proponer nombres de tópicos y redactar la interpretación. Los embeddings, agrupamientos, ejemplos y métricas proceden de ejecución local real. La revisión de nombres realizada por el asistente es interpretativa y debe ser revisada por quien entrega el trabajo. No se atribuye validación humana independiente.')
page()
heading('12. Bibliografía y acceso al código')
refs=[
('1. Riyosha (2024). Letterboxd Movie Reviews (~90,000), versión 1. Kaggle.','https://www.kaggle.com/datasets/riyosha/letterboxd-movie-reviews-90000'),
('2. Sentence Transformers. Ficha del modelo paraphrase-multilingual-MiniLM-L12-v2.','https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
('3. Grootendorst, M. BERTopic: The Algorithm. Documentación oficial.','https://maartengr.github.io/BERTopic/algorithm/algorithm.html'),
('4. BERTopic. Parameter tuning. Documentación oficial.','https://maartengr.github.io/BERTopic/getting_started/parameter%20tuning/parametertuning.html'),
('5. NLTK. Repositorio nltk_data: stopwords.','https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/stopwords.xml'),
('6. BERTopic. Outlier reduction. Documentación oficial.','https://maartengr.github.io/BERTopic/getting_started/outlier_reduction/outlier_reduction.html')]
for title,url in refs:
    add(escape(title)); add(f'<link href="{escape(url)}" color="#126A82">{escape(url)}</link>','SmallX')
add('Fuentes consultadas el 30 de octubre de 2026. La revisión exacta del encoder está en data/embedding_metadata.json; el hash SHA-256 del CSV y el embudo de selección están en results/segmentation.json.', 'SmallX')
add('Código y archivos de respaldo','Sub')
for title,url in [('Código de adquisición y segmentación','../../src/prepare.py'),('Código BERTopic y métricas','../../src/model.py'),('Código de este reporte','../../src/report.py'),('Guía de ejecución','../../README.md'),('Tabla completa A con nombres','../../results/topic_info_named_A.csv'),('Tabla completa B con nombres','../../results/topic_info_named_B.csv'),('Catálogo A: 52 tópicos y ejemplos','../../results/catalogue_A.md'),('Catálogo B: 6 tópicos y ejemplos','../../results/catalogue_B.md')]:
    add(f'<link href="{url}" color="#126A82">{title}</link>')
add('Estos enlaces son relativos al paquete de entrega; conservar su estructura de carpetas. No se creó un repositorio público. Si la institución exige una URL pública, subir el paquete a un repositorio propio o almacenamiento institucional y compartir ese enlace. La entrega local no presupone publicación ni acceso permanente desde otro equipo.', 'SmallX')

def footer(canvas,doc):
    canvas.saveState(); canvas.setStrokeColor(colors.HexColor('#CBDDE4')); canvas.line(50,45,545,45)
    canvas.setFont('Body',8); canvas.setFillColor(colors.HexColor('#526C78'))
    canvas.drawString(50,31,'LETTERBOXD EN ESPAÑOL | BERTopic | 2026'); canvas.drawRightString(545,31,str(doc.page)); canvas.restoreState()
target=ROOT/'output/pdf/reporte_letterboxd.pdf'; target.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(target),pagesize=(595.28,841.89),rightMargin=55,leftMargin=55,topMargin=48,bottomMargin=60,title='Tópicos en reseñas de Letterboxd en español',author='Análisis académico asistido por IA')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
