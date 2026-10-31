"""Añade nombres analíticos y tres ejemplos íntegros para ambas configuraciones."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
names_a=[
'Infancia y relaciones con los adultos','Ford y el mito de Liberty Valance',
'Drama familiar y tradición japonesa','Kurosawa y puesta en escena japonesa',
'Cámara, encuadre y narración','Ética, justicia y condición humana',
'Infancia, pobreza y memoria (mixto)','Lynch y la humanidad de The Elephant Man',
'Yugoslavia e historia en Underground','Cine de autor: Tarkovsky y Tarr',
'Guerra, nazismo y resistencia','Slam Dunk: deporte y adaptación',
'Experiencia de visionado y expectativas','Fe, culpa y El quinto sello',
'Wyler y figuras de la fama (mixto)','Amor, emoción y recuerdo del cine',
'Hitchcock y el suspense','El oficio de la crítica cinematográfica',
'Dersu Uzala: amistad y naturaleza','Evangelion: Shinji e identidad',
'Wilder y desenlaces del cine clásico','János y el circo en Werckmeister',
'Chaplin: humor y esperanza','Marcello y escenas de La dolce vita (mixto)',
'Música y dilemas en La vida de los otros','Tiempo, vida y bienestar personal',
'Lecturas del cine clásico','Arte, violencia y ambición (mixto)',
'Kiarostami: viajes y diálogo','Cinema Paradiso y películas favoritas (mixto)',
'Bergman: familia, muerte y Dios','Pobreza, vejez y vínculos familiares',
'Nolan: trucos, narración y Oppenheimer','Schindler y la representación del Holocausto',
'Elogio de actuaciones y montaje (mixto)','Cine negro y violencia política',
'Miss Cooper y la amenaza en The Night of the Hunter','Dreyer y la lectura de Ordet',
'Policial, crimen y códigos morales','Entusiasmo por Spider-Man y otros filmes (mixto)',
'Identidad, rostro y deshumanización','Abel Gance y la épica de Napoléon',
'La trilogía de El señor de los anillos','Citas y diálogos de personajes (difuso)',
'Llanto, tristeza y respuesta emocional','Vértigo: Scottie, Madeleine y recorrido',
'Recuerdos del visionado de Fincher','Leone: duelos y western',
'Neorrealismo y Visconti','Objetos, espacio y símbolos narrativos',
'Magnolia: personajes y medios','Wiesler y Dreyman: autoría y reconocimiento']
interpretation=json.loads((R/'interpretation.json').read_text(encoding='utf-8'))
names_b={int(k):v['name'] for k,v in interpretation['topics'].items()}
for cfg,names in [('A',dict(enumerate(names_a))),('B',names_b)]:
    info=pd.read_csv(R/f'topic_info_{cfg}.csv')
    assert set(info.Topic)-{-1}==set(names)
    info['Interpretive_Name']=info.Topic.map({-1:'Sin grupo denso (outliers)',**names})
    info.to_csv(R/f'topic_info_named_{cfg}.csv',index=False)
    reps=json.loads((R/f'representatives_{cfg}.json').read_text(encoding='utf-8'))
    lines=[f'# Catálogo completo: configuración {cfg}', '',
           'Nombres interpretativos propuestos por el asistente después de leer términos y ejemplos. '
           'Los tópicos marcados como mixtos o difusos requieren especial cautela; no son etiquetas verdaderas.', '']
    for topic,name in names.items():
        row=info[info.Topic==topic].iloc[0]
        lines.extend([f'## {topic}. {name}', '', f'Segmentos: {row.Count}. Nombre automático: {row.Name}.', ''])
        for e in reps[str(topic)]:
            lines.extend([f"**{e['id']} | {e['movie']}**",'',e['text'],''])
    (R/f'catalogue_{cfg}.md').write_text('\n'.join(lines),encoding='utf-8')
print('Catálogos y tablas con nombres generados para A/B.')
