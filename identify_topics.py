'''
Para usar KMeans en lugar de NMF, se reemplaza el modelo NMF por KMeans, y en lugar de acceder a componentes temáticos, accederemos a los clusters generados.

Explicación de cambios:

1.- Cambio de modelo: Se usa KMeans en lugar de NMF.
2.- Extracción de palabras representativas: En lugar de nmf_model.components_, se utiliza kmeans_model.cluster_centers_ para obtener las palabras más representativas de cada cluster.
3.- Muestra de resultados: Se muestran las palabras de cada cluster sin los valores numéricos de probabilidad, solo la palabra.

Con este enfoque, KMeans agrupa palabras similares en clusters, mostrando los términos que caracterizan cada grupo.
'''

import re
import os
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from nltk.tokenize import word_tokenize
from pathlib import Path
import nltk
nltk.download('punkt_tab')
import fitz  # PyMuPDF
import numpy as np
import shutil

# Función para extraer texto de un archivo PDF manteniendo el orden lógico
def extraer_texto_de_pdf(ruta_pdf):
  # Abrir el documento PDF
  documento = fitz.open(ruta_pdf)
  texto_completo = []

  # Iterar sobre cada página del documento
  for pagina in documento:
    # get_text("blocks") extrae el texto agrupado visualmente (párrafos/columnas)
    # Retorna una lista de tuplas: (x0, y0, x1, y1, "texto", block_no, block_type)
    bloques = pagina.get_text("blocks")
    
    # Ordenamos los bloques: Primero de arriba a abajo (y0), luego de izquierda a derecha (x0)
    # Esto asegura que si hay columnas, lea primero la col 1 completa antes de la col 2.
    # Nota: A veces PyMuPDF ya los trae ordenados, pero esto fuerza el orden de lectura humano.

    # La clave de ordenamiento es una tupla (y0, x0)
    bloques.sort(key=lambda b: (b[1], b[0]))

    # Extraer el texto de cada bloque ordenado
    for b in bloques:
      # b[4] es el contenido de texto del bloque
      texto_bloque = b[4]
      
      # Filtramos bloques que no sean texto (imágenes o gráficos a veces generan basura)
      if texto_bloque.strip():
        texto_completo.append(texto_bloque)

  # Unimos todo con saltos de línea para mantener separación de párrafos
  return "\n".join(texto_completo)

# Función para convertir todos los archivos PDF en una carpeta a archivos .txt
def pdf_a_txt(carpeta):
  # Ruta del directorio
  directorio = Path(carpeta)

  # Iterar sobre todos los archivos PDF en el directorio
  for archivo_pdf in directorio.rglob("*.pdf"):
    # Extraer el texto del PDF
    print(f"Procesando: {archivo_pdf.name}")
    texto = extraer_texto_de_pdf(f"{carpeta}/{archivo_pdf.name}")
    
    # Crear carpeta de salida de texto si no existe 
    if not os.path.exists("corpus_txt"):
      os.makedirs("corpus_txt")

    # Guardar el texto extraído en un archivo .txt en la carpeta corpus_txt
    with open(f"corpus_txt/{archivo_pdf.name.replace(".pdf", "")}.txt", "w") as file:
      file.write(texto)

# Función para cargar stopwords desde un archivo
def cargar_stopwords(archivo_stopwords):
  stopwords_personalizadas = set()
  with open(archivo_stopwords, 'r', encoding='utf-8') as f:
    for linea in f:
      palabra = linea.split(':')[0].strip().lower()
      stopwords_personalizadas.add(palabra)
  return list(stopwords_personalizadas)  # Convertimos el conjunto a una lista

# Función para cargar y preprocesar el corpus desde una carpeta con archivos .txt
def cargar_corpus(carpeta, stopwords):
  corpus = []
  archivos = []

  # Iterar sobre todos los archivos .txt en la carpeta
  for archivo in os.listdir(carpeta):
    if archivo.endswith('.txt'):
      ruta = os.path.join(carpeta, archivo)
      print(f"Leyendo archivo: {archivo}")
      # Guardar el nombre del archivo
      archivos.append(archivo)
      
      # Leer y preprocesar el texto
      with open(ruta, 'r', encoding='utf-8') as f:
        texto = f.read().lower()  # Convertir a minúsculas
        texto = re.sub(r'\d+', '', texto)  # Eliminar números
        texto = re.sub(f"[{string.punctuation}«»”“•‘’¿?¡●ʻ–©…—]", " ", texto)  # Eliminar puntuación

        palabras = word_tokenize(texto)
        palabras = [palabra for palabra in palabras if palabra not in stopwords]
        corpus.append(" ".join(palabras))
  return corpus, archivos

# Función para selección automática del número óptimo de clusters usando Silhouette Score
def seleccionar_num_clusters(X, min_k=2, max_k=10):
  silhouette_scores = []
  posibles_k = range(min_k, max_k + 1)
  for k in posibles_k:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)
  # Opcional: mostrar la gráfica del Silhouette Score
  plt.plot(posibles_k, silhouette_scores, marker='o')
  plt.xlabel('Número de clusters (k)')
  plt.ylabel('Silhouette Score')
  plt.title('Selección automática de clusters')
  plt.show()
  # Selecciona el k con mayor Silhouette Score
  mejor_k = posibles_k[silhouette_scores.index(max(silhouette_scores))]
  print(f"Mejor número de clusters según Silhouette Score: {mejor_k}")
  return mejor_k

# Función para sugerencia de etiqueta de tópico
def sugerencia_etiqueta(palabras_representativas, n=2):
  '''
  Se toman las n palabras más representativas para formar una sugerencia
  de etiqueta.
  '''
  etiqueta = "-".join(palabras_representativas[:n])
  del palabras_representativas[:n]
  return etiqueta

# === Programa Principal ===

# Convertir corpus de archivos pdf a txt
pdf_a_txt("corpus_pdf/")

# Cargar las stopwords
archivo_stopwords = 'stopwords/Purepecha_stopwords.txt'
stopwords_personalizadas = cargar_stopwords(archivo_stopwords)

# Cargar el corpus desde la carpeta corpus_txt y obtener corpus y nombres de archivos
carpeta_corpus = "corpus_txt/"
corpus, archivos = cargar_corpus(carpeta_corpus, stopwords_personalizadas)

# Convertir el corpus a una representación TF-IDF
# e.g. max_df=0.5 elimina palabras que aparecen en más del 50% de los documentos
# e.g. min_df=2 elimina palabras que aparecen en solo 1 documento

# En corpus pequeños, valores altos de min_df pueden eliminar demasiadas palabras útiles.
# En corpus grandes, puedes usar valores más altos para ambos parámetros para filtrar mejor palabras poco informativas o demasiado raras.
vectorizer = TfidfVectorizer(max_df=0.45, min_df=1, stop_words=stopwords_personalizadas)
X = vectorizer.fit_transform(corpus)

# Entrenar el modelo KMeans
num_clusters = seleccionar_num_clusters(X, 2, len(os.listdir(carpeta_corpus)) - 1) # Número máximo de clusters igual al número total de documentos menos uno
kmeans_model = KMeans(n_clusters=num_clusters, random_state=0)
kmeans_model.fit(X)

print("\nEjemplo de clusterización de documentos:")
# Mostrar cuántos documentos hay en cada cluster
for i in range(num_clusters):
    print(f"Cluster {i}: {np.sum(kmeans_model.labels_ == i)} documentos")

# Se crea una carpeta para agrupar los subdirectorios generados 
if not os.path.exists('directorios'):
  os.makedirs('directorios')

# Mostrar los tópicos y palabras más representativas
print("Tópicos detectados y palabras más representativas:")
feature_names = vectorizer.get_feature_names_out()
etiquetas = {} # Diccionario para mapear índices de cluster a etiquetas
for cluster_idx in range(num_clusters):
  # Obtener los índices de los términos más cercanos al centroide del cluster
  centroide = kmeans_model.cluster_centers_[cluster_idx]
  palabras_representativas = [feature_names[i] for i in centroide.argsort()[:-11:-1]]

  # Se obtiene una sugerencia de etiqueta
  etiqueta = sugerencia_etiqueta(palabras_representativas)
  print(f"\nTópico {cluster_idx + 1} {etiqueta}: {' '.join(palabras_representativas)}")

  # Guardar la etiqueta en el diccionario
  etiquetas[cluster_idx] = etiqueta

  # Crear carpeta para la etiqueta si no existe
  if not os.path.exists(f"directorios/{etiqueta}"):
    os.makedirs(f"directorios/{etiqueta}")


# Copiar los archivos PDF a las carpetas correspondientes según su cluster
for i,j in dict(zip(archivos, kmeans_model.labels_)).items():
  shutil.copy(f"corpus_pdf/{i.replace('.txt', '.pdf')}", f"directorios/{etiquetas[j]}")