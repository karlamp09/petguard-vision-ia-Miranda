# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 08:49:46 2025

@author: alumnoc2a70105
"""

from transformers import pipeline
from PIL import Image
import json
from sklearn.cluster import KMeans
import numpy as np

# ================================
# 1. Cargar imagen
# ================================
img_path = "assets/pug.jpg"   # <-- reemplaza por la ruta real
img = Image.open(img_path)

# ================================
# 2. MODELO: Descripción de imagen
# ================================
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
descripcion = captioner(img)[0]["generated_text"]

# ================================
# 3. MODELO: Clasificación de imagen
# ================================
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
clasificacion = classifier(img)

# ================================
# 4. EXTRA: Extractor de colores dominantes
# ================================
def colores_dominantes(image, k=3):
    image = image.resize((100, 100))
    data = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(data)

    colors = kmeans.cluster_centers_.astype(int)
    return colors.tolist()

colores = colores_dominantes(img)

# ================================
# 5. Armar el JSON final
# ================================
resultado = {
    "descripcion_imagen": descripcion,
    "clasificacion": clasificacion,
    "colores_dominantes_rgb": colores
}

# ================================
# 6. Guardar JSON
# ================================
with open("resultado_imagen.json", "w") as f:
    json.dump(resultado, f, indent=4)

print("ANÁLISIS COMPLETO:\n")
print(json.dumps(resultado, indent=4))
