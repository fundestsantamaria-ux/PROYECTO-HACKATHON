# ════════════════════════════════════════════════════════════════════════════
# INFERENCIA EN CARPETA COMPLETA (STANDALONE - CARGA MODELOS DESDE PESOS)
# Pipeline: Imágenes → Embeddings (ViT) → Predicciones (MLP)
# ════════════════════════════════════════════════════════════════════════════

import os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE RUTAS Y DISPOSITIVO
# ════════════════════════════════════════════════════════════════════════════

# Rutas de los modelos
VIT_WEIGHTS_PATH = "MODELS/deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt"
MLP_WEIGHTS_PATH = "MODELS/best_mlp_model.pth"
SOURCE_IMAGES = "DATASET_PRUEBA"

# Directorio raíz para resultados (guardar CSV de predicciones aquí)
RESULTS_DIR = os.path.join(os.getcwd(), 'RESULTS')
# Asegurar que exista
os.makedirs(RESULTS_DIR, exist_ok=True)

# Clases del modelo (deben coincidir con las usadas en entrenamiento)
CLASSES = ['AVE_GRANDE', 'AVE_PEQUEÑA', 'MAMIFERO_GRANDE', 'MAMIFERO_MEDIANO', 'MAMIFERO_PEQUEÑO', 'RUIDO']

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Usando dispositivo: {device}")

# ════════════════════════════════════════════════════════════════════════════
# DEFINICIÓN DEL CLASIFICADOR MLP
# ════════════════════════════════════════════════════════════════════════════

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden=512, num_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO ViT (EXTRACTOR DE EMBEDDINGS)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("📥 CARGANDO MODELO ViT (Extractor de Embeddings)")
print("=" * 70)

vit_model = timm.create_model(
    "vit_large_patch14_dinov2.lvd142m",
    pretrained=False,
    num_classes=38,
    dynamic_img_size=True
)

# Cargar pesos del ViT
checkpoint_vit = torch.load(VIT_WEIGHTS_PATH, map_location=device, weights_only=False)
state_dict = checkpoint_vit["state_dict"]

# Remover prefijo "base_model." de las claves
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("base_model."):
        new_key = key.replace("base_model.", "")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

vit_model.load_state_dict(new_state_dict)
vit_model = vit_model.to(device)
vit_model.eval()

# Congelar parámetros
for p in vit_model.parameters():
    p.requires_grad = False

print(f"✅ ViT cargado desde: {VIT_WEIGHTS_PATH}")

# ════════════════════════════════════════════════════════════════════════════
# CARGAR MODELO MLP (CLASIFICADOR)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("📥 CARGANDO MODELO MLP (Clasificador)")
print("=" * 70)

mlp_model = MLPClassifier(num_classes=len(CLASSES)).to(device)
checkpoint_mlp = torch.load(MLP_WEIGHTS_PATH, map_location=device, weights_only=False)
mlp_model.load_state_dict(checkpoint_mlp['model_state_dict'])
mlp_model.eval()

print(f"✅ MLP cargado desde: {MLP_WEIGHTS_PATH}")
print(f"   Epoch guardado: {checkpoint_mlp['epoch'] + 1}")
print(f"   Val F1-Score: {checkpoint_mlp['val_f1']:.4f}")

# ════════════════════════════════════════════════════════════════════════════
# PREPROCESAMIENTO (DeepFaune oficial)
# ════════════════════════════════════════════════════════════════════════════

preprocess = transforms.Compose([
    transforms.Resize((182, 182)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE INFERENCIA EN CARPETA
# ════════════════════════════════════════════════════════════════════════════

def predict_folder(folder_path, save_csv=True, show_results=True):
    """
    Realiza inferencia en todas las imágenes de una carpeta.
    
    Pipeline:
        1. Cargar imagen
        2. Extraer embedding con ViT (1024D)
        3. Clasificar con MLP entrenado
    
    Args:
        folder_path: Ruta a la carpeta con imágenes
        save_csv: Si True, guarda resultados en CSV
        show_results: Si True, muestra resumen en consola
    
    Returns:
        DataFrame con resultados (nombre_archivo, clase_predicha, confianza)
    """
    
    # Verificar que la carpeta existe
    if not os.path.exists(folder_path):
        print(f"❌ Error: La carpeta '{folder_path}' no existe")
        return None
    
    # Obtener lista de imágenes
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if len(image_files) == 0:
        print(f"❌ No se encontraron imágenes en '{folder_path}'")
        return None
    
    # Crear diccionario de archivos originales (nombre_base -> nombre_completo)
    source_files_map = {}
    if os.path.exists(SOURCE_IMAGES):
        for f in os.listdir(SOURCE_IMAGES):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                nombre_base = os.path.splitext(f)[0]
                source_files_map[nombre_base] = f
    
    print("\n" + "=" * 70)
    print("🔮 INFERENCIA EN CARPETA")
    print("=" * 70)
    print(f"📁 Carpeta: {folder_path}")
    print(f"🖼️  Imágenes encontradas: {len(image_files)}")
    print(f"📂 Archivos originales mapeados: {len(source_files_map)}")
    print("=" * 70 + "\n")
    
    results = []
    
    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)
        
        try:
            # PASO 1: Cargar imagen
            img = Image.open(image_path).convert("RGB")
            
            # PASO 2: Extraer embedding con ViT (1024D)
            x = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = vit_model.forward_features(x)
                embedding = features[:, 0, :]  # Token CLS → 1024D
            
            # PASO 3: Clasificar con MLP
            with torch.no_grad():
                logits = mlp_model(embedding)
                probs = torch.softmax(logits, dim=1)[0]
                confidence, pred_idx = torch.max(probs, dim=0)
            
            predicted_class = CLASSES[pred_idx.item()]
            confidence_value = confidence.item()
            
            # Extraer nombre base (antes de _crop)
            name_without_ext = os.path.splitext(filename)[0]  # quitar .jpg
            if '_crop' in name_without_ext:
                nombre_base = name_without_ext.split('_crop')[0]
            else:
                nombre_base = name_without_ext
            
            # Buscar archivo original con extensión correcta en SOURCE_IMAGES
            if nombre_base in source_files_map:
                archivo_parent = os.path.join(SOURCE_IMAGES, source_files_map[nombre_base])
            else:
                archivo_parent = nombre_base  # Fallback si no se encuentra
            
            # Ruta completa del archivo (carpeta + nombre)
            ruta_completa = os.path.join(folder_path, filename)
            
            results.append({
                'archivo': ruta_completa,
                'archivo_parent': archivo_parent,
                'clase_predicha': predicted_class,
                'confianza': confidence_value
            })
            
            # Mostrar progreso cada 10 imágenes o al final
            if show_results and ((idx + 1) % 10 == 0 or idx == len(image_files) - 1):
                print(f"Procesadas: {idx + 1}/{len(image_files)} imágenes...")
                
        except Exception as e:
            print(f"⚠️  Error procesando '{filename}': {str(e)}")
            # Extraer nombre base para casos de error también
            name_without_ext = os.path.splitext(filename)[0]
            if '_crop' in name_without_ext:
                nombre_base = name_without_ext.split('_crop')[0]
            else:
                nombre_base = name_without_ext
            
            # Buscar archivo original con extensión correcta en SOURCE_IMAGES
            if nombre_base in source_files_map:
                archivo_parent = os.path.join(SOURCE_IMAGES, source_files_map[nombre_base])
            else:
                archivo_parent = nombre_base  # Fallback si no se encuentra
            
            # Ruta completa del archivo (carpeta + nombre)
            ruta_completa = os.path.join(folder_path, filename)
            
            results.append({
                'archivo': ruta_completa,
                'archivo_parent': archivo_parent,
                'clase_predicha': 'ERROR',
                'confianza': 0.0
            })
    
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)
    
    # MOSTRAR RESUMEN
    if show_results:
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE PREDICCIONES")
        print("=" * 70)
        
        # Conteo por clase
        class_counts = df_results['clase_predicha'].value_counts()
        print("\nDistribución de predicciones:")
        print("-" * 40)
        for cls, count in class_counts.items():
            percentage = (count / len(df_results)) * 100
            bar = "█" * int(percentage / 3)
            print(f"{cls:25} | {count:4} ({percentage:5.1f}%) {bar}")
        
        # Estadísticas de confianza
        valid_conf = df_results[df_results['clase_predicha'] != 'ERROR']['confianza']
        if len(valid_conf) > 0:
            print(f"\n📈 Estadísticas de confianza:")
            print(f"   Promedio: {valid_conf.mean():.4f}")
            print(f"   Mínima:   {valid_conf.min():.4f}")
            print(f"   Máxima:   {valid_conf.max():.4f}")
        
        # Imágenes con baja confianza (< 0.5)
        low_confidence = df_results[df_results['confianza'] < 0.5]
        if len(low_confidence) > 0:
            print(f"\n⚠️  Imágenes con baja confianza (<50%): {len(low_confidence)}")
            for _, row in low_confidence.head(5).iterrows():
                print(f"   • {row['archivo']}: {row['clase_predicha']} ({row['confianza']:.2%})")
    
    # GUARDAR CSV
    if save_csv:
        csv_path = os.path.join(RESULTS_DIR, "predicciones.csv")  # Guardar dentro de RESULTS
        df_results.to_csv(csv_path, index=False)
        print(f"\n💾 Resultados guardados en: {csv_path}")
    
    print("\n" + "=" * 70)
    print("✅ Inferencia completada")
    print("=" * 70)
    
    return df_results


# ════════════════════════════════════════════════════════════════════════════
# EJECUTAR INFERENCIA
# ════════════════════════════════════════════════════════════════════════════

# Carpeta con imágenes a clasificar
carpeta_inferencia = "RESULTS/crops_clahe"

# Ejecutar inferencia (modelos ya cargados desde pesos)
resultados = predict_folder(
    folder_path=carpeta_inferencia,
    save_csv=True,
    show_results=True
)

# Ver primeras filas del DataFrame
if resultados is not None:
    print("\n📋 Primeras 10 predicciones:")
    print(resultados.head(10).to_string(index=False))