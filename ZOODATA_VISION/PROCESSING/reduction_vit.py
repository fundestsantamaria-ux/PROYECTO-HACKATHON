import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json
from tqdm import tqdm

def extraer_features_deepfaune(
    input_dir,
    output_features,
    output_filenames,
    weights_path, # Ruta al archivo .pt de DeepFaune
    batch_size=32
):
    """
    Extrae vectores de características usando DeepFaune (ViT Large DINOv2).
    
    Parámetros:
    - input_dir: carpeta con imágenes.
    - output_features: archivo .npy de salida.
    - output_filenames: archivo .json de salida.
    - weights_path: ruta absoluta al modelo .pt de DeepFaune.
    """

    # --- 1. PREPARAR MODELO (DeepFaune) ---
    print(f"--- Configurando DeepFaune ViT ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Crear la arquitectura vacía
    model = timm.create_model(
        "vit_large_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=38, 
        dynamic_img_size=True
    )

    print(f"Cargando pesos desde: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No se encontró el modelo en: {weights_path}")

    # Cargar el checkpoint
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Limpiar prefijo "base_model." del state_dict (Lógica original de DeepFaune)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("base_model."):
            new_key = key.replace("base_model.", "")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    
    # --- PASO CRÍTICO: ELIMINAR CABEZAL DE CLASIFICACIÓN ---
    # Esto hace que el modelo devuelva el vector de características (1024 dim)
    # en lugar de la probabilidad de las 38 clases.
    model.reset_classifier(0) 

    model = model.to(device)
    model.eval()

    # Congelar parámetros (ahorra memoria y asegura que no cambien)
    for p in model.parameters():
        p.requires_grad = False

    # --- 2. TRANSFORMACIONES (Específicas de DeepFaune) ---
    # DeepFaune usa 182x182. Es vital respetar esto.
    preprocess = transforms.Compose([
        transforms.Resize((182, 182)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --- 3. RECOLECTAR IMÁGENES ---
    print("Buscando imágenes...")
    image_paths = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))

    print(f"Total de imágenes a procesar: {len(image_paths)}")

    features_list = []
    valid_paths = []

    # --- 4. PROCESAMIENTO EN LOTES ---
    print("Iniciando extracción de features...")
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            
            # Cargar lote
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    tensor = preprocess(img)
                    batch_tensors.append(tensor)
                    valid_paths.append(img_path) # Solo agregamos si cargó bien
                except Exception as e:
                    print(f"Error cargando {img_path}: {e}")

            if not batch_tensors:
                continue

            # Inferencia
            batch_stack = torch.stack(batch_tensors).to(device)
            outputs = model(batch_stack) # Salida: [Batch_Size, 1024]

            features_list.append(outputs.cpu().numpy())

    # --- 5. GUARDAR ---
    if features_list:
        all_features = np.concatenate(features_list, axis=0)
        print(f"\nExtracción completa.")
        print(f"Dimensiones finales: {all_features.shape}") 
        # Debería ser (N_imagenes, 1024) para ViT-Large

        np.save(output_features, all_features)
        print(f"Features guardados en: {output_features}")

        with open(output_filenames, 'w') as f:
            json.dump(valid_paths, f)
        print(f"Lista de archivos guardada en: {output_filenames}")

    else:
        print("No se generaron características.")