import os
import shutil
import numpy as np
import umap
import hdbscan
from tqdm import tqdm
import json

from PROCESSING.reduction_vit import extraer_features_deepfaune

def auto_cluster_dl_hdbscan(
    input_dir,
    output_dir,
    temp_features_file='temp_features.npy',
    temp_filenames_file='temp_filenames.json',
    batch_size=32,
    min_cluster_size=15,
    min_samples=None,
    n_neighborsumap=15
):
    """
    Pipeline: DEEPFAUNE -> UMAP -> HDBSCAN -> File Sorting
    """
    
    # --- 1. EXTRACCIÓN (Igual que antes) ---
    print(f"--- Paso 1: Extracción de Features con  DEEPFAUNE ---")
    
    if not os.path.exists(temp_features_file):
        extraer_features_deepfaune(
            input_dir=input_dir,
            output_features=temp_features_file,
            output_filenames=temp_filenames_file,
            weights_path=os.path.join(os.getcwd(),"MODELS","deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt"),
            batch_size=batch_size
        )
    else:
        print("Features ya extraídos previamente. Cargando...")

    if not os.path.exists(temp_features_file):
        print("Error: No se generaron features.")
        return

    features = np.load(temp_features_file)
    with open(temp_filenames_file, 'r') as f:
        filenames = json.load(f)

    print(f"Features cargados: {features.shape}") 

    # --- 2. REDUCCIÓN CON UMAP ---
    print(f"--- Paso 2: Reducción de Dimensionalidad (UMAP) ---")
    
    # NOTA: Para HDBSCAN, es mejor bajar a pocas dimensiones (ej. 10 a 50).
    # 400 dimensiones es demasiado disperso para calcular densidades correctamente.
    reducer = umap.UMAP(
        n_neighbors=n_neighborsumap,    # Balance entre estructura local y global
        n_components=15,   # <--- CAMBIO: Bajamos a 15 dimensiones para que HDBSCAN funcione bien
        metric='cosine',   
        min_dist=0.0,      # Compactar los puntos ayuda a HDBSCAN
        random_state=42
    )
    embedding = reducer.fit_transform(features)
    print(f"Dimensiones reducidas con UMAP: {embedding.shape}")

    # --- 3. CLUSTERING CON HDBSCAN ---
    print(f"--- Paso 3: Clustering con HDBSCAN (min_cluster_size={min_cluster_size}) ---")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples, # Si es None, por defecto es igual a min_cluster_size
        metric='euclidean',      # En UMAP reducido, euclidiana suele ir bien
        cluster_selection_method='leaf' # 'eom' suele hacer grupos más grandes, 'leaf' más pequeños
    )
    
    labels = clusterer.fit_predict(embedding)
    
    # Contar cuántos clusters salieron (excluyendo el -1 que es ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\n>>> RESULTADO: Se encontraron {n_clusters} clusters y {n_noise} imágenes de Ruido (-1).")

    # --- 4. ORGANIZACIÓN DE ARCHIVOS ---
    print(f"--- Paso 4: Organizando archivos ---")

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    conteo = {}

    for i, src_path in enumerate(tqdm(filenames, desc="Moviendo archivos")):
        label = labels[i]
        filename = os.path.basename(src_path)
        
        # Lógica especial para el ruido
        if label == -1:
            folder_name = "Cluster_Ruido_Outliers"
        else:
            folder_name = f"Cluster_{label:02d}"
        
        cluster_folder = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_folder, exist_ok=True)
        
        dst_path = os.path.join(cluster_folder, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            conteo[folder_name] = conteo.get(folder_name, 0) + 1
        except Exception as e:
            print(f"Error copiando {filename}: {e}")

    print("\n--- RESUMEN FINAL ---")
    # Ordenar para imprimir bonito
    sorted_keys = sorted(conteo.keys())
    for k in sorted_keys:
        print(f"{k}: {conteo[k]} imágenes")