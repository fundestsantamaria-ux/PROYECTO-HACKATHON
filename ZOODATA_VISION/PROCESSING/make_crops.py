import json
import os
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# Para evitar errores con imágenes truncadas o muy grandes
ImageFile.LOAD_TRUNCATED_IMAGES = True

def calculate_median_size(json_data, input_folder, conf_threshold, accepted_categories):
    """
    Pasada 1: Recolecta las dimensiones (en píxeles) de todos los bboxes válidos
    para encontrar la mediana.
    """
    widths = []
    heights = []
    
    print("--- Fase 1: Analizando dimensiones para encontrar la mediana ---")
    
    for entry in tqdm(json_data['images'], desc="Analizando tamaños"):
        filename = entry['file']
        img_path = os.path.join(input_folder, filename)
        
        # Saltamos si no existe, pero solo leemos el header de la imagen (rápido)
        if not os.path.exists(img_path):
            continue
            
        detections = [d for d in entry.get('detections', []) 
                      if d['category'] in accepted_categories and d['conf'] >= conf_threshold]
        
        if not detections:
            continue

        try:
            # Abrimos solo para obtener tamaño (lazy loading)
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
                for det in detections:
                    _, _, w_rel, h_rel = det['bbox']
                    # Convertir relativo a absoluto
                    widths.append(w_rel * img_w)
                    heights.append(h_rel * img_h)
                    
        except Exception:
            pass # Ignoramos errores en esta fase para no detener el flujo

    if not widths:
        return None, None

    # Calculamos la mediana
    median_w = int(np.percentile(widths, 75))
    median_h = int(np.percentile(heights, 75))
    
    print(f"\n>>> TAMAÑO MEDIO CALCULADO: {median_w}x{median_h} px")
    return median_w, median_h

def get_expanded_crop_coords(img_w, img_h, bbox, target_w, target_h):
    """
    Calcula coordenadas para recortar un área de tamaño fijo (target_w, target_h)
    centrada en el bbox original. Maneja los bordes de la imagen.
    """
    x_rel, y_rel, w_rel, h_rel = bbox
    
    # Coordenadas absolutas del bbox original (el animal pequeño)
    box_x = x_rel * img_w
    box_y = y_rel * img_h
    box_w = w_rel * img_w
    box_h = h_rel * img_h
    
    # Encontrar el centro del animal
    center_x = box_x + (box_w / 2)
    center_y = box_y + (box_h / 2)
    
    # Calcular las nuevas coordenadas (Izquierda/Arriba) restando la mitad del target al centro
    new_left = int(center_x - (target_w / 2))
    new_top = int(center_y - (target_h / 2))
    
    # --- AJUSTE DE BORDES (CLAMPING) ---
    # Si nos salimos por la izquierda, pegamos el corte al borde 0
    if new_left < 0:
        new_left = 0
    # Si nos salimos por arriba, pegamos al borde 0
    if new_top < 0:
        new_top = 0
        
    # Calcular derecha y abajo basados en el nuevo left/top
    new_right = new_left + target_w
    new_bottom = new_top + target_h
    
    # Si nos salimos por la derecha, empujamos todo hacia la izquierda
    if new_right > img_w:
        new_right = img_w
        new_left = max(0, new_right - target_w) # Asegurar que no sea negativo
        
    # Si nos salimos por abajo, empujamos todo hacia arriba
    if new_bottom > img_h:
        new_bottom = img_h
        new_top = max(0, new_bottom - target_h)

    return new_left, new_top, new_right, new_bottom

def make_crops(json_file, input_folder, output_folder,
                     conf_threshold=0.4,
                     accepted_categories=['1']):
    
    os.makedirs(output_folder, exist_ok=True)

    print(f"Cargando {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    print("\n--- Iniciando Recorte Simple (Solo BBox del Animal) ---")
    
    crops_count = 0
    errors = 0
    
    for entry in tqdm(data['images'], desc="Generando crops"):
        filename = entry['file']
        img_path = os.path.join(input_folder, filename)

        if not os.path.exists(img_path):
            continue

        detections = entry.get('detections', [])
        # Filtrar detecciones válidas primero
        valid_dets = [d for d in detections if d['category'] in accepted_categories and d['conf'] >= conf_threshold]
        
        if not valid_dets:
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
                
                # Convertir a RGB si es necesario (para guardar como jpg)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                for i, det in enumerate(valid_dets):
                    x_rel, y_rel, w_rel, h_rel = det['bbox']
                    
                    # Convertir coordenadas relativas a absolutas
                    left = int(x_rel * img_w)
                    top = int(y_rel * img_h)
                    right = int((x_rel + w_rel) * img_w)
                    bottom = int((y_rel + h_rel) * img_h)
                    
                    # Asegurar límites de la imagen
                    left = max(0, left)
                    top = max(0, top)
                    right = min(img_w, right)
                    bottom = min(img_h, bottom)
                    
                    # Recortar solo el bbox del animal
                    crop = img.crop((left, top, right, bottom))

                    # Guardar
                    base_name = os.path.basename(filename)
                    name_only, ext = os.path.splitext(base_name)
                    save_name = f"{name_only}_crop{i}.jpg" 
                    save_path = os.path.join(output_folder, save_name)

                    crop.save(save_path, quality=95)
                    crops_count += 1

        except Exception as e:
            print(f"Error en {filename}: {e}")
            errors += 1

    print("\n--- Resumen Final ---")
    print(f"Recortes generados: {crops_count}")
    print(f"Errores: {errors}")