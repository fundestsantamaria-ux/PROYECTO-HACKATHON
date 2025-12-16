import json
import os
import shutil

import os
import shutil

def normalize_path(path: str) -> str:
    """
    Limpia rutas con:
    - espacios al inicio/final
    - saltos de línea
    - tabulaciones
    - comillas sobrantes
    - BOM UTF-8
    """
    if not isinstance(path, str):
        path = str(path)

    # Remover caracteres invisibles
    path = path.replace("\x00", "")

    # Remover BOM
    path = path.replace("\ufeff", "")

    # Remover saltos de línea y tabs
    path = path.strip(" \n\r\t\"'")

    return path

def safe_copy(src_path, dst_path):

    # Normalizar rutas
    src_path = normalize_path(src_path)
    dst_path = normalize_path(dst_path)

    # Asegurar carpeta destino
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    # Verificar existencia real
    if not os.path.exists(src_path):
        return f"ERROR: Ruta origen no existe (limpia): {src_path}"

    try:
        shutil.copy2(src_path, dst_path)
        return None
    except Exception as e:
        return f"ERROR copiando '{src_path}' → '{dst_path}': {e}"


def divide_images(
    json_file,
    source_folder,
    dest_root,
    animals_folder_name,
    empty_folder_name,
    conf_threshold=0.4,
    accepted_categories=['1']
):
    """
    Clasifica imágenes en dos carpetas: Animales y Vacías,
    usando los resultados del JSON de MegaDetector.

    Parámetros:
    - json_file: archivo JSON con detecciones.
    - source_folder: carpeta donde están las imágenes originales.
    - dest_root: carpeta donde se crearán las subcarpetas 'Animales' y 'Vacias'.
    - conf_threshold: umbral mínimo para considerar detección válida.
    - accepted_categories: categorías aceptadas (por defecto ['1'] = animales).
    """

    folder_true = os.path.join(dest_root, animals_folder_name)
    folder_false = os.path.join(dest_root, empty_folder_name)

    os.makedirs(folder_true, exist_ok=True)
    os.makedirs(folder_false, exist_ok=True)

    print(f"Cargando {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    count_moved_true = 0
    count_moved_false = 0
    errors = 0

    print("--- Comenzando la clasificación ---")

    for entry in data['images']:
        filename = entry['file']
        detections = entry['detections']

        src_path = os.path.join(source_folder, filename)

        # Lógica de clasificación
        is_true = False
        max_conf = 0.0

        for det in detections:
            cat = det['category']
            conf = det['conf']

            if cat in accepted_categories and conf >= conf_threshold:
                is_true = True
                max_conf = max(max_conf, conf)

        # Destino según la clasificación
        if is_true:
            dst_path = os.path.join(folder_true, filename)
        else:
            dst_path = os.path.join(folder_false, filename)

        # Copiar archivo
        try:
            if os.path.exists(src_path):
                safe_copy(src_path, dst_path)
                print(f"Movido: {filename} → {'Animales' if is_true else 'Vacias'} (Conf: {max_conf:.2f})")

                if is_true:
                    count_moved_true += 11
                else:
                    count_moved_false += 1
            else:
                print(f"ADVERTENCIA: No se encontró el archivo {src_path}")
                errors += 1

        except Exception as e:
            print(f"ERROR moviendo {filename} desde {src_path}: {e}")
            errors += 1

    print("\n--- Resumen ---")
    print(f"Total procesado: {len(data['images'])}")
    print(f"Animales (True): {count_moved_true} -> {folder_true}")
    print(f"Vacías/Otros (False): {count_moved_false} -> {folder_false}")

    if errors > 0:
        print(f"Errores: {errors}")


# Ejemplo de uso:
# clasificar_imagenes(
#     json_file='resultados_megadetector.json',
#     source_folder='./images',
#     dest_root='./images_sorted'
# )
