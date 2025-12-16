import os
import sys
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from megadetector.detection.run_detector_batch import load_and_run_detector_batch, write_results_to_file
from megadetector.utils import path_utils

# --- FUNCIÓN WORKER (Ejecutada por cada núcleo del CPU) ---
def resize_single_image(args):
    """
    Función auxiliar para procesar una sola imagen en un proceso separado.
    Recibe una tupla: (input_path, temp_folder, index, max_side)
    """
    input_path, temp_folder, idx, max_side = args
    
    filename = os.path.basename(input_path)
    # Mantenemos tu nomenclatura para evitar duplicados
    output_path = os.path.join(temp_folder, filename)
    
    try:
        with Image.open(input_path) as img:
            w, h = img.size
            max_original = max(w, h)

            if max_original <= max_side:
                # Si es pequeña, solo guardamos (o copiamos)
                img.save(output_path)
            else:
                scale = max_side / max_original
                new_w = int(w * scale)
                new_h = int(h * scale)

                # LANCZOS es pesado, aquí es donde la paralelización brilla
                img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                img.save(output_path, quality=90)
        
        return output_path

    except Exception as e:
        print(f"Error procesando {filename}: {e}")
        return None

# --- FUNCIÓN GESTORA PARALELA ---
def preprocess_resize_batch_parallel(image_paths, temp_folder, max_side=1600):
    """
    Hace resize a todas las imágenes usando TODOS los núcleos disponibles.
    """
    if os.path.exists(temp_folder):
        results = [os.path.join(temp_folder, r) for r in os.listdir(temp_folder)]
        return results

    os.makedirs(temp_folder)

    # Preparamos los argumentos para cada tarea: (ruta, carpeta, indice, tamaño)
    tasks = [
        (path, temp_folder, i, max_side) 
        for i, path in enumerate(image_paths)
    ]

    print(f"Iniciando resize paralelo con 4 núcleos...")

    # ProcessPoolExecutor se encarga de distribuir el trabajo
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Usamos tqdm para envolver el iterador y mostrar progreso real
        # chunksize=10 mejora el rendimiento enviando lotes de tareas en vez de 1 por 1
        results = list(tqdm(
            executor.map(resize_single_image, tasks, chunksize=10), 
            total=len(tasks), 
            desc="Redimensionando (Multiprocess)"
        ))

    # Filtramos los resultados None (errores)
    valid_paths = [r for r in results if r is not None]
    return valid_paths


def megadetector_classify(
    input_folder,
    output_file,
    model_version='MDV5A',
    conf_threshold=0.2,
    recursive=True
):
    """
    Ejecuta MegaDetector sobre una carpeta de imágenes, 
    haciendo resize previo a 1600 px en PARALELO.
    """

    print(f"--- Iniciando MegaDetector Pipeline ---")

    input_folder_abs = os.path.abspath(input_folder)
    output_file_abs = os.path.abspath(output_file)
    temp_folder = os.path.join(input_folder_abs, "_resized_for_md")

    print(f"Carpeta de imágenes: {input_folder_abs}")

    if not os.path.exists(input_folder_abs):
        print(f"ERROR: No encuentro la carpeta '{input_folder}'.")
        sys.exit(1)

    # --- PASO 1: Listar imágenes ---
    print("Buscando imágenes...")
    image_file_names = path_utils.find_images(input_folder_abs, recursive=recursive)

    if len(image_file_names) == 0:
        print("ERROR: No encontré ninguna imagen en la carpeta.")
        sys.exit(1)

    print(f"Se encontraron {len(image_file_names)} imágenes.")

    # --- PASO 2: Preprocesamiento: RESIZE PARALELO ---
    print(f"Redimensionando imágenes a máximo 1600 px en: {temp_folder}")
    
    # AQUÍ ESTÁ EL CAMBIO PRINCIPAL
    resized_images = preprocess_resize_batch_parallel(image_file_names, temp_folder, max_side=1600)

    if not resized_images:
        print("Error: No se pudieron procesar las imágenes.")
        sys.exit(1)

    # --- PASO 3: Ejecutar Detección ---
    print("Cargando modelo y ejecutando detección...")
    results = load_and_run_detector_batch(
        model_file=model_version,
        image_file_names=resized_images,
        checkpoint_path=None,
        confidence_threshold=conf_threshold,
        quiet=False 
    )

    # --- PASO 4: Guardar Resultados ---
    print(f"Guardando resultados en {output_file_abs}...")

    write_results_to_file(
        results,
        output_file_abs,
        relative_path_base=input_folder_abs,
        detector_file=model_version
    )

    print("--- ¡Proceso Terminado! ---")
