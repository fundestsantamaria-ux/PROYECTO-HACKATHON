import os
import time
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

def crop_single_image(src_path, dst_path, pixels_to_cut, quality):
    """
    Procesa una sola imagen usando rutas completas.
    """
    try:
        with Image.open(src_path) as img:
            width, height = img.size

            if height <= pixels_to_cut:
                return f"SALTADA: {src_path} ({height}px es muy pequeña)"

            crop_area = (0, 0, width, height - pixels_to_cut)
            img_cropped = img.crop(crop_area)

            # Crear carpeta donde irá la imagen
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            img_cropped.save(dst_path, quality=quality)
            return None

    except Exception as e:
        return f"ERROR en {src_path}: {e}"


def remove_footer(input_folder, output_folder, pixels_to_cut=400, extensions=None, quality=95):
    """
    Recorta el footer de TODAS las imágenes dentro del input_folder (recursivo).
    """
    if extensions is None:
        extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

    print(f"Escaneando {input_folder} recursivamente...")

    # --- BUSCAR ARCHIVOS RECURSIVAMENTE ---
    files = []
    for root, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith(extensions):
                full_src = os.path.join(root, filename)

                # ruta relativa para mantener estructura de carpetas si quieres
                rel_path = os.path.relpath(full_src, input_folder)
                full_dst = os.path.join(output_folder, rel_path)

                files.append((full_src, full_dst))

    total = len(files)

    if total == 0:
        print("No se encontraron imágenes.")
        return

    # --- DETECCIÓN DE CPUS ---
    try:
        max_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        max_workers = os.cpu_count() or 4

    print(f"--- Iniciando proceso paralelo ---")
    print(f"Imágenes encontradas: {total}")
    print(f"Workers (CPUs): {max_workers}")
    print(f"Recortando {pixels_to_cut}px inferiores...")

    start_time = time.time()

    # --- PROCESAMIENTO PARALELO ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(crop_single_image, src, dst, pixels_to_cut, quality): src
            for src, dst in files
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()

            if result:
                print(result)

            if completed % 100 == 0 or completed == total:
                print(f"Procesando {completed}/{total} ({(completed/total)*100:.1f}%)")

    # --- TIEMPOS ---
    duration = time.time() - start_time

    print("\n--- ¡Proceso Terminado! ---")
    print(f"Tiempo total: {duration:.2f} segundos")
    print(f"Velocidad: {total/duration:.1f} imágenes/segundo")
    print(f"Guardado en: {os.path.abspath(output_folder)}")