from PROCESSING.clahe import chalhe_images
from PROCESSING.make_crops import make_crops
from PROCESSING.megadetector_step import megadetector_classify
from PROCESSING.divide import divide_images
from PROCESSING.remove_footer import remove_footer
import os
import shutil

# --- CONFIGURACIÓN ---
# Usar  (ENSAMBLADO) como PATH base
PATH = os.getcwd()
SOURCE_IMAGES = "DATASET_PRUEBA"
SRC_IMAGES = os.path.join(PATH, SOURCE_IMAGES)  # Carpeta de imágenes originales
IMAGES = SRC_IMAGES  # Usar directamente las imágenes originales (sin recortar footer)
# Directorio raíz para resultados (todas las salidas irán dentro de `RESULTS`)
RESULTS_DIR = os.path.join(PATH, 'RESULTS')
# Carpetas intermedias dentro de RESULTS
SORTED_DIR = os.path.join(RESULTS_DIR, 'images_sorted')
CROPS_RAW_DIR = os.path.join(RESULTS_DIR, 'crops')
CROPS_CLAHE_DIR = os.path.join(RESULTS_DIR, 'crops_clahe')

# Asegurar que los directorios existen antes de ejecutar el pipeline
for _d in (RESULTS_DIR, SORTED_DIR, CROPS_RAW_DIR, CROPS_CLAHE_DIR):
    os.makedirs(_d, exist_ok=True)
# Definimos rutas claras

# Ruta del JSON de MegaDetector (guardar dentro de RESULTS)
MD_JSON = os.path.join(RESULTS_DIR, 'resultados_megadetector.json')

ANIMALS_FOLDER = 'Animales'
EMPTY_FOLDER = 'Vacias'

path_animales_img = os.path.join(SORTED_DIR, ANIMALS_FOLDER)

# --- PIPELINE ---

print("=== INICIANDO PIPELINE (MegaDetector -> Crop -> CLAHE  ===")

# 0. RECORTAR FOOTER (DESACTIVADO)
# remove_footer(
#     input_folder=SRC_IMAGES,
#     output_folder=IMAGES,
#     pixels_to_cut=400,
#     quality=95
# )

# 1. MEGADETECTOR
print("\n--- Paso 1: MegaDetector ---")
if os.path.exists(MD_JSON):
    print("JSON detectado. Saltando.")
else:
    megadetector_classify(
        input_folder=IMAGES,
        output_file=MD_JSON,
        model_version='MDV5A',
        conf_threshold=0.2,
        recursive=True
    )

# 2. DIVIDE
print("\n--- Paso 2: Dividir ---")
divide_images(
    json_file=MD_JSON,
    source_folder=IMAGES,
    dest_root=SORTED_DIR,
    animals_folder_name=ANIMALS_FOLDER,
    empty_folder_name=EMPTY_FOLDER,
    conf_threshold=0.4,
    accepted_categories=['1']
)

# shutil.rmtree(IMAGES)

# 3. MAKE CROPS
print("\n--- Paso 3: Recortes (Crops) ---")

make_crops(
    json_file=MD_JSON,
    input_folder=path_animales_img,
    output_folder=CROPS_RAW_DIR,
    conf_threshold=0.4,
    accepted_categories=['1']
)

shutil.rmtree(SORTED_DIR)

# 4. CLAHE
print("\n--- Paso 4: CLAHE ---")

chalhe_images(
    input_dir=CROPS_RAW_DIR,
    output_dir=CROPS_CLAHE_DIR
)

shutil.rmtree(CROPS_RAW_DIR)  # Elimina los recortes sin filtro CLAHE

# 5. LIMPIEZA FINAL
print("\n--- Paso 5: Limpieza ---")
resized_folder = os.path.join(SRC_IMAGES, '_resized_for_md')
if os.path.exists(resized_folder):
    shutil.rmtree(resized_folder)
    print(f"Carpeta eliminada: {resized_folder}")

print("\n=== PIPELINE COMPLETADO ===")
