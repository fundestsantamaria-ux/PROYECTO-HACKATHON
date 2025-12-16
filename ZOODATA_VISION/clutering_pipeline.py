from PROCESSING.auto_cluster_dl import auto_cluster_dl_hdbscan
from PROCESSING.clahe import chalhe_images
from PROCESSING.make_crops import make_crops
from PROCESSING.megadetector_step import megadetector_classify
from PROCESSING.divide import divide_images
from PROCESSING.remove_footer import remove_footer
import os
import shutil

# --- CONFIGURACIÓN ---
PATH = os.getcwd()
SRC_IMAGES = os.path.join(PATH, 'DATASET_COMPLETO_PRUEBA')  # Carpeta de imágenes originales
#IMAGES = os.path.join(PATH, 'images')
# Carpetas intermedias
SORTED_DIR = os.path.join(PATH, 'images_sorted')
CROPS_RAW_DIR = os.path.join(PATH, 'crops')
CROPS_CLAHE_DIR = os.path.join(PATH, 'crops_clahe_processed')
CLUSTERS_OUTPUT = os.path.join(PATH, 'CLUSTERS')       
# Definimos rutas claras

ANIMALS_FOLDER = 'Animales'
EMPTY_FOLDER = 'Vacias'

path_animales_img = os.path.join(SORTED_DIR, ANIMALS_FOLDER)

# --- PIPELINE ---

print("=== INICIANDO PIPELINE (MegaDetector -> Crop -> CLAHE -> deepfaune) ===")

# 0. RECORTAR FOOTER
# remove_footer(
#     input_folder=SRC_IMAGES,
#     output_folder=IMAGES,
#     pixels_to_cut=400,
#     quality=95
# )
# 1. MEGADETECTOR
print("\n--- Paso 1: MegaDetector ---")
if os.path.exists('resultados_megadetector.json'):
    print("JSON detectado. Saltando.")
else:
    megadetector_classify(
        input_folder=SRC_IMAGES,
        output_file='resultados_megadetector.json',
        model_version='MDV5A',
        conf_threshold=0.2,
        recursive=True
    )

# 2. DIVIDE
print("\n--- Paso 2: Dividir ---")
divide_images(
    json_file='resultados_megadetector.json',
    source_folder=SRC_IMAGES,
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
    json_file='resultados_megadetector.json',
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

shutil.rmtree(CROPS_RAW_DIR)

# 5. AUTO CLUSTER (RESNET)
print("\n--- Paso 5: Clustering con ResNet50 + UMAP ---")
# CAMBIO: Usamos la función que integra ResNet
auto_cluster_dl_hdbscan(
    input_dir=CROPS_CLAHE_DIR,    # Usamos los crops mejorados
    output_dir=CLUSTERS_OUTPUT,
    min_cluster_size=15
)


print(f"\n¡LISTO! Revisa la carpeta: {CLUSTERS_OUTPUT}")

# --- LIMPIEZA DE ARCHIVOS TEMPORALES ---
for temp_file in ["temp_features.npy", "temp_filenames.json", "resultados_megadetector.json"]:
    # Eliminar carpeta crops_clahe_processed si existe
    crops_clahe_dir = "crops_clahe_processed"
    if os.path.exists(crops_clahe_dir) and os.path.isdir(crops_clahe_dir):
        try:
            shutil.rmtree(crops_clahe_dir)
            print(f"Carpeta temporal eliminada: {crops_clahe_dir}")
        except Exception as e:
            print(f"No se pudo eliminar la carpeta {crops_clahe_dir}: {e}")

    # Eliminar carpeta DATASET_COMPLETO_PRUEBA/_resized_for_md si existe
    resized_dir = os.path.join("DATASET_COMPLETO_PRUEBA", "_resized_for_md")
    if os.path.exists(resized_dir) and os.path.isdir(resized_dir):
        try:
            shutil.rmtree(resized_dir)
            print(f"Carpeta temporal eliminada: {resized_dir}")
        except Exception as e:
            print(f"No se pudo eliminar la carpeta {resized_dir}: {e}")
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Archivo temporal eliminado: {temp_file}")
    except Exception as e:
        print(f"No se pudo eliminar {temp_file}: {e}")
