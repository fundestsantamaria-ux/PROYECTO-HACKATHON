import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------
# 1. PAD TO SQUARE
# ---------------------------------------------------------
def pad_to_square(img, pad_value=0):
    h, w = img.shape[:2]
    size = max(h, w)
    padded = np.full((size, size, 3), pad_value, dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return padded

# ---------------------------------------------------------
# 2. GRAY WORLD NORMALIZATION (normalización de color suave)
# ---------------------------------------------------------
def gray_world_normalization(img):
    img = img.astype(np.float32)
    avgB = np.mean(img[:, :, 0])
    avgG = np.mean(img[:, :, 1])
    avgR = np.mean(img[:, :, 2])

    avg_gray = (avgB + avgG + avgR) / 3

    img[:, :, 0] = img[:, :, 0] * (avg_gray / avgB)
    img[:, :, 1] = img[:, :, 1] * (avg_gray / avgG)
    img[:, :, 2] = img[:, :, 2] * (avg_gray / avgR)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# ---------------------------------------------------------
# 3. CLAHE en canal L (Lab)
# ---------------------------------------------------------
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L = clahe.apply(L)

    lab = cv2.merge((L, A, B))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# ---------------------------------------------------------
# 4. Bilateral filter opcional
# ---------------------------------------------------------
def bilateral_smooth(img, enable=True):
    if not enable:
        return img
    return cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

# ---------------------------------------------------------
# 5. PIPELINE COMPLETO PARA UNA IMAGEN
# ---------------------------------------------------------
def preprocess_image(img, resize_size=224, use_bilateral=True):
    # pad → cuadrada
    img = pad_to_square(img, pad_value=0)

    # resize → tamaño uniforme para ResNet
    img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)

    # normalizar color (Gray World)
    img = gray_world_normalization(img)

    # CLAHE
    img = apply_clahe(img)

    # bilateral smoothing (leve)
    img = bilateral_smooth(img, enable=use_bilateral)

    return img

# ---------------------------------------------------------
# 6. PROCESAR TODA UNA CARPETA
# ---------------------------------------------------------
def chalhe_images(input_dir, output_dir, resize_size=224, use_bilateral=True):
    os.makedirs(output_dir, exist_ok=True)

    images = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(root, f))

    print(f"Procesando {len(images)} imágenes con CLAHE + Normalize + Pad + Resize...")

    for img_path in tqdm(images):
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠ No pude leer: {img_path}")
            continue

        processed = preprocess_image(
            img,
            resize_size=resize_size,
            use_bilateral=use_bilateral
        )

        # Mantener misma estructura de carpetas
        rel_path = os.path.relpath(img_path, input_dir)
        save_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cv2.imwrite(save_path, processed)
