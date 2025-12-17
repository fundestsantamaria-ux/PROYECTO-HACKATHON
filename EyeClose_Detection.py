import cv2
import mediapipe as mp
import time
import numpy as np
import winsound
import os
import requests
from datetime import datetime
from scipy.spatial import distance as dist
from sklearn.svm import OneClassSVM

#CONFIGURACIÓN
TIEMPO_LIMITE = 3.0       # Segundos dormido antes de sonar y guardar datos
TIEMPO_CALIBRACION = 8.0  # Tiempo para aprender los ojos
ANGULO_CABEZA_ADELANTE = -10 
ANGULO_CABEZA_ATRAS = 20      

# Índices de MediaPipe
OJO_IZQ_IDX = [362, 385, 387, 263, 373, 380]
OJO_DER_IDX = [33, 160, 158, 133, 153, 144]

# Crear carpeta de evidencias si no existe
if not os.path.exists("evidencias"):
    os.makedirs("evidencias")

#FUNCIONES

def calcular_ear(landmarks, indices_ojo, w, h):
    coords = []
    for idx in indices_ojo:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    A = dist.euclidean(coords[1], coords[5])
    B = dist.euclidean(coords[2], coords[4])
    C = dist.euclidean(coords[0], coords[3])
    return (A + B) / (2.0 * C), coords

def obtener_rotacion_cabeza(img, landmarks):
    h, w, _ = img.shape
    face_3d = []
    face_2d = []
    indices_clave = [1, 152, 33, 263, 61, 291]

    for idx in indices_clave:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, w/2],
                           [0, focal_length, h/2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    return angles[0] * 360 # Pitch

def obtener_ubicacion():
    """Obtiene ubicación aproximada por IP. Timeout bajo para no congelar el video."""
    try:
        response = requests.get('http://ip-api.com/json/', timeout=1.5)
        data = response.json()
        if data['status'] == 'success':
            return f"{data.get('city')}, {data.get('country')} (Lat: {data.get('lat')}, Lon: {data.get('lon')})"
        return "Ubicacion no encontrada"
    except:
        return "Sin conexion a Internet"

#INICIO DEL PROGRAMA

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

# Variables de estado
datos_calibracion = []
modelo_ia = None
calibrado = False
inicio_calibracion = None
tiempo_inicio_sueno = None

# Variable de control para no guardar mil fotos por segundo
evidencia_guardada = False 

print("Iniciando SISTEMA DE DETECCION DE SOMNOLENCIA AI...")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    texto_estado = "Esperando..."
    color_texto = (255, 255, 255)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm_list = face_landmarks.landmark

            # 1. Métricas
            leftEAR, l_coords = calcular_ear(lm_list, OJO_IZQ_IDX, w, h)
            rightEAR, r_coords = calcular_ear(lm_list, OJO_DER_IDX, w, h)
            avgEAR = (leftEAR + rightEAR) / 2.0
            pitch = obtener_rotacion_cabeza(frame, lm_list)

            # 2. Lógica Principal
            if not calibrado:
                # --- FASE CALIBRACIÓN ---
                if inicio_calibracion is None:
                    inicio_calibracion = time.time()
                
                tiempo_transcurrido = time.time() - inicio_calibracion
                datos_calibracion.append([avgEAR])
                
                cv2.putText(frame, f"CALIBRANDO... {int(TIEMPO_CALIBRACION - tiempo_transcurrido)}s", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                if tiempo_transcurrido >= TIEMPO_CALIBRACION:
                    print("[INFO] Entrenando IA...")
                    X = np.array(datos_calibracion)
                    modelo_ia = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
                    modelo_ia.fit(X)
                    calibrado = True
                    winsound.Beep(1000, 200)

            else:
                # --- FASE MONITOREO ---
                prediccion = modelo_ia.predict([[avgEAR]])[0]
                ojo_cerrado = True if prediccion == -1 else False
                
                cabeza_mal = False
                tipo_cabeza = ""

                if pitch < ANGULO_CABEZA_ADELANTE:
                    cabeza_mal = True; tipo_cabeza = "ADELANTE"
                elif pitch > ANGULO_CABEZA_ATRAS:
                    cabeza_mal = True; tipo_cabeza = "ATRAS"

                detectado_sueno = cabeza_mal or ojo_cerrado
                
                if detectado_sueno:
                    texto_estado = f"ALERTA: {tipo_cabeza}" if cabeza_mal else "ALERTA: OJOS"
                    color_texto = (0, 0, 255)

                    if tiempo_inicio_sueno is None:
                        tiempo_inicio_sueno = time.time()
                    
                    duracion = time.time() - tiempo_inicio_sueno
                    cv2.putText(frame, f"PELIGRO: {duracion:.1f}s", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    #DETECCIÓN CRÍTICA
                    if duracion >= TIEMPO_LIMITE:
                        cv2.putText(frame, "DESPIERTA!", (150, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                        winsound.Beep(2500, 100)

                        # GUARDADO
                        if not evidencia_guardada:
                            print("[ALERTA] Guardando evidencia...")
                            
                            # 1. Preparar datos
                            ahora = datetime.now()
                            timestamp = ahora.strftime("%Y%m%d_%H%M%S")
                            nombre_base = f"evidencias/ALERTA_{timestamp}"
                            
                            # 2. Guardar Foto
                            cv2.imwrite(f"{nombre_base}.jpg", frame)
                            
                            # 3. Guardar Datos (TXT)
                            ubicacion = obtener_ubicacion() # Se llama solo una vez
                            
                            with open(f"{nombre_base}.txt", "w") as f:
                                f.write("REPORTE DE INCIDENTE - SISTEMA DE DETECCION DE SOMNOLENCIA AI\n")
                                f.write("=====================================\n")
                                f.write(f"Fecha: {ahora.strftime('%d/%m/%Y')}\n")
                                f.write(f"Hora: {ahora.strftime('%H:%M:%S')}\n")
                                f.write(f"Causa: {texto_estado}\n")
                                f.write(f"Ubicacion: {ubicacion}\n")
                                f.write(f"EAR (Ojos): {avgEAR:.3f}\n")
                                f.write(f"Angulo Cabeza: {int(pitch)}\n")
                            
                            print(f"[EXITO] Guardado en {nombre_base}")
                            evidencia_guardada = True # Bloquea el guardado hasta despertar

                else:
                    # Si despierta, resetea todo
                    texto_estado = "CONDUCCION SEGURA"
                    color_texto = (0, 255, 0)
                    tiempo_inicio_sueno = None
                    evidencia_guardada = False #Permite guardar el siguiente evento

                # Dibujar ojos visualmente
                for p in l_coords: cv2.circle(frame, p, 1, (0, 255, 0), -1)
                for p in r_coords: cv2.circle(frame, p, 1, (0, 255, 0), -1)

    cv2.putText(frame, texto_estado, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_texto, 2)
    cv2.imshow("DriverGuard AI - Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("LA CARPETA ESTÁ EN:", os.path.abspath("evidencias"))