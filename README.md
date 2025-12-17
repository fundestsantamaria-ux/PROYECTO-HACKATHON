# EyeClose Detection 👁️💤  
**Sistema de Detección de Somnolencia en Tiempo Real**

Proyecto Final – Módulo de Inteligencia Artificial  
Programa **Samsung Innovation Campus (SIC) 2025**

---

## 📌 Descripción General

**EyeClose Detection** es un sistema inteligente de visión por computadora diseñado para detectar somnolencia en tiempo real mediante el análisis del cierre prolongado de los ojos y la inclinación de la cabeza.  
El sistema utiliza únicamente una **webcam convencional**, sin necesidad de hardware especializado, lo que lo convierte en una solución **accesible, económica y eficiente**.

El proyecto está orientado a la prevención de accidentes en actividades que requieren atención constante, como la conducción de vehículos, el estudio prolongado o el trabajo nocturno.

---

## 🎯 Objetivos

### Objetivo General
Desarrollar un sistema inteligente capaz de detectar somnolencia mediante el análisis del cierre de ojos en tiempo real, generando alertas visuales y sonoras para prevenir accidentes.

### Objetivos Específicos
- Implementar el cálculo del **Eye Aspect Ratio (EAR)** para medir la apertura ocular.
- Detectar cierres prolongados de los párpados usando **MediaPipe Face Mesh**.
- Incorporar detección de inclinación de cabeza como indicador adicional de somnolencia.
- Generar alertas visuales y sonoras en tiempo real.
- Guardar evidencias del evento detectado (imagen y reporte).
- Aplicar técnicas de **Python e Inteligencia Artificial** aprendidas en el SIC 2025.

---

## 🧠 ¿Cómo funciona el sistema?

1. Captura video en tiempo real desde la webcam.
2. Detecta el rostro y genera una malla facial de **468 puntos** usando MediaPipe.
3. Calcula el **EAR (Eye Aspect Ratio)** a partir de puntos específicos de los ojos.
4. Realiza una **fase de calibración inicial** para aprender el patrón normal del usuario.
5. Utiliza un modelo de **One-Class SVM** para detectar anomalías (ojos cerrados).
6. Analiza la inclinación de la cabeza (pitch) como señal adicional de somnolencia.
7. Si la somnolencia se mantiene durante un tiempo definido:
   - Muestra alertas visuales en pantalla
   - Emite una alarma sonora
   - Guarda una imagen y un reporte del evento

---

## 🛠️ Tecnologías y Librerías Utilizadas

- **Python 3**
- **OpenCV** – Captura y procesamiento de video
- **MediaPipe Face Mesh** – Detección facial y landmarks
- **NumPy** – Procesamiento numérico
- **SciPy** – Cálculo de distancias (EAR)
- **Scikit-learn** – Modelo One-Class SVM
- **winsound** – Alarma sonora
- **requests** – Obtención de ubicación aproximada por IP

---

## ⚙️ Requisitos

- Webcam funcional
- Iluminación adecuada
- Sistema operativo Windows (para `winsound`)
- Python 3.9 o superior

---
