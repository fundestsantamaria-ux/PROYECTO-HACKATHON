import streamlit as st

st.set_page_config(page_title="ZooData Vision", layout="wide")
# Mostrar logo centrado arriba
import base64
from PIL import Image
import io
import os



# Mostrar banner a la izquierda y logo a la derecha

# Mostrar banner estirado a lo largo y logo a la derecha
logo_path = "IMAGES/LOGO.jpeg"
banner_path = "IMAGES/banner-cloudforest.jpg"
if os.path.exists(logo_path) and os.path.exists(banner_path):
	import base64
	banner_b64 = base64.b64encode(open(banner_path, 'rb').read()).decode()
	logo_b64 = base64.b64encode(open(logo_path, 'rb').read()).decode()
	st.markdown(
		f"""
		<div style='display: flex; align-items: center; justify-content: space-between; width: 100%;'>
			<img src='data:image/jpeg;base64,{banner_b64}' style='width: 100%; max-height: 160px; object-fit: cover; border-radius: 8px; margin-right: 20px;'>
			<img src='data:image/jpeg;base64,{logo_b64}' style='height: 155px; border-radius: 8px;'>
		</div>""",
		unsafe_allow_html=True
	)
elif os.path.exists(logo_path):
	st.image(logo_path, width=180)

st.title("ZooData Vision")

st.markdown("""
Bienvenido a ZooData Vision. Esta aplicación permite procesar imágenes de fauna silvestre y visualizar los resultados del pipeline de clasificación y recorte de imágenes.
""")


# Input para la ruta de la carpeta de imágenes
input_folder = st.text_input(
	"Ruta de la carpeta con imágenes a procesar",
	value="DATASET_PRUEBA",
	help="Especifique la ruta de la carpeta que contiene las imágenes a procesar."
)

# Botón para ejecutar el pipeline
import subprocess
import os

if st.button("Procesar imágenes"):
	st.info("Procesando imágenes, esto puede tardar varios minutos...")
	# Ejecutar el pipeline con la carpeta seleccionada
	env = os.environ.copy()
	env["DATASET_PRUEBA"] = input_folder  # Para compatibilidad, aunque el pipeline usa la variable en el código
	# Modificar prediction_pipeline.py y/o Inferencia.py para aceptar la carpeta por variable de entorno si es necesario
	# Aquí, sobreescribimos el valor en prediction_pipeline.py antes de ejecutar
	with open("prediction_pipeline.py", "r") as f:
		code = f.read()
	import re
	code = re.sub(r'SOURCE_IMAGES = ".*"', f'SOURCE_IMAGES = "{input_folder}"', code)
	with open("prediction_pipeline_temp.py", "w") as f:
		f.write(code)
	# Ejecutar el pipeline modificado
	result1 = subprocess.run(["python3", "prediction_pipeline_temp.py"], capture_output=True, text=True)
	if result1.returncode != 0:
		st.error(f"Error en prediction_pipeline.py: {result1.stderr}")
	else:
		# Ejecutar Inferencia.py (modificando SOURCE_IMAGES si es necesario)
		with open("Inferencia.py", "r") as f:
			code2 = f.read()
		code2 = re.sub(r'SOURCE_IMAGES = ".*"', f'SOURCE_IMAGES = "{input_folder}"', code2)
		with open("Inferencia_temp.py", "w") as f:
			f.write(code2)
		result2 = subprocess.run(["python3", "Inferencia_temp.py"], capture_output=True, text=True)
		# Eliminar archivos temporales
		try:
			os.remove("prediction_pipeline_temp.py")
		except Exception:
			pass
		try:
			os.remove("Inferencia_temp.py")
		except Exception:
			pass
		if result2.returncode != 0:
			st.error(f"Error en Inferencia.py: {result2.stderr}")
		else:
			st.success("Imágenes procesadas correctamente.")

# Mostrar los primeros 5 registros del CSV si existe
import pandas as pd
import pathlib
csv_path = pathlib.Path("RESULTS/predicciones.csv")
if csv_path.exists():
	df = pd.read_csv(csv_path)
	st.subheader("Primeros 5 registros de predicciones:")
	st.dataframe(df.head(5))

	st.subheader("Visualización de imágenes originales y recortes (primeros 5 registros):")
	for idx, row in df.head(5).iterrows():
		st.markdown(f"**Registro {idx+1}:** Clase: {row['clase_predicha']} | Confianza: {row['confianza']:.2f}")
		cols = st.columns(2)
		# Imagen original
		orig_path = os.path.join(row['archivo_parent'])
		if os.path.exists(orig_path):
			cols[0].image(orig_path, caption="Imagen original", width=600)
		else:
			cols[0].warning(f"No se encontró la imagen original: {orig_path}")
		# Recorte
		crop_path = os.path.join(row['archivo'])
		if os.path.exists(crop_path):
			cols[1].image(crop_path, caption="Recorte", width=350)
		else:
			cols[1].warning(f"No se encontró el recorte: {crop_path}")

	st.info(f"Total de imágenes procesadas: {len(df)}")
