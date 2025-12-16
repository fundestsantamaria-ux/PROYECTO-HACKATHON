# Proyecto de Análisis de Seguridad Urbana con K-Means y Mapa Interactivo

## Planteamiento del problema

La seguridad urbana es un factor clave para la calidad de vida de los ciudadanos. Sin embargo, analizar grandes volúmenes de datos relacionados con iluminación, comercios, reportes de incidentes y flujo de personas puede resultar complejo sin herramientas adecuadas.  
Este proyecto propone el uso de técnicas de análisis de datos y Machine Learning para identificar patrones de seguridad en distintas zonas urbanas y facilitar su interpretación mediante visualización geográfica.

## Objetivos del proyecto

### Objetivo general
Analizar y clasificar zonas urbanas según sus características de seguridad utilizando el algoritmo K-Means y representar los resultados en un mapa interactivo.

### Objetivos específicos
- Limpiar y estandarizar un conjunto de datos reales de seguridad urbana.
- Convertir variables cualitativas y cuantitativas a formato numérico.
- Agrupar zonas con características similares mediante Machine Learning no supervisado.
- Visualizar los clústeres obtenidos en un mapa interactivo.
- Facilitar la interpretación visual del nivel de seguridad de cada zona.

## Descripción general del proyecto

Este proyecto permite analizar datos de seguridad urbana, agrupar zonas según sus características usando el algoritmo K-Means y visualizar los resultados en un mapa interactivo generado con la librería Folium.

El sistema toma datos recopilados en campo (nivel de iluminación, cantidad de comercios, número de reportes, flujo de personas, distancia a puntos policiales, entre otros) y genera un mapa con círculos de colores, donde cada color representa un clúster con un nivel similar de seguridad.

## Características principales

✔ Carga de un archivo Excel con datos reales  
✔ Limpieza y estandarización del dataset  
✔ Conversión de datos a formato numérico  
✔ Clasificación automática de zonas usando K-Means  
✔ Generación de un mapa HTML interactivo  
✔ Uso de círculos de colores y tamaño ampliado para mejor visualización  
✔ Código desarrollado en Python, fácil de modificar o ampliar  

## ¿El proyecto utiliza Inteligencia Artificial?

Sí. El proyecto emplea un algoritmo de Machine Learning no supervisado llamado **K-Means**, el cual permite agrupar zonas urbanas según similitudes en variables como:

- Nivel de iluminación  
- Cantidad de comercios  
- Número de reportes de incidentes  
- Flujo de personas  
- Distancia al punto policial más cercano  

El sistema no predice eventos futuros, sino que **identifica patrones y clasifica zonas** de acuerdo con su nivel de similitud o riesgo relativo.

## Estructura del proyecto

Proyecto-Seguridad/
│
├── SEGURIDAD.xlsx # Base de datos original
├── mapa_zonas.html # Mapa interactivo generado
├── main.py # Código principal del análisis
└── README.md # Documentación del proyecto


## Tecnologías y herramientas utilizadas

- **Python**: lenguaje principal del proyecto  
- **Pandas**: carga, limpieza y procesamiento de datos  
- **Scikit-learn**: implementación del algoritmo K-Means  
- **Folium**: creación de mapas interactivos  
- **MarkerCluster**: agrupación visual de marcadores en el mapa  

## Interpretación de colores del mapa

| Clúster | Color      | Interpretación aproximada                  |
|--------:|------------|--------------------------------------------|
| 0       | 🟢 Verde   | Zonas con mejores indicadores de seguridad |
| 1       | 🟠 Naranja | Zonas intermedias o mixtas                  |
| 2       | 🔴 Rojo    | Zonas con mayor riesgo relativo            |

## Resultado final del proyecto

Como resultado, se genera el archivo **`mapa_zonas.html`**, el cual presenta:

- Visualización geográfica interactiva
- Círculos de gran tamaño para mejorar la visibilidad
- Clasificación por colores según el clúster asignado
- Información detallada de cada punto mediante ventanas emergentes (popups)
- Navegación intuitiva similar a Google Maps (zoom, desplazamiento)

Este enfoque permite analizar de manera visual y comprensible la distribución de la seguridad urbana en distintas zonas.


## Avance para el Hackathon

Era necesario adaptarlo para que funcione en tiempo real. Para lograr esto, estuve investigando y una opción viable es utilizar Google Forms, ya que permite publicar los resultados en una hoja de cálculo de Excel en línea que se actualiza automáticamente conforme se reciben nuevas respuestas.

Adicionalmente, necesitamos crear una página web. Dado que no contamos con conocimientos avanzados en desarrollo web, podríamos guiarnos en la creación de una página sencilla, enfocada únicamente en mostrar la información necesaria.

Por otro lado, se identificaron dificultades con el separador de las coordenadas, específicamente con el uso del punto y coma. Para solucionar este inconveniente, se puede automatizar el código de modo que detecte el punto y coma y lo convierta automáticamente al formato requerido.