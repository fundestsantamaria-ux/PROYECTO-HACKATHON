#!/bin/bash

# Script para ejecutar el pipeline completo de predicción
# Combina prediction_pipeline.py e Inferencia.py en un solo paso

echo "=========================================="
echo "  Pipeline Completo de Clasificación"
echo "=========================================="
echo ""

# Directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paso 1: Ejecutar prediction_pipeline.py
echo "[1/2] Ejecutando prediction_pipeline.py..."
echo "-------------------------------------------"
python3 prediction_pipeline.py

# Verificar si el primer script terminó correctamente
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: prediction_pipeline.py falló"
    exit 1
fi

echo ""
echo "✅ prediction_pipeline.py completado exitosamente"
echo ""

# Paso 2: Ejecutar Inferencia.py
echo "[2/2] Ejecutando Inferencia.py..."
echo "-------------------------------------------"
python3 Inferencia.py

# Verificar si el segundo script terminó correctamente
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error: Inferencia.py falló"
    exit 1
fi

echo ""
echo "✅ Inferencia.py completado exitosamente"
echo ""
echo "=========================================="
echo "  Pipeline completo finalizado"
echo "=========================================="
