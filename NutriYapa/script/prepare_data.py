import pandas as pd
from pathlib import Path

# Get the project root directory (parent of script directory)
PROJECT_ROOT = Path(__file__).parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def clean_easy_accessible_recipes():
    """Cargar recetas fáciles y accesibles"""
    try:
        if (RAW / "recetas_faciles_accesibles.csv").exists():
            df = pd.read_csv(RAW / "recetas_faciles_accesibles.csv")
            print(f"✅ Recetas fáciles y accesibles cargadas: {len(df)} recetas")
            return df
        else:
            print(f"⚠️ Archivo recetas_faciles_accesibles.csv no encontrado")
            return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error cargando recetas fáciles: {e}")
        return pd.DataFrame()

def clean_ecuadorian_recipes():
    """Procesar recetas ecuatorianas nativas"""
    try:
        # Intentar cargar el dataset expandido primero
        if (RAW / "recetas_ecuatorianas_expandido.csv").exists():
            df = pd.read_csv(RAW / "recetas_ecuatorianas_expandido.csv")
            print(f"✅ Recetas ecuatorianas expandidas cargadas: {len(df)} recetas")
        else:
            df = pd.read_csv(RAW / "recetas_ecuatorianas.csv")
            print(f"✅ Recetas ecuatorianas cargadas: {len(df)} recetas")
        
        # Verificar que tenga las columnas necesarias
        required_cols = ['name', 'calories', 'protein', 'ingredients']
        if all(col in df.columns for col in required_cols):
            df.to_csv(PROC / "recetas_ecuatorianas.csv", index=False)
            print(f"✅ Recetas ecuatorianas procesadas: {len(df)} recetas")
            
            # Mostrar estadísticas
            if 'precio_aprox' in df.columns:
                print(f"\n   📊 Por precio:")
                for precio, count in df['precio_aprox'].value_counts().items():
                    print(f"      {precio}: {count} recetas")
            
            if 'meal_type' in df.columns:
                print(f"\n   🍽️ Por tipo de comida:")
                for meal, count in df['meal_type'].value_counts().head(5).items():
                    print(f"      {meal}: {count} recetas")
            
            return df
        else:
            print(f"⚠️ Faltan columnas requeridas en recetas ecuatorianas")
            return pd.DataFrame()
    except FileNotFoundError:
        print(f"⚠️ Archivo recetas_ecuatorianas.csv no encontrado en {RAW}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error procesando recetas ecuatorianas: {e}")
        return pd.DataFrame()

def clean_ecuadorian_products():
    """Procesar productos ecuatorianos"""
    try:
        df = pd.read_csv(RAW / "productos_ecuatorianos.csv")
        print(f"✅ Productos ecuatorianos cargados: {len(df)} productos")
        
        df.to_csv(PROC / "productos_ecuatorianos.csv", index=False)
        print(f"✅ Productos ecuatorianos procesados: {len(df)} productos")
        return df
    except FileNotFoundError:
        print(f"⚠️ Archivo productos_ecuatorianos.csv no encontrado en {RAW}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ Error procesando productos ecuatorianos: {e}")
        return pd.DataFrame()

def clean_recipes():
    """Ya no procesamos recetas en inglés - solo usamos recetas ecuatorianas en español"""
    print("⚠️ Recetas en inglés ignoradas - usando solo recetas ecuatorianas en español")
    return pd.DataFrame()

def clean_products():
    """Ya no procesamos productos en inglés - solo usamos productos ecuatorianos en español"""
    print("⚠️ Productos en inglés ignorados - usando solo productos ecuatorianos en español")
    return pd.DataFrame()

def combine_recipes():
    """Combinar recetas ecuatorianas tradicionales con recetas fáciles y accesibles"""
    ecuadorian = pd.read_csv(PROC / "recetas_ecuatorianas.csv") if (PROC / "recetas_ecuatorianas.csv").exists() else pd.DataFrame()
    
    # Cargar recetas fáciles y accesibles
    easy_recipes = clean_easy_accessible_recipes()
    
    if len(ecuadorian) > 0 and len(easy_recipes) > 0:
        # Combinar ambos datasets
        combined = pd.concat([ecuadorian, easy_recipes], ignore_index=True)
        
        # Eliminar duplicados por nombre
        combined = combined.drop_duplicates(subset=['name'], keep='first')
        
        combined.to_csv(PROC / "recipes.csv", index=False)
        print(f"\n✅ Recetas combinadas: {len(ecuadorian)} tradicionales + {len(easy_recipes)} fáciles = {len(combined)} total (sin duplicados)")
        
        # Mostrar distribución por meal_type
        if 'meal_type' in combined.columns:
            print(f"\n   🍽️ Distribución por tipo de comida:")
            for meal, count in combined['meal_type'].value_counts().items():
                print(f"      {meal}: {count} recetas")
        
        return combined
    elif len(ecuadorian) > 0:
        ecuadorian.to_csv(PROC / "recipes.csv", index=False)
        print(f"✅ Usando SOLO recetas ecuatorianas tradicionales: {len(ecuadorian)} recetas")
        return ecuadorian
    elif len(easy_recipes) > 0:
        easy_recipes.to_csv(PROC / "recipes.csv", index=False)
        print(f"✅ Usando SOLO recetas fáciles: {len(easy_recipes)} recetas")
        return easy_recipes
    else:
        print("❌ No hay recetas disponibles")
        return pd.DataFrame()

if __name__ == "__main__":
    print("=" * 60)
    print("🥗 PREPARACIÓN DE DATOS NUTRIYAPA")
    print("=" * 60)
    
    # Procesar recetas ecuatorianas (prioritario)
    print("\n📍 Procesando recetas ecuatorianas...")
    clean_ecuadorian_recipes()
    
    # Procesar productos ecuatorianos
    print("\n🛒 Procesando productos ecuatorianos...")
    clean_ecuadorian_products()
    
    # Procesar recetas generales (opcional)
    print("\n🌎 Procesando recetas generales (si existen)...")
    clean_recipes()
    
    # Procesar productos generales (opcional)
    print("\n🌎 Procesando productos generales (si existen)...")
    clean_products()
    
    # Combinar datasets de recetas
    print("\n🔄 Combinando datasets...")
    final_recipes = combine_recipes()
    
    print("\n" + "=" * 60)
    print("✅ DATASETS PREPARADOS")
    print("=" * 60)
    print(f"📁 Ubicación: {PROC}")
    
    if len(final_recipes) > 0:
        print(f"\n📊 Resumen de recetas:")
        print(f"   Total de recetas: {len(final_recipes)}")
        if 'region' in final_recipes.columns:
            print(f"\n   Por región:")
            for region, count in final_recipes['region'].value_counts().items():
                print(f"      {region}: {count}")
    
    print("\n🎉 ¡Preparación completada!")
    print("=" * 60)
