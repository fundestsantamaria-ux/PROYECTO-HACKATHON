import pandas as pd
from pathlib import Path

# Get the project root directory (parent of script directory)
PROJECT_ROOT = Path(__file__).parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
PROC = PROJECT_ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

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
    """Procesar recetas generales (si existen)"""
    try:
        df = pd.read_csv(RAW / "recipes.csv")
        print(f"Recipes loaded: {len(df)} rows")
        # Map actual column names to desired names
        column_mapping = {
            'Name': 'name',
            'RecipeIngredientParts': 'ingredients',
            'Calories': 'calories',
            'ProteinContent': 'protein',
            'FatContent': 'fat',
            'CarbohydrateContent': 'carbs',
            'Description': 'description',
            'RecipeInstructions': 'instructions'
        }
        # Select and rename columns that exist
        cols_to_keep = [col for col in column_mapping.keys() if col in df.columns]
        df = df[cols_to_keep]
        df = df.rename(columns=column_mapping)
        # Drop rows with missing critical data
        df = df.dropna(subset=['name', 'calories'], how='any')
        df.to_csv(PROC / "recipes.csv", index=False)
        print(f"Recipes processed: {len(df)} rows with columns {df.columns.tolist()}")
        return df
    except FileNotFoundError:
        print(f"⚠️ Archivo recipes.csv no encontrado - usando solo recetas ecuatorianas")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing recipes: {e}")
        return pd.DataFrame()

def clean_products():
    """Procesar productos generales (si existen)"""
    try:
        # Read with error handling for malformed lines
        df = pd.read_csv(RAW / "en.openfoodfacts.org.products.csv", 
                        sep='\t', 
                        low_memory=False, 
                        on_bad_lines='skip',  # Skip malformed lines
                        encoding='utf-8',
                        nrows=100000)  # Limit to first 100k rows for performance
        # Select columns that exist
        cols_to_keep = []
        for col in ['product_name', 'ingredients_text', 'nutriscore_grade', 
                    'energy-kcal_100g', 'proteins_100g', 'carbohydrates_100g', 'fat_100g']:
            if col in df.columns:
                cols_to_keep.append(col)
        if cols_to_keep:
            df = df[cols_to_keep]
        df.to_csv(PROC / "products.csv", index=False)
        print(f"Products processed: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"⚠️ Archivo products.csv no encontrado - usando solo productos ecuatorianos")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing products: {e}")
        print("Creating empty products.csv as fallback")
        return pd.DataFrame()

def combine_recipes():
    """Combinar recetas ecuatorianas con recetas generales"""
    ecuadorian = pd.read_csv(PROC / "recetas_ecuatorianas.csv") if (PROC / "recetas_ecuatorianas.csv").exists() else pd.DataFrame()
    general = pd.read_csv(PROC / "recipes.csv") if (PROC / "recipes.csv").exists() else pd.DataFrame()
    
    if len(ecuadorian) > 0:
        if len(general) > 0:
            # Combinar ambos datasets
            combined = pd.concat([ecuadorian, general], ignore_index=True)
            print(f"📊 Recetas combinadas: {len(ecuadorian)} ecuatorianas + {len(general)} generales = {len(combined)} total")
        else:
            combined = ecuadorian
            print(f"📊 Usando solo recetas ecuatorianas: {len(combined)} total")
        
        combined.to_csv(PROC / "recipes.csv", index=False)
        return combined
    elif len(general) > 0:
        print(f"📊 Usando solo recetas generales: {len(general)} total")
        return general
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
