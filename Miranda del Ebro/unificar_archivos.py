import pandas as pd
import numpy as np
import os

# =========================
# CONFIGURACIÓN
# =========================
carpeta = "datos"
archivo_salida = "dataset_unificado.csv"

# =========================
# UNIFICAR ARCHIVOS
# =========================
lista_dfs = []

for archivo in os.listdir(carpeta):
    if archivo.endswith(".xlsx") and archivo != "datos.xlsx":
        
        ruta = os.path.join(carpeta, archivo)
        print(f"📂 Procesando: {archivo}")
        
        df = pd.read_excel(ruta)

        # Normalizar columnas
        df.columns = df.columns.str.strip().str.lower()

        columnas_validas = ["fecha", "nivel (m)", "caudal (m³/s)", "lluvia", "desbordamiento"]
        df = df[columnas_validas]

        municipio = archivo.replace(".xlsx", "").strip()
        df["municipio"] = municipio

        lista_dfs.append(df)

df_final = pd.concat(lista_dfs, ignore_index=True)

# =========================
# FORMATEO
# =========================
df_final["fecha"] = pd.to_datetime(df_final["fecha"], dayfirst=True, errors="coerce")
df_final = df_final.sort_values(by=["municipio", "fecha"]).reset_index(drop=True)

df_final.to_csv(archivo_salida, index=False)

print("\n✅ Dataset unificado creado correctamente")

# =========================
# PREPARACIÓN PARA MODELO
# =========================
df_modelo = df_final.copy()

df_modelo.replace(["No hay datos", "", " "], np.nan, inplace=True)

columnas_numericas = ["nivel (m)", "caudal (m³/s)", "lluvia", "desbordamiento"]

for col in columnas_numericas:
    df_modelo[col] = pd.to_numeric(df_modelo[col], errors="coerce")

# =========================
# 🔥 MEDIR DATOS ORIGINALES
# =========================
mask_original = df_modelo[columnas_numericas].notna()

# =========================
# INTERPOLACIÓN ROBUSTA
# =========================
df_modelo = df_modelo.sort_values(by=["municipio", "fecha"])

df_modelo[columnas_numericas] = df_modelo.groupby("municipio")[columnas_numericas].transform(
    lambda x: x.interpolate(limit_direction="both").ffill().bfill()
)

# =========================
# 🔥 MEDIR DATOS INTERPOLADOS
# =========================
mask_despues = df_modelo[columnas_numericas].notna()

datos_inventados = (~mask_original) & (mask_despues)

total_inventados = datos_inventados.sum().sum()
total_datos = mask_despues.size
porcentaje = (total_inventados / total_datos) * 100

print("\n📊 DATOS INTERPOLADOS (GLOBAL):")
print(f"Total interpolados: {total_inventados}")
print(f"Total datos: {total_datos}")
print(f"Porcentaje interpolado: {porcentaje:.2f}%")

# =========================
# 🔥 POR VARIABLE
# =========================
print("\n📊 DATOS INTERPOLADOS POR VARIABLE:")

for col in columnas_numericas:
    datos_col = datos_inventados[col].sum()
    total_col = len(df_modelo)
    print(f"{col}: {datos_col} ({(datos_col/total_col)*100:.2f}%)")

# =========================
# 🔥 POR MUNICIPIO
# =========================
print("\n📊 DATOS INTERPOLADOS POR MUNICIPIO:")

df_temp = datos_inventados.copy()
df_temp["municipio"] = df_modelo["municipio"]

resumen = df_temp.groupby("municipio").sum()

print(resumen)

# =========================
# LIMPIEZA FINAL (MÍNIMA)
# =========================
df_modelo = df_modelo.dropna(subset=["nivel (m)", "caudal (m³/s)"])

# =========================
# RENOMBRAR COLUMNAS
# =========================
df_modelo.rename(columns={
    "nivel (m)": "nivel_m",
    "caudal (m³/s)": "caudal_m3s",
    "lluvia": "lluvia_mm"
}, inplace=True)

# =========================
# ONE-HOT MUNICIPIOS
# =========================
municipios = [
    "Arroyo", "Asco", "Castejon", "Gelsa", "Logroño",
    "Mendavia", "Miranda del Ebro", "Palazuelos",
    "Reinosa-Nestares", "Riosequillo", "Tortosa",
    "Tudela", "Villafranca de Ebro", "Zaragoza"
]

df_modelo = pd.get_dummies(df_modelo, columns=["municipio"])

# Asegurar todas las columnas
for m in municipios:
    col = f"municipio_{m}"
    if col not in df_modelo.columns:
        df_modelo[col] = 0

columnas_municipio = [f"municipio_{m}" for m in municipios]

df_modelo = df_modelo[
    ["fecha", "nivel_m", "caudal_m3s", "lluvia_mm", "desbordamiento"]
    + columnas_municipio
]

# =========================
# GUARDAR FINAL
# =========================
df_modelo.to_csv("dataset_modelo.csv", index=False)

print("\n🚀 Dataset listo para entrenamiento creado")

# =========================
# RESUMEN FINAL
# =========================
print("\n📊 RESUMEN FINAL:")
print(f"Filas originales: {len(df_final)}")
print(f"Filas modelo: {len(df_modelo)}")

print("\nMunicipios presentes:")
print(df_final["municipio"].value_counts())

print("\nMunicipios en modelo:")
print(df_modelo[columnas_municipio].sum())