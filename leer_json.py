import json

# Cargar el archivo JSON
with open(r"C:\Users\javie\Documents\poryecto_natural_2025-09\prueba_txt\TOMO_16.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Función recursiva para recorrer y mostrar todas las claves
def mostrar_claves(obj, nivel=0):
    if isinstance(obj, dict):
        for k, v in obj.items():
            print("  " * nivel + f"- {k}")
            mostrar_claves(v, nivel + 1)
    elif isinstance(obj, list):
        print("  " * nivel + f"[Lista con {len(obj)} elementos]")
        if len(obj) > 0:
            mostrar_claves(obj[0], nivel + 1)

# Mostrar claves principales
print("Campos encontrados en el JSON:\n")
mostrar_claves(data)
