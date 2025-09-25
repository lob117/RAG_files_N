import os
import glob
from pathlib import Path

def buscar_pdfs(ruta_principal):
    """
    Busca todos los archivos PDF en una ruta y sus subcarpetas.
    
    Args:
        ruta_principal (str): Ruta del directorio principal a buscar
        
    Returns:
        list: Lista de tuplas con (nombre_archivo, ruta_completa)
    """
    pdfs_encontrados = []
    
    # Verificar si la ruta existe
    if not os.path.exists(ruta_principal):
        print(f"Error: La ruta '{ruta_principal}' no existe.")
        return pdfs_encontrados
    
    # Método 1: Usando os.walk() - Recorre recursivamente todas las carpetas
    print("🔍 Buscando archivos PDF...")
    print("-" * 50)
    
    for root, dirs, files in os.walk(ruta_principal):
        for file in files:
            if file.lower().endswith('.pdf'):
                ruta_completa = os.path.join(root, file)
                pdfs_encontrados.append((file, ruta_completa))
    
    return pdfs_encontrados

def buscar_pdfs_pathlib(ruta_principal):
    """
    Alternativa usando pathlib (más moderno).
    
    Args:
        ruta_principal (str): Ruta del directorio principal a buscar
        
    Returns:
        list: Lista de tuplas con (nombre_archivo, ruta_completa)
    """
    pdfs_encontrados = []
    
    try:
        path = Path(ruta_principal)
        if not path.exists():
            print(f"Error: La ruta '{ruta_principal}' no existe.")
            return pdfs_encontrados
        
        # Buscar recursivamente todos los archivos .pdf
        for pdf_file in path.rglob("*.pdf"):
            pdfs_encontrados.append((pdf_file.name, str(pdf_file)))
            
    except Exception as e:
        print(f"Error al buscar archivos: {e}")
    
    return pdfs_encontrados

def mostrar_resultados(pdfs_encontrados):
    """
    Muestra los resultados de forma organizada.
    
    Args:
        pdfs_encontrados (list): Lista de tuplas con PDFs encontrados
    """
    if not pdfs_encontrados:
        print("❌ No se encontraron archivos PDF.")
        return
    
    print(f"✅ Se encontraron {len(pdfs_encontrados)} archivos PDF:")
    print("=" * 80)
    
    for i, (nombre, ruta) in enumerate(pdfs_encontrados, 1):
        print(f"{i:3d}. {nombre}")
        print(f"     📁 {ruta}")
        print("-" * 80)

def guardar_resultados_txt(pdfs_encontrados, ruta_base, archivo_salida="pdfs_encontrados.txt"):
    """
    Guarda los resultados en un archivo de texto.
    
    Args:
        pdfs_encontrados (list): Lista de PDFs encontrados
        ruta_base (str): Ruta base para calcular rutas relativas
        archivo_salida (str): Nombre del archivo donde guardar los resultados
    """
    try:
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            f.write(f"ARCHIVOS PDF ENCONTRADOS - Total: {len(pdfs_encontrados)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, (nombre, ruta_completa) in enumerate(pdfs_encontrados, 1):
                # Calcular ruta relativa desde la carpeta base
                ruta_relativa = os.path.relpath(ruta_completa, ruta_base)
                # Si el archivo está en la carpeta raíz, solo mostrar el nombre
                if ruta_relativa == nombre:
                    ruta_mostrar = nombre
                else:
                    ruta_mostrar = ruta_relativa
                
                f.write(f"{i}. {nombre}\n")
                f.write(f"   Ruta: {ruta_mostrar}\n\n")
        
        ruta_completa_archivo = os.path.abspath(archivo_salida)
        print(f"💾 Resultados guardados en: {ruta_completa_archivo}")
        
    except Exception as e:
        print(f"Error al guardar archivo: {e}")

def main():
    """Función principal"""
    print("🔎 BUSCADOR DE ARCHIVOS PDF")
    print("=" * 40)
    
    # Solicitar la ruta al usuario
    ruta = input("Ingresa la ruta donde buscar PDFs: ").strip()
    
    # Eliminar comillas si las hay
    if ruta.startswith('"') and ruta.endswith('"'):
        ruta = ruta[1:-1]
    
    print(f"\n📂 Buscando en: {ruta}")
    
    # Buscar PDFs usando el método tradicional
    pdfs = buscar_pdfs(ruta)
    
    # Mostrar resultados
    mostrar_resultados(pdfs)
    
    # Preguntar si quiere guardar los resultados
    if pdfs:
        guardar = input("\n¿Quieres guardar los resultados en un archivo? (s/n): ").lower()
        if guardar in ['s', 'si', 'sí', 'y', 'yes']:
            nombre_archivo = input("Nombre del archivo (Enter para 'pdfs_encontrados.txt'): ").strip()
            if not nombre_archivo:
                nombre_archivo = "pdfs_encontrados.txt"
            if not nombre_archivo.endswith('.txt'):
                nombre_archivo += '.txt'
            
            guardar_resultados_txt(pdfs, ruta, nombre_archivo)

# Función alternativa para uso directo
def buscar_pdfs_en_ruta(ruta):
    """
    Función simplificada para usar directamente.
    
    Args:
        ruta (str): Ruta donde buscar
        
    Returns:
        list: Lista de rutas completas de PDFs encontrados
    """
    pdfs = []
    for root, dirs, files in os.walk(ruta):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdfs.append(os.path.join(root, file))
    return pdfs

if __name__ == "__main__":
    main()

# EJEMPLOS DE USO:
# 
# 1. Uso interactivo:
#    python script.py
#
# 2. Uso directo en código:
#    pdfs = buscar_pdfs_en_ruta("C:/mi_carpeta")
#    for pdf in pdfs:
#        print(pdf)
#
# 3. Con pathlib:
#    pdfs = buscar_pdfs_pathlib("C:/mi_carpeta")
#    mostrar_resultados(pdfs)