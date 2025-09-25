import os
import shutil
from pathlib import Path
try:
    from PyPDF2 import PdfReader, PdfWriter
except ImportError:
    print("❌ Necesitas instalar PyPDF2: pip install PyPDF2")
    exit(1)

def agregar_metadata_pdf(ruta_pdf_original, ruta_pdf_destino, ruta_relativa, metadata_adicional=None):
    """
    Crea una copia de un PDF y le agrega metadata con su ruta relativa.
    
    Args:
        ruta_pdf_original (str): Ruta del PDF original
        ruta_pdf_destino (str): Ruta donde guardar la copia con metadata
        ruta_relativa (str): Ruta relativa para agregar en metadata
        metadata_adicional (dict): Metadata adicional opcional
        
    Returns:
        bool: True si fue exitoso, False si hubo error
    """
    try:
        # Leer el PDF original
        reader = PdfReader(ruta_pdf_original)
        writer = PdfWriter()
        
        # Copiar todas las páginas
        for page in reader.pages:
            writer.add_page(page)
        
        # Obtener metadata existente
        metadata_existente = reader.metadata if reader.metadata else {}
        
        # Crear nueva metadata
        nueva_metadata = {
            '/Title': metadata_existente.get('/Title', Path(ruta_pdf_original).stem),
            '/Author': metadata_existente.get('/Author', 'PDF Organizer'),
            '/Subject': metadata_existente.get('/Subject', 'Documento procesado'),
            '/Creator': metadata_existente.get('/Creator', 'PDF Metadata Updater'),
            '/Producer': metadata_existente.get('/Producer', 'PyPDF2'),
            '/Keywords': f"ruta_original:{ruta_relativa}",
            '/Custom_Path': ruta_relativa,  # Campo personalizado
        }
        
        # Agregar metadata adicional si se proporciona
        if metadata_adicional:
            nueva_metadata.update(metadata_adicional)
        
        # Aplicar la nueva metadata
        writer.add_metadata(nueva_metadata)
        
        # Crear directorio de destino si no existe
        os.makedirs(os.path.dirname(ruta_pdf_destino), exist_ok=True)
        
        # Escribir el nuevo PDF
        with open(ruta_pdf_destino, 'wb') as archivo_salida:
            writer.write(archivo_salida)
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando {ruta_pdf_original}: {str(e)}")
        return False

def procesar_pdfs_con_metadata(ruta_base, carpeta_destino="pdfs_con_metadata"):
    """
    Procesa todos los PDFs en una ruta y crea copias con metadata de ubicación.
    
    Args:
        ruta_base (str): Ruta base donde buscar PDFs
        carpeta_destino (str): Carpeta donde guardar las copias
        
    Returns:
        dict: Estadísticas del procesamiento
    """
    
    # Buscar todos los PDFs
    print("🔍 Buscando archivos PDF...")
    pdfs_encontrados = []
    
    for root, dirs, files in os.walk(ruta_base):
        for file in files:
            if file.lower().endswith('.pdf'):
                ruta_completa = os.path.join(root, file)
                ruta_relativa = os.path.relpath(ruta_completa, ruta_base)
                pdfs_encontrados.append((file, ruta_completa, ruta_relativa))
    
    if not pdfs_encontrados:
        print("❌ No se encontraron archivos PDF.")
        return {"procesados": 0, "errores": 0, "total": 0}
    
    print(f"✅ Encontrados {len(pdfs_encontrados)} archivos PDF")
    
    # Crear carpeta de destino
    ruta_destino_completa = os.path.join(os.getcwd(), carpeta_destino)
    os.makedirs(ruta_destino_completa, exist_ok=True)
    
    # Contadores
    procesados_exitosos = 0
    errores = 0
    
    print(f"\n📝 Procesando PDFs y agregando metadata...")
    print("-" * 60)
    
    for i, (nombre_archivo, ruta_original, ruta_relativa) in enumerate(pdfs_encontrados, 1):
        print(f"[{i:3d}/{len(pdfs_encontrados)}] {nombre_archivo}")
        
        # Crear estructura de carpetas en destino manteniendo la jerarquía
        ruta_relativa_dir = os.path.dirname(ruta_relativa) if os.path.dirname(ruta_relativa) else ""
        carpeta_destino_archivo = os.path.join(ruta_destino_completa, ruta_relativa_dir)
        ruta_destino_archivo = os.path.join(carpeta_destino_archivo, nombre_archivo)
        
        # Procesar el PDF
        exito = agregar_metadata_pdf(
            ruta_original, 
            ruta_destino_archivo, 
            ruta_relativa.replace('\\', '/'),  # Normalizar separadores
            metadata_adicional={
                '/CreationDate': f"D:{__import__('datetime').datetime.now().strftime('%Y%m%d%H%M%S')}",
                '/ModDate': f"D:{__import__('datetime').datetime.now().strftime('%Y%m%d%H%M%S')}",
            }
        )
        
        if exito:
            procesados_exitosos += 1
            print(f"     ✅ Procesado → {os.path.relpath(ruta_destino_archivo)}")
        else:
            errores += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESUMEN:")
    print(f"   Total encontrados: {len(pdfs_encontrados)}")
    print(f"   Procesados exitosamente: {procesados_exitosos}")
    print(f"   Errores: {errores}")
    print(f"   📁 Copias guardadas en: {ruta_destino_completa}")
    
    return {
        "procesados": procesados_exitosos,
        "errores": errores,
        "total": len(pdfs_encontrados),
        "carpeta_destino": ruta_destino_completa
    }

def leer_metadata_pdf(ruta_pdf):
    """
    Lee y muestra la metadata de un PDF.
    
    Args:
        ruta_pdf (str): Ruta del archivo PDF
    """
    try:
        reader = PdfReader(ruta_pdf)
        metadata = reader.metadata
        
        print(f"\n📋 Metadata de: {os.path.basename(ruta_pdf)}")
        print("-" * 50)
        
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print("No se encontró metadata.")
            
    except Exception as e:
        print(f"Error leyendo metadata: {e}")

def procesar_desde_txt(archivo_txt, ruta_base, carpeta_destino="pdfs_con_metadata"):
    """
    Procesa PDFs usando la lista de un archivo de texto generado anteriormente.
    
    Args:
        archivo_txt (str): Ruta del archivo de texto con la lista de PDFs
        ruta_base (str): Ruta base original de los PDFs
        carpeta_destino (str): Carpeta donde guardar las copias
    """
    try:
        with open(archivo_txt, 'r', encoding='utf-8') as f:
            contenido = f.read()
        
        # Extraer información de PDFs del archivo
        lineas = contenido.split('\n')
        pdfs_a_procesar = []
        
        for i, linea in enumerate(lineas):
            if linea.strip().endswith('.pdf'):
                # Buscar la línea siguiente que contiene "Ruta:"
                if i + 1 < len(lineas) and 'Ruta:' in lineas[i + 1]:
                    nombre = linea.strip().split('. ', 1)[1] if '. ' in linea else linea.strip()
                    ruta_relativa = lineas[i + 1].replace('   Ruta:', '').strip()
                    ruta_completa = os.path.join(ruta_base, ruta_relativa)
                    
                    if os.path.exists(ruta_completa):
                        pdfs_a_procesar.append((nombre, ruta_completa, ruta_relativa))
        
        print(f"📄 Procesando {len(pdfs_a_procesar)} PDFs desde archivo de texto...")
        
        # Procesar cada PDF
        return procesar_lista_pdfs(pdfs_a_procesar, carpeta_destino)
        
    except Exception as e:
        print(f"Error procesando archivo de texto: {e}")
        return {"procesados": 0, "errores": 1, "total": 0}

def procesar_lista_pdfs(lista_pdfs, carpeta_destino):
    """
    Procesa una lista específica de PDFs.
    
    Args:
        lista_pdfs (list): Lista de tuplas (nombre, ruta_completa, ruta_relativa)
        carpeta_destino (str): Carpeta de destino
    """
    ruta_destino_completa = os.path.join(os.getcwd(), carpeta_destino)
    os.makedirs(ruta_destino_completa, exist_ok=True)
    
    procesados_exitosos = 0
    errores = 0
    
    for i, (nombre_archivo, ruta_original, ruta_relativa) in enumerate(lista_pdfs, 1):
        print(f"[{i:3d}/{len(lista_pdfs)}] {nombre_archivo}")
        
        # Crear estructura de carpetas
        ruta_relativa_dir = os.path.dirname(ruta_relativa) if os.path.dirname(ruta_relativa) else ""
        carpeta_destino_archivo = os.path.join(ruta_destino_completa, ruta_relativa_dir)
        ruta_destino_archivo = os.path.join(carpeta_destino_archivo, nombre_archivo)
        
        # Procesar el PDF
        exito = agregar_metadata_pdf(ruta_original, ruta_destino_archivo, ruta_relativa.replace('\\', '/'))
        
        if exito:
            procesados_exitosos += 1
            print(f"     ✅ Procesado")
        else:
            errores += 1
    
    return {"procesados": procesados_exitosos, "errores": errores, "total": len(lista_pdfs)}

def main():
    """Función principal"""
    print("📝 ACTUALIZADOR DE METADATA DE PDFs")
    print("=" * 50)
    
    print("Opciones:")
    print("1. Procesar PDFs desde una carpeta")
    print("2. Procesar PDFs desde archivo de texto")
    print("3. Leer metadata de un PDF específico")
    
    opcion = input("\nElige una opción (1-3): ").strip()
    
    if opcion == "1":
        ruta = input("Ingresa la ruta base donde buscar PDFs: ").strip()
        if ruta.startswith('"') and ruta.endswith('"'):
            ruta = ruta[1:-1]
        
        carpeta_dest = input("Carpeta de destino (Enter para 'pdfs_con_metadata'): ").strip()
        if not carpeta_dest:
            carpeta_dest = "pdfs_con_metadata"
        
        procesar_pdfs_con_metadata(ruta, carpeta_dest)
        
    elif opcion == "2":
        archivo_txt = input("Ruta del archivo de texto: ").strip()
        ruta_base = input("Ruta base original de los PDFs: ").strip()
        carpeta_dest = input("Carpeta de destino (Enter para 'pdfs_con_metadata'): ").strip()
        
        if archivo_txt.startswith('"') and archivo_txt.endswith('"'):
            archivo_txt = archivo_txt[1:-1]
        if ruta_base.startswith('"') and ruta_base.endswith('"'):
            ruta_base = ruta_base[1:-1]
        if not carpeta_dest:
            carpeta_dest = "pdfs_con_metadata"
            
        procesar_desde_txt(archivo_txt, ruta_base, carpeta_dest)
        
    elif opcion == "3":
        ruta_pdf = input("Ruta del PDF para leer metadata: ").strip()
        reader = PdfReader(ruta_pdf)
        metadata = reader.metadata
        for k, v in metadata.items():
            print(k, v)
    
    else:
        print("Opción no válida.")

if __name__ == "__main__":
    main()