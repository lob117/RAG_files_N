import fitz  # PyMuPDF
import pandas as pd
import tabula
import os
from typing import List, Dict, Tuple
import logging
import re

# Importar librerías con manejo de errores
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import camelot.io as camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    try:
        import camelot
        CAMELOT_AVAILABLE = True
    except ImportError:
        CAMELOT_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFTableDetector:
    def __init__(self):
        """
        Inicializa el detector de tablas para PDFs de texto.
        """
        self.results = []
        logger.info(f"PDFPlumber: {'✅' if PDFPLUMBER_AVAILABLE else '❌'}")
        logger.info(f"Camelot: {'✅' if CAMELOT_AVAILABLE else '❌'}")
    
    def detect_tables_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """
        Detecta tablas usando PDFplumber - IDEAL para tablas sin bordes
        """
        tables_found = []
        
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("PDFplumber no disponible")
            return tables_found
        
        try:
            logger.info("🔍 Analizando con PDFplumber...")
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"Página {page_num + 1}/{len(pdf.pages)}")
                    
                    # Método 1: Detección automática
                    try:
                        tables = page.extract_tables()
                        if tables:
                            for i, table in enumerate(tables):
                                if self._is_valid_table(table):
                                    confidence = self._calculate_table_confidence(table)
                                    tables_found.append({
                                        'page': page_num + 1,
                                        'method': 'pdfplumber_auto',
                                        'accuracy': confidence['score'],
                                        'table_index': i,
                                        'shape': (len(table), len(table[0]) if table else 0),
                                        'confidence': confidence['level'],
                                        'table_data': table[:3] if table else []  # Primeras 3 filas como muestra
                                    })
                    except Exception as e:
                        logger.warning(f"PDFplumber auto falló en página {page_num + 1}: {e}")
                    
                    # Método 2: Detección por análisis de texto con coordenadas
                    try:
                        text_tables = self._detect_tables_by_coordinates(page, page_num + 1)
                        tables_found.extend(text_tables)
                    except Exception as e:
                        logger.warning(f"Detección por coordenadas falló: {e}")
                    
                    # Método 3: Búsqueda de patrones específicos (TABLA, CUADRO, etc.)
                    try:
                        pattern_tables = self._detect_tables_by_patterns(page, page_num + 1)
                        tables_found.extend(pattern_tables)
                    except Exception as e:
                        logger.warning(f"Detección por patrones falló: {e}")
                        
        except Exception as e:
            logger.error(f"Error en PDFplumber: {e}")
        
        return tables_found
    
    def _detect_tables_by_coordinates(self, page, page_num: int) -> List[Dict]:
        """
        Detecta tablas analizando las coordenadas del texto
        """
        tables_found = []
        
        try:
            # Extraer caracteres con coordenadas
            chars = page.chars
            if not chars:
                return tables_found
            
            # Agrupar caracteres por líneas (similar Y)
            lines = {}
            for char in chars:
                y_pos = round(char['y0'])  # Posición Y redondeada
                if y_pos not in lines:
                    lines[y_pos] = []
                lines[y_pos].append(char)
            
            # Analizar líneas para detectar estructura tabular
            sorted_lines = sorted(lines.keys(), reverse=True)  # De arriba hacia abajo
            
            table_regions = []
            current_region = []
            
            for i, y_pos in enumerate(sorted_lines):
                line_chars = sorted(lines[y_pos], key=lambda x: x['x0'])
                line_text = ''.join([c['text'] for c in line_chars]).strip()
                
                if not line_text:
                    continue
                
                # Detectar si la línea tiene estructura tabular
                if self._is_tabular_line(line_text, line_chars):
                    current_region.append({
                        'y_pos': y_pos,
                        'text': line_text,
                        'chars': line_chars
                    })
                else:
                    # Si tenemos una región acumulada, evaluarla
                    if len(current_region) >= 3:  # Mínimo 3 filas para considerar tabla
                        table_confidence = self._evaluate_table_region(current_region)
                        if table_confidence['score'] > 60:
                            tables_found.append({
                                'page': page_num,
                                'method': 'pdfplumber_coordinates',
                                'accuracy': table_confidence['score'],
                                'table_index': len(tables_found),
                                'shape': (len(current_region), table_confidence['columns']),
                                'confidence': table_confidence['level'],
                                'region_data': current_region[:3]  # Muestra
                            })
                    current_region = []
            
            # Evaluar última región si existe
            if len(current_region) >= 3:
                table_confidence = self._evaluate_table_region(current_region)
                if table_confidence['score'] > 60:
                    tables_found.append({
                        'page': page_num,
                        'method': 'pdfplumber_coordinates',
                        'accuracy': table_confidence['score'],
                        'table_index': len(tables_found),
                        'shape': (len(current_region), table_confidence['columns']),
                        'confidence': table_confidence['level'],
                        'region_data': current_region[:3]
                    })
                    
        except Exception as e:
            logger.error(f"Error en detección por coordenadas: {e}")
        
        return tables_found
    
    def _is_tabular_line(self, text: str, chars: List[Dict]) -> bool:
        """
        Determina si una línea tiene estructura tabular
        """
        criteria = 0
        
        # 1. Múltiples números
        numbers = re.findall(r'\d+\.?\d*', text)
        if len(numbers) >= 2:
            criteria += 1
        
        # 2. Espaciado consistente entre elementos
        if len(chars) > 5:
            spaces = []
            last_x = chars[0]['x1']
            
            for char in chars[1:]:
                if char['text'] != ' ':
                    space = char['x0'] - last_x
                    if space > 5:
                        spaces.append(space)
                    last_x = char['x1']
            
            if len(spaces) >= 2:
                criteria += 1
        
        # 3. Patrones típicos de tabla
        table_patterns = [r'\d+\.\d+', r'\d+,\d+', r'mg\s*/l', r'UNIDAD', r'PARAMETROS']
        for pattern in table_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                criteria += 1
                break
        
        return criteria >= 2
    
    def _evaluate_table_region(self, region: List[Dict]) -> Dict:
        """
        Evalúa la calidad de una región como tabla
        """
        if not region:
            return {'score': 0, 'level': 'low', 'columns': 0}
        
        score = 0
        row_lengths = []
        total_numbers = 0
        
        for row in region:
            words = row['text'].split()
            row_lengths.append(len(words))
            numbers = re.findall(r'\d+\.?\d*', row['text'])
            total_numbers += len(numbers)
        
        if row_lengths:
            avg_length = sum(row_lengths) / len(row_lengths)
            consistency = 1 - (max(row_lengths) - min(row_lengths)) / max(avg_length, 1)
            score += consistency * 40
        
        total_words = sum(row_lengths)
        if total_words > 0:
            numeric_density = total_numbers / total_words
            score += numeric_density * 40
        
        if len(region) >= 5:
            score += 20
        elif len(region) >= 3:
            score += 10
        
        level = 'high' if score > 80 else 'medium' if score > 60 else 'low'
        estimated_columns = int(avg_length) if row_lengths else 0
        
        return {
            'score': min(100, score),
            'level': level,
            'columns': estimated_columns
        }
    
    def _detect_tables_by_patterns(self, page, page_num: int) -> List[Dict]:
        """
        Detecta tablas buscando patrones específicos como 'TABLA', 'CUADRO'
        """
        tables_found = []
        
        try:
            text = page.extract_text()
            if not text:
                return tables_found
            
            lines = text.split('\n')
            
            table_indicators = [
                r'TABLA\s+\d+', r'CUADRO\s+\d+', r'Table\s+\d+',
                r'PARAMETROS.*UNIDAD', r'ANALISIS.*FISICO.*QUIMICO'
            ]
            
            for i, line in enumerate(lines):
                for pattern in table_indicators:
                    if re.search(pattern, line, re.IGNORECASE):
                        logger.info(f"🎯 Patrón detectado en página {page_num}: {line.strip()}")
                        table_data = self._extract_table_after_header(lines, i)
                        
                        if table_data['rows'] >= 3:
                            tables_found.append({
                                'page': page_num,
                                'method': 'pdfplumber_pattern',
                                'accuracy': table_data['confidence'],
                                'table_index': len(tables_found),
                                'shape': (table_data['rows'], table_data['columns']),
                                'confidence': 'high' if table_data['confidence'] > 80 else 'medium',
                                'header': line.strip(),
                                'sample_data': table_data['sample']
                            })
                        break
                        
        except Exception as e:
            logger.error(f"Error en detección por patrones: {e}")
        
        return tables_found
    
    def _extract_table_after_header(self, lines: List[str], header_index: int) -> Dict:
        """
        Extrae datos de tabla después de encontrar un encabezado
        """
        table_rows = 0
        max_columns = 0
        sample_data = []
        confidence = 0
        
        for i in range(header_index + 1, min(len(lines), header_index + 21)):
            line = lines[i].strip()
            
            if not line:
                continue
            
            if (len(line) > 200 or 
                any(keyword in line.lower() for keyword in ['conclusión', 'resultado', 'en el', 'según'])):
                break
            
            words = line.split()
            numbers = re.findall(r'\d+\.?\d*', line)
            
            if len(words) >= 2 and (len(numbers) >= 1 or any(word in line for word in ['mg', 'UNID', '/'])):
                table_rows += 1
                max_columns = max(max_columns, len(words))
                if len(sample_data) < 3:
                    sample_data.append(line)
                confidence += len(numbers) * 5 + len(words) * 2
        
        if table_rows > 0:
            confidence = min(100, confidence / table_rows)
        
        return {
            'rows': table_rows,
            'columns': max_columns,
            'confidence': confidence,
            'sample': sample_data
        }
    
    def detect_tables_vectorized_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Detecta tablas en PDFs vectorizados - Estrategia híbrida optimizada
        """
        tables_found = []
        
        # Prioridad 1: PDFplumber (mejor para tablas sin bordes)
        if PDFPLUMBER_AVAILABLE:
            pdfplumber_tables = self.detect_tables_with_pdfplumber(pdf_path)
            tables_found.extend(pdfplumber_tables)
            
            if pdfplumber_tables:
                logger.info(f"✅ PDFplumber encontró {len(pdfplumber_tables)} tablas")
                return tables_found
        
        # Prioridad 2: Camelot stream (para tablas sin bordes)
        if CAMELOT_AVAILABLE:
            try:
                logger.info("🔍 Intentando Camelot stream...")
                camelot_stream = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                for i, table in enumerate(camelot_stream):
                    if hasattr(table, 'parsing_report') and table.parsing_report['accuracy'] > 30:
                        tables_found.append({
                            'page': table.parsing_report['page'],
                            'method': 'camelot_stream',
                            'accuracy': table.parsing_report['accuracy'],
                            'table_index': i,
                            'shape': table.df.shape,
                            'confidence': 'medium' if table.parsing_report['accuracy'] > 50 else 'low'
                        })
                        logger.info(f"✅ Camelot stream: tabla en página {table.parsing_report['page']}")
            except Exception as e:
                logger.warning(f"Camelot stream falló: {e}")
        
        # Prioridad 3: Tabula-py como respaldo
        try:
            logger.info("🔍 Intentando Tabula-py...")
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            if tabula_tables:
                doc = fitz.open(pdf_path)
                total_pages = len(doc)
                doc.close()
                
                for i, table in enumerate(tabula_tables):
                    if not table.empty and table.shape[0] > 1 and table.shape[1] > 1:
                        estimated_page = min((i % total_pages) + 1, total_pages)
                        
                        non_null_cells = table.count().sum()
                        total_cells = table.shape[0] * table.shape[1]
                        fill_ratio = non_null_cells / total_cells if total_cells > 0 else 0
                        
                        if fill_ratio > 0.1:
                            confidence = 'high' if fill_ratio > 0.6 else 'medium' if fill_ratio > 0.3 else 'low'
                            
                            tables_found.append({
                                'page': estimated_page,
                                'method': 'tabula',
                                'accuracy': round(fill_ratio * 100, 2),
                                'table_index': i,
                                'shape': table.shape,
                                'confidence': confidence
                            })
                            logger.info(f"✅ Tabula: tabla en página {estimated_page}")
                            
        except Exception as e:
            logger.warning(f"Tabula falló: {e}")
        
        return tables_found
    
    def _is_valid_table(self, table: List[List]) -> bool:
        """
        Valida si una tabla extraída es realmente una tabla
        """
        if not table or len(table) < 2 or not table[0] or len(table[0]) < 2:
            return False
        
        numeric_cells = 0
        total_cells = 0
        
        for row in table:
            for cell in row:
                if cell:
                    total_cells += 1
                    if re.search(r'\d', str(cell)):
                        numeric_cells += 1
        
        return numeric_cells / max(total_cells, 1) > 0.2
    
    def _calculate_table_confidence(self, table: List[List]) -> Dict:
        """
        Calcula la confianza de una tabla extraída
        """
        if not table:
            return {'score': 0, 'level': 'low'}
        
        score = 0
        
        row_lengths = [len(row) for row in table if row]
        if row_lengths:
            max_len = max(row_lengths)
            consistency = sum(1 for length in row_lengths if length == max_len) / len(row_lengths)
            score += consistency * 30
        
        total_cells = sum(len(row) for row in table)
        numeric_cells = 0
        for row in table:
            for cell in row:
                if cell and re.search(r'\d+\.?\d*', str(cell)):
                    numeric_cells += 1
        
        if total_cells > 0:
            numeric_ratio = numeric_cells / total_cells
            score += numeric_ratio * 40
        
        if len(table) >= 5:
            score += 20
        elif len(table) >= 3:
            score += 10
        
        if len(table) > 0 and len(table[0]) >= 3:
            score += 10
        
        level = 'high' if score > 80 else 'medium' if score > 60 else 'low'
        
        return {'score': min(100, score), 'level': level}
    
    def detect_tables(self, pdf_path: str) -> Dict:
        """
        Función principal para detectar tablas en PDFs de texto.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"El archivo {pdf_path} no existe")
        
        logger.info(f"🚀 Iniciando detección optimizada para: {pdf_path}")
        
        pdf_type = 'vectorized'  # Asumimos que todos los PDFs son de texto
        logger.info(f"📄 Tipo de PDF: {pdf_type.upper()}")
        
        # Llamar directamente al método para PDFs vectorizados
        tables = self.detect_tables_vectorized_pdf(pdf_path)
        
        # Remover duplicados
        tables = self._remove_duplicate_tables(tables)
        
        # Organizar resultados
        results = {
            'pdf_path': pdf_path,
            'pdf_type': pdf_type,
            'total_tables': len(tables),
            'tables_by_page': {},
            'summary': [],
            'detection_methods': list(set([t['method'] for t in tables]))
        }
        
        for table in tables:
            page = table['page']
            if page not in results['tables_by_page']:
                results['tables_by_page'][page] = []
            results['tables_by_page'][page].append(table)
        
        for page, page_tables in results['tables_by_page'].items():
            high_conf = len([t for t in page_tables if t['confidence'] == 'high'])
            medium_conf = len([t for t in page_tables if t['confidence'] == 'medium'])
            low_conf = len([t for t in page_tables if t['confidence'] == 'low'])
            
            results['summary'].append({
                'page': page,
                'tables_count': len(page_tables),
                'high_confidence': high_conf,
                'medium_confidence': medium_conf,
                'low_confidence': low_conf,
                'methods_used': list(set([t['method'] for t in page_tables]))
            })
        
        return results
    
    def _remove_duplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Remueve tablas duplicadas (misma página detectada por múltiples métodos)
        """
        if not tables:
            return tables
        
        by_page = {}
        for table in tables:
            page = table['page']
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(table)
        
        filtered_tables = []
        for page, page_tables in by_page.items():
            if len(page_tables) == 1:
                filtered_tables.extend(page_tables)
            else:
                sorted_tables = sorted(page_tables, 
                                     key=lambda x: (x['confidence'] == 'high', 
                                                  x['accuracy'] if isinstance(x['accuracy'], (int, float)) else 0),
                                     reverse=True)
                
                filtered_tables.append(sorted_tables[0])
                
                if (len(sorted_tables) > 1 and 
                    sorted_tables[1]['confidence'] == 'high' and
                    sorted_tables[0]['method'] != sorted_tables[1]['method']):
                    filtered_tables.append(sorted_tables[1])
        
        return filtered_tables
    
    def print_results(self, results: Dict):
        """
        Imprime resultados con formato mejorado
        """
        print(f"\n{'='*60}")
        print(f"🔍 DETECTOR DE TABLAS PDF - RESULTADOS")
        print(f"{'='*60}")
        print(f"📁 Archivo: {os.path.basename(results['pdf_path'])}")
        print(f"📄 Tipo: {results['pdf_type'].upper()}")
        print(f"🔧 Métodos usados: {', '.join(results['detection_methods'])}")
        print(f"📊 Total tablas encontradas: {results['total_tables']}")
        
        if results['total_tables'] > 0:
            print(f"\n{'📋 RESUMEN POR PÁGINA'}")
            print("-" * 40)
            
            for page_info in results['summary']:
                print(f"📄 Página {page_info['page']}: {page_info['tables_count']} tabla(s)")
                
                if page_info['high_confidence'] > 0:
                    print(f"   ✅ Alta confianza: {page_info['high_confidence']}")
                if page_info['medium_confidence'] > 0:
                    print(f"   🟡 Media confianza: {page_info['medium_confidence']}")
                if page_info['low_confidence'] > 0:
                    print(f"   🔴 Baja confianza: {page_info['low_confidence']}")
                
                print(f"   🔧 Métodos: {', '.join(page_info['methods_used'])}")
                print()
            
            print(f"\n{'📋 DETALLES DE TABLAS'}")
            print("-" * 40)
            
            for page, tables in results['tables_by_page'].items():
                for i, table in enumerate(tables):
                    confidence_icon = {'high': '✅', 'medium': '🟡', 'low': '🔴'}.get(table['confidence'], '❓')
                    
                    print(f"{confidence_icon} Página {page}, Tabla {i+1}:")
                    print(f"   Método: {table['method']}")
                    print(f"   Confianza: {table['confidence']} ({table.get('accuracy', 0):.1f}%)")
                    if 'shape' in table:
                        print(f"   Dimensiones: {table['shape'][0]} filas x {table['shape'][1]} columnas")
                    
                    if 'table_data' in table and table['table_data']:
                        print(f"   Muestra de datos:")
                        for j, row in enumerate(table['table_data'][:2]):
                            row_text = " | ".join([str(cell)[:20] + "..." if len(str(cell)) > 20 else str(cell) 
                                                 for cell in row[:4]])
                            print(f"     Fila {j+1}: {row_text}")
                    
                    elif 'sample_data' in table and table['sample_data']:
                        print(f"   Muestra de datos:")
                        for j, sample in enumerate(table['sample_data'][:2]):
                            print(f"     Fila {j+1}: {sample[:80]}...")
                    
                    print()
            
            print(f"🎯 PÁGINAS CON TABLAS: {', '.join(map(str, sorted(results['tables_by_page'].keys())))}")
        else:
            print(f"\n❌ No se encontraron tablas en el PDF")
        
        print(f"{'='*60}\n")

    # ----- INICIO DE LA MODIFICACIÓN -----
    def export_results_to_json(self, results: Dict, output_path: str = None):
        """
        Exporta los resultados a un archivo JSON de forma simplificada.
        Solo incluye el nombre del archivo y las páginas con tablas.
        """
        import json
        
        # Si no se encontraron tablas, no se genera el archivo.
        if not results['tables_by_page']:
            logger.info("No se encontraron tablas, no se generará archivo JSON.")
            return None
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(results['pdf_path']))[0]
            output_path = f"{base_name}_resultados_tablas.json"
        
        # Crear la estructura de datos simplificada
        simplified_data = {
            'nombre_pdf': os.path.basename(results['pdf_path']),
            'paginas_con_tablas': self.get_pages_with_tables(results)
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_data, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Resultados simplificados exportados a: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exportando resultados simplificados: {e}")
            return None
    # ----- FIN DE LA MODIFICACIÓN -----

    def get_pages_with_tables(self, results: Dict) -> List[int]:
        """
        Retorna una lista simple de páginas que contienen tablas
        """
        return sorted(list(results['tables_by_page'].keys()))

    def get_high_confidence_tables(self, results: Dict) -> Dict:
        """
        Retorna solo las tablas con alta confianza
        """
        high_conf_tables = {}
        
        for page, tables in results['tables_by_page'].items():
            high_conf = [t for t in tables if t['confidence'] == 'high']
            if high_conf:
                high_conf_tables[page] = high_conf
        
        return high_conf_tables

# Función de conveniencia para uso rápido
def detect_pdf_tables(pdf_path: str, export_json: bool = False) -> Dict:
    """
    Función de conveniencia para detectar tablas rápidamente
    """
    detector = PDFTableDetector()
    results = detector.detect_tables(pdf_path)
    detector.print_results(results)
    
    if export_json:
        detector.export_results_to_json(results)
    
    return results

# Ejemplo de uso
def main():
    """
    Ejemplo de uso principal
    """
    pdf_path = r"C:\Users\javie\CORPORACIÓN NATURAL SIG\pruebas_docs - OCR_Pdf\061-96-TM-1.pdf"
    
    try:
        print("🚀 DETECTOR DE TABLAS PDF - Versión solo para texto")
        print("=" * 60)
        
        results = detect_pdf_tables(pdf_path, export_json=True)
        
        if results and results['total_tables'] > 0:
            detector = PDFTableDetector()
            print("\n🔧 FUNCIONALIDADES ADICIONALES:")
            print("-" * 30)
            
            pages_with_tables = detector.get_pages_with_tables(results)
            print(f"📋 Páginas con tablas: {pages_with_tables}")
            
            high_conf = detector.get_high_confidence_tables(results)
            if high_conf:
                print(f"✅ Páginas con tablas de alta confianza: {list(high_conf.keys())}")
    
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {pdf_path}")
    
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        logger.error(f"Error en main: {e}")

if __name__ == "__main__":
    main()