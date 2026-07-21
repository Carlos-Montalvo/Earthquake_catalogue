#!/usr/bin/env python3
"""
Compara un catálogo sísmico en XML (QuakeML) con un catálogo CMT en CSV
y crea un nuevo XML con los eventos coincidentes.

Usa librerías de ObsPy para máxima compatibilidad con flujos de sismología.

Uso:
    python match_cmt_catalog.py --input-xml catalog.xml --cmt-csv cmt_catalog.csv --output output.xml
    
Criterios de coincidencia:
    - Diferencia de tiempo: ±30 segundos (ajustable)
    - Distancia epicentral: ≤25 km (ajustable)
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

from obspy import read_events
from obspy.core.event import Catalog, Event
from obspy.geodetics import locations2degrees
import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CMTCatalog:
    """Lee y almacena eventos del catálogo CMT en formato CSV."""
    
    def __init__(self, csv_path: str):
        """
        Inicializa el catálogo CMT desde un archivo CSV.
        
        Args:
            csv_path: Ruta del archivo CSV
        """
        self.events: Dict[str, Dict] = {}
        self.load_csv(csv_path)
    
    def load_csv(self, csv_path: str):
        """
        Carga eventos desde CSV de CMT usando pandas para mejor robustez.
        
        Args:
            csv_path: Ruta del archivo CSV
        """
        try:
            # Leer CSV con pandas
            df = pd.read_csv(csv_path)
            
            if df.empty:
                raise ValueError("CSV vacío")
            
            logger.info(f"CSV cargado: {len(df)} filas")
            
            # Verificar que existen las columnas requeridas
            required_cols = ['PublicID', 'Date', 'Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Columnas faltantes: {missing_cols}")
            
            # Parsear fechas de forma robusta
            df['Date_parsed'] = pd.to_datetime(
                df['Date'],
                format='mixed',  # Permite múltiples formatos
                utc=True,
                errors='coerce'  # Convierte errores a NaT
            )
            
            # Verificar si hay fechas que no se pudieron parsear
            failed_dates = df[df['Date_parsed'].isna()]
            if len(failed_dates) > 0:
                logger.warning(f"No se pudieron parsear {len(failed_dates)} fechas:")
                for idx, row in failed_dates.iterrows():
                    logger.warning(f"  Fila {idx+2}: {row['Date']}")
            
            # Procesar cada evento
            for idx, row in df.iterrows():
                try:
                    event_id = row['PublicID']
                    
                    # Obtener datetime parseado
                    dt_parsed = row['Date_parsed']
                    if pd.isna(dt_parsed):
                        logger.warning(f"Evento {event_id} (fila {idx+2}): fecha no válida, ignorando")
                        continue
                    
                    # Convertir de Timestamp a datetime de Python
                    dt = dt_parsed.to_pydatetime()
                    
                    # Verificar coordenadas
                    try:
                        lat = float(row['Latitude'])
                        lon = float(row['Longitude'])
                        
                        if not (-90 <= lat <= 90):
                            logger.warning(f"Evento {event_id}: latitud inválida {lat}, ignorando")
                            continue
                        if not (-180 <= lon <= 180):
                            logger.warning(f"Evento {event_id}: longitud inválida {lon}, ignorando")
                            continue
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Evento {event_id}: coordenadas inválidas, ignorando")
                        continue
                    
                    self.events[event_id] = {
                        'id': event_id,
                        'datetime': dt,
                        'latitude': lat,
                        'longitude': lon,
                        'ml': float(row['ML']),
                        'mw': float(row['Mw']),
                        'mo': float(row['Mo']),
                        'strike1': float(row['strike1']),
                        'dip1': float(row['dip1']),
                        'rake1': float(row['rake1']),
                        'strike2': float(row['strike2']),
                        'dip2': float(row['dip2']),
                        'rake2': float(row['rake2']),
                        'cd': float(row['CD']),
                        'ns': float(row['NS']),
                        'dc': float(row['DC']),
                        'vr': float(row['VR']),
                        'row': row  # Guardar fila completa para referencia
                    }
                
                except Exception as e:
                    logger.warning(f"Error procesando evento en fila {idx+2}: {e}")
                    continue
            
            logger.info(f"Cargados exitosamente {len(self.events)} eventos del catálogo CMT")
            
            if len(self.events) == 0:
                logger.warning("⚠️  No se cargó ningún evento válido del CSV")
        
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar CSV: {e}")
            raise
    
    @staticmethod
    def _parse_float(value: Optional[str]) -> Optional[float]:
        """Parsea un valor a float, manejando valores nulos."""
        if not value or value.strip() == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _parse_scientific(value: Optional[str]) -> Optional[float]:
        """Parsea notación científica."""
        if not value or value.strip() == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def find_match(self, time_window: timedelta, distance_threshold: float,
                   target_time: datetime, target_lat: float, 
                   target_lon: float) -> Optional[Tuple[str, Dict]]:
        """
        Busca un evento coincidente en el catálogo CMT.

        Args:
            time_window: Ventana de tiempo para coincidencia (timedelta)
            distance_threshold: Distancia máxima en km
            target_time: Tiempo del evento objetivo
            target_lat: Latitud del evento objetivo
            target_lon: Longitud del evento objetivo

        Returns:
            Tupla (event_id, event_dict) del evento CMT coincidente o None
        """
        # Normalizar target_time a UTC aware si es naive
        if target_time.tzinfo is None:
            from datetime import timezone
            target_time = target_time.replace(tzinfo=timezone.utc)

        best_match = None
        best_distance = distance_threshold

        for event_id, event in self.events.items():
            # Verificar ventana de tiempo
            event_dt = event['datetime']

            # Asegurar que ambos tengan la misma zona horaria
            if event_dt.tzinfo is None:
                from datetime import timezone
                event_dt = event_dt.replace(tzinfo=timezone.utc)

            time_diff = abs((event_dt - target_time).total_seconds())
            if time_diff > time_window.total_seconds():
                continue
            
            # Calcular distancia epicentral usando ObsPy
            try:
                distance_deg = locations2degrees(
                    target_lat, target_lon,
                    event['latitude'], event['longitude']
                )
                # Convertir grados a km (aproximado: 1 grado ≈ 111 km)
                distance_km = distance_deg * 111.0
            except Exception as e:
                logger.warning(f"Error calculando distancia para {event_id}: {e}")
                continue
            
            # Verificar distancia epicentral
            if distance_km <= best_distance:
                best_distance = distance_km
                best_match = (event_id, event)

        return best_match

class SeismicCatalogMatch:
    """Maneja la comparación y generación de catálogos usando ObsPy."""
    
    def __init__(self, input_xml: str):
        """
        Inicializa desde un catálogo XML (QuakeML).
        
        Args:
            input_xml: Ruta del archivo XML
        """
        try:
            self.catalog = read_events(input_xml)
            logger.info(f"Catálogo XML cargado: {len(self.catalog)} eventos")
        except Exception as e:
            logger.error(f"Error al cargar XML: {e}")
            raise
    
    def extract_event_info(self, event: Event) -> Optional[Dict]:
        """
        Extrae información relevante de un evento ObsPy.
        
        Args:
            event: Objeto Event de ObsPy
        
        Returns:
            Diccionario con info del evento o None si falta info crítica
        """
        try:
            origin = event.preferred_origin()
            if origin is None:
                origin = event.origins[0] if event.origins else None
            
            if origin is None or origin.time is None:
                logger.warning(f"Evento {event.resource_id} sin origen válido")
                return None
            
            if origin.latitude is None or origin.longitude is None:
                logger.warning(f"Evento {event.resource_id} sin coordenadas")
                return None
            
            return {
                'id': str(event.resource_id),
                'event': event,
                'datetime': origin.time.datetime,
                'latitude': origin.latitude,
                'longitude': origin.longitude,
                'depth': origin.depth / 1000 if origin.depth else None,  # Convertir a km
                'magnitude': event.preferred_magnitude().mag 
                            if event.preferred_magnitude() else None
            }
        except Exception as e:
            logger.warning(f"Error extrayendo info de evento: {e}")
            return None
    
    def get_all_events(self) -> List[Dict]:
        """
        Extrae información de todos los eventos.
        
        Returns:
            Lista de diccionarios con información de eventos
        """
        events = []
        for event in self.catalog:
            event_info = self.extract_event_info(event)
            if event_info:
                events.append(event_info)
        
        logger.info(f"Extraídos {len(events)} eventos válidos del catálogo XML")
        return events
    
    def match_with_cmt(self, cmt_catalog: CMTCatalog, 
                       time_window: timedelta, 
                       distance_threshold: float) -> Tuple[Catalog, List[Dict]]:
        """
        Busca coincidencias 1:1 con catálogo CMT (evita duplicados).
        Cada CMT se asigna al mejor XML encontrado.
        
        Args:
            cmt_catalog: Instancia de CMTCatalog
            time_window: Ventana temporal
            distance_threshold: Distancia máxima
        
        Returns:
            Tupla (Catalog con eventos coincidentes, lista de matches CMT)
        """
        matched_catalog = Catalog()
        matched_cmt = []
        
        # Obtener todos los eventos XML
        xml_events = self.get_all_events()
        
        logger.info(f"Matching 1:1 entre {len(cmt_catalog.events)} CMTs y {len(xml_events)} XMLs")
        logger.info(f"Parámetros: ±{time_window.total_seconds()}s, ≤{distance_threshold}km...")
        
        # Tracking de XMLs ya utilizados (evitar reutilización)
        matched_xml_indices = set()
        
        # Iterar sobre CMTs (el catálogo más pequeño)
        for cmt_id, cmt_event in cmt_catalog.events.items():
            cmt_time = cmt_event['datetime']
            cmt_lat = cmt_event['latitude']
            cmt_lon = cmt_event['longitude']
            
            # Normalizar CMT datetime a UTC aware
            from datetime import timezone
            if cmt_time.tzinfo is None:
                cmt_time = cmt_time.replace(tzinfo=timezone.utc)
            
            best_xml_idx = None
            best_delta = float('inf')
            best_dist = float('inf')
            
            # Buscar el mejor XML para este CMT
            for xml_idx, xml_event in enumerate(xml_events):
                # Saltar si ya fue utilizado
                if xml_idx in matched_xml_indices:
                    continue
                
                xml_time = xml_event['datetime']
                xml_lat = xml_event['latitude']
                xml_lon = xml_event['longitude']
                
                # Normalizar XML datetime a UTC aware
                if xml_time.tzinfo is None:
                    xml_time = xml_time.replace(tzinfo=timezone.utc)
                
                # 1. Verificar ventana de tiempo
                time_diff = abs((cmt_time - xml_time).total_seconds())
                if time_diff > time_window.total_seconds():
                    continue
                
                # 2. Verificar distancia epicentral
                try:
                    from obspy.geodetics import gps2dist_azimuth
                    dist_m, _, _ = gps2dist_azimuth(
                        lat1=cmt_lat,
                        lon1=cmt_lon,
                        lat2=xml_lat,
                        lon2=xml_lon
                    )
                    dist_km = dist_m / 1000.0
                except Exception as e:
                    logger.warning(f"Error calculando distancia para {cmt_id}: {e}")
                    continue
                
                if dist_km > distance_threshold:
                    continue
                
                # 3. Actualizar mejor match (priorizar: tiempo > distancia)
                is_better = False
                if time_diff < best_delta:
                    is_better = True
                elif time_diff == best_delta and dist_km < best_dist:
                    is_better = True
                
                if is_better:
                    best_xml_idx = xml_idx
                    best_delta = time_diff
                    best_dist = dist_km
            
            # Si encontró un buen match
            if best_xml_idx is not None:
                xml_event = xml_events[best_xml_idx]
                matched_catalog.append(xml_event['event'])
                matched_cmt.append(cmt_event)
                matched_xml_indices.add(best_xml_idx)
                
                # Log detallado
                lat_diff = abs(xml_event['latitude'] - cmt_lat)
                lon_diff = abs(xml_event['longitude'] - cmt_lon)
                
                logger.info(
                    f"✓ Match {len(matched_cmt)}: {cmt_id} ← {xml_event['id']} "
                    f"(Δt={best_delta:.1f}s, Δlat={lat_diff:.3f}°, Δlon={lon_diff:.3f}°)"
                )
        
        logger.info(f"Se encontraron {len(matched_catalog)} eventos coincidentes (1:1)")
        logger.info(f"XMLs utilizados: {len(matched_xml_indices)} de {len(xml_events)}")
        return matched_catalog, matched_cmt
    
    def save_matched_catalog(self, matched_catalog: Catalog, output_path: str):
        """
        Guarda el catálogo de eventos coincidentes.
        
        Args:
            matched_catalog: Catálogo con eventos coincidentes
            output_path: Ruta del archivo de salida
        """
        try:
            matched_catalog.write(output_path, format='QUAKEML')
            logger.info(f"Catálogo coincidente guardado en {output_path}")
        except Exception as e:
            logger.error(f"Error guardando catálogo: {e}")
            raise
    
    # @staticmethod
    # def save_cmt_matches_csv(matched_cmt: List[Dict], output_path: str):
    #     """
    #     Guarda los datos CMT de los eventos coincidentes en CSV.
        
    #     Args:
    #         matched_cmt: Lista de eventos CMT coincidentes
    #         output_path: Ruta del archivo CSV de salida
    #     """
    #     if not matched_cmt:
    #         logger.info("No hay eventos coincidentes para guardar en CSV")
    #         return
        
    #     try:
    #         # Usar headers del primer evento
    #         fieldnames = list(matched_cmt[0]['row'].keys())
            
    #         with open(output_path, 'w', newline='') as f:
    #             writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
    #             writer.writeheader()
    #             for cmt_event in matched_cmt:
    #                 writer.writerow(cmt_event['row'])
            
    #         logger.info(f"CSV de coincidencias guardado en {output_path}")
    #     except Exception as e:
    #         logger.error(f"Error guardando CSV: {e}")
    #         raise


def main():
    parser = argparse.ArgumentParser(
        description='Compara catálogo sísmico XML con catálogo CMT CSV usando ObsPy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s --input-xml catalog.xml --cmt-csv cmt.csv --output matched.xml
  %(prog)s --input-xml catalog.xml --cmt-csv cmt.csv --output matched.xml \\
           --time-window 60 --distance-threshold 20
        """
    )
    
    parser.add_argument('--input-xml', required=True, 
                        help='Ruta del catálogo sísmico en XML (QuakeML)')
    parser.add_argument('--cmt-csv', required=True,
                        help='Ruta del catálogo CMT en CSV')
    parser.add_argument('--output', required=True,
                        help='Ruta del archivo XML de salida')
    parser.add_argument('--time-window', type=int, default=30,
                        help='Ventana de tiempo en segundos (default: 30)')
    parser.add_argument('--distance-threshold', type=float, default=25,
                        help='Distancia máxima en km (default: 25)')
    
    args = parser.parse_args()
    
    # Validar que los archivos existen
    if not Path(args.input_xml).exists():
        logger.error(f"Archivo no encontrado: {args.input_xml}")
        return
    if not Path(args.cmt_csv).exists():
        logger.error(f"Archivo no encontrado: {args.cmt_csv}")
        return
    
    try:
        # Cargar catálogos
        logger.info("Cargando catálogos...")
        cmt_catalog = CMTCatalog(args.cmt_csv)
        xml_catalog = SeismicCatalogMatch(args.input_xml)
        
        # Buscar coincidencias
        time_window = timedelta(seconds=args.time_window)
        matched_catalog, matched_cmt = xml_catalog.match_with_cmt(
            cmt_catalog,
            time_window=time_window,
            distance_threshold=args.distance_threshold
        )
        
        # Guardar resultados
        xml_catalog.save_matched_catalog(matched_catalog, args.output)
        
        # Guardar CSV de coincidencias
        # csv_output = args.output.replace('.xml', '_matches.csv')
        # xml_catalog.save_cmt_matches_csv(matched_cmt, csv_output)
        
        logger.info("¡Proceso completado exitosamente!")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())