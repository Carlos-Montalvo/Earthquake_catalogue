#!/usr/bin/env python3
"""
Waveform cutting for GrowClust3D cross-correlations and RPNet focal mechanisms.
Processes a single day and outputs:
  - GrowClust streams: CATALOGS/GROWCLUST/STREAMS/{year}_{jday}.p
  - RPNet waveforms:   CATALOGS/RPNET/WAVEFORMS/{event_id}/{station}.mseed
"""

import sys
import pickle
import datetime
import os
import gc
import argparse
import math
from glob import glob
from obspy import *
from obspy.taup import TauPyModel
from os.path import join, exists


# DPRI stations whitelist (valid stations only)
DPRI_WHITELIST = {'AGR', 'APD', 'BVL', 'CCB', 'CR1', 'CRF', 'GBR', 'GOT', 'GVR', 'GWS', 
                  'IKR', 'JSP', 'KVR', 'LWP', 'MES', 'MHR', 'MLF', 'NMC', 'NZ2', 'RSR', 
                  'SJQ', 'SRB', 'SVR', 'IGF', 'WDP'}

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

def organize_stations_by_network(inventory):
    """
    Extract and organize stations by network from ObsPy inventory.
    """
    stations_by_network = {
        'MORIA': [],
        'DPRI': [],
        'GEONET': []
    }
    for network in inventory:
        net_code = network.code
        for station in network:
            sta_code = station.code
            if net_code == '5L':
                stations_by_network['MORIA'].append(sta_code)
            elif net_code == 'DP':
                # Only include DPRI stations in the whitelist
                if sta_code in DPRI_WHITELIST:
                    stations_by_network['DPRI'].append(sta_code)
            elif net_code == 'NZ':
                stations_by_network['GEONET'].append(sta_code)
    return stations_by_network


def epicentral_distance_km(event_lat, event_lon, sta_lat, sta_lon):
    """
    Calcula la distancia epicentral en km usando la fórmula de Haversine.
    """
    R = 6371  # Radio de la Tierra en km
    lat1, lon1 = math.radians(event_lat), math.radians(event_lon)
    lat2, lon2 = math.radians(sta_lat), math.radians(sta_lon)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def check_mseed_integrity(file_path, min_duration_hours=18):
    """
    Verifica si un archivo MSEED tiene cobertura temporal mínima.
    Retorna (is_valid, duration_hours, message)
    """
    try:
        from obspy import read
        st = read(file_path)
        if len(st) == 0:
            return False, 0, "Empty stream"
        
        max_duration = 0
        for trace in st:
            duration = (trace.stats.endtime - trace.stats.starttime) / 3600
            max_duration = max(max_duration, duration)
        
        if max_duration < min_duration_hours:
            return False, max_duration, f"Only {max_duration:.1f}h (expected ~24h)"
        else:
            return True, max_duration, f"OK ({max_duration:.1f}h)"
    except Exception as e:
        return False, 0, f"Error: {str(e)[:50]}"

def read_day_data(jday, year, yearpath, nll_dir, station_groups):
    """
    Lee el catálogo NLL y los streams del día PRINCIPAL solo.
    Usa glob para expandir wildcards correctamente.
    Compartido por cut_growclust_streams y cut_rpnet_streams.
    """
    MORIA = station_groups['MORIA']
    DPRI = station_groups['DPRI']
    GEONET = station_groups['GEONET']

    # Leer catálogo
    try:
        cat = read_events(join(nll_dir, f'{year}/{year}_{int(jday):03d}_nll.xml'))
    except Exception as e:
        print(f'No events found for day {jday}: {e}')
        return None, None

    if len(cat) <= 1:
        print(f'No data for day {jday}')
        return None, None

    print(f'Found {len(cat)} events for day {jday}')

    # Después de leer el catálogo en read_day_data
    all_catalog_stations = set()
    for event in cat:
        for pick in event.picks:
            all_catalog_stations.add(pick.waveform_id.station_code)

    all_group_stations = set(MORIA) | set(DPRI) | set(GEONET)
    unclassified = all_catalog_stations - all_group_stations
    if unclassified:
        print(f'  Stations in catalog but not in any group: {sorted(unclassified)}')

    # Leer streams solo del día principal
    jday_str = str(int(jday)).zfill(3)
    st = Stream()
    for network, label in [(MORIA, 'MORIA'), (DPRI, 'DPRI'), (GEONET, 'GEONET')]:
        cnt = 0
        total_traces = 0
        for station in network:
            try:
                pattern = join(yearpath, f'{label}/{station}/*{year}.{jday_str}')
                files = glob(pattern)
                if files:
                    for file_path in files:
                        try:
                            st_tmp = read(file_path)
                            if len(st_tmp) > 0:
                                st += st_tmp
                                total_traces += len(st_tmp)
                        except Exception:
                            pass
                    cnt += 1
            except Exception:
                pass
        print(f'{cnt}/{len(network)} {label} stations read for day {jday} ({total_traces} traces total)')

    # Después del loop de GEONET
    stations_read = set(tr.stats.station for tr in st)
    missing_geonet = [sta for sta in GEONET if sta not in stations_read]
    print(f'  Missing GEONET stations: {missing_geonet}')
    
    # DEBUG: mostrar rango temporal de datos cargados
    if len(st) > 0:
        st_start = min(tr.stats.starttime for tr in st)
        st_end = max(tr.stats.endtime for tr in st)
        print(f'  Data coverage: {st_start} to {st_end}')
    
    if len(st) == 0:
        print(f'  WARNING: No data loaded for day {jday}!')

    return cat, st


def read_adjacent_day_data(jday, year, yearpath, station, label, day_offset):
    """
    Lee datos de un día adyacente BAJO DEMANDA para una estación específica.
    day_offset: -1 (día anterior), +1 (día siguiente)
    """
    from datetime import datetime, timedelta
    
    base_date = datetime.strptime(f'{year}-{int(jday):03d}', '%Y-%j')
    search_date = base_date + timedelta(days=day_offset)
    
    search_year = search_date.year
    search_jday = search_date.timetuple().tm_yday
    search_jday_str = f'{search_jday:03d}'
    
    # Construir ruta correcta si el año cambió
    if search_year != int(year):
        yearpath_search = join(yearpath.rsplit(f'/{year}', 1)[0], str(search_year))
    else:
        yearpath_search = yearpath
    
    pattern = join(yearpath_search, f'{label}/{station}/*{search_year}.{search_jday_str}')
    files = glob(pattern)
    
    st = Stream()
    for file_path in files:
        try:
            st_tmp = read(file_path)
            if len(st_tmp) > 0:
                st += st_tmp
        except Exception:
            pass
    
    return st


def cut_growclust_streams(event, st, inv, station_groups, jday=None, year=None, yearpath=None, 
                          length=3, debug_boundary_picks=False):
    """
    Corta ventanas ±length segundos alrededor de cada pick (P y S) para GrowClust.
    Si un pick cae fuera del Stream principal, intenta leer del día adyacente bajo demanda.
    Retorna un Stream con todas las trazas válidas del evento.
    """
    MORIA = station_groups['MORIA']
    DPRI = station_groups['DPRI']
    GEONET = station_groups['GEONET']

    st2 = Stream()
    valid_picks = 0
    picks_lost_no_channel = 0
    picks_lost_no_data = 0
    picks_lost_no_time_window = 0
    picks_recovered_adjacent = 0
    s_waves_optimized = 0
    
    # Para debug de picks perdidos
    boundary_picks_lost = []

    st_copy = st.copy()

    # Función auxiliar para intentar leer del día adyacente bajo demanda
    def try_adjacent_day_data(pick_time, station, channel_code, force_adjacent=False):
        """
        Intenta leer datos del día adyacente si el pick cae fuera del rango 
        o si force_adjacent=True (cuando no hay datos principales).
        day_offset: -1 (día anterior), +1 (día siguiente)
        """
        nonlocal picks_recovered_adjacent
        
        if not all([jday, year, yearpath]):
            return None
        
        # Convertir a int si es necesario
        jday_int = int(jday) if isinstance(jday, str) else jday
        year_int = int(year) if isinstance(year, str) else year
        
        # Si force_adjacent=True, intenta ambos días adyacentes
        if force_adjacent:
            day_offsets = [-1, 1]
        else:
            # Calcular limites del día en UTC
            day_start = UTCDateTime(year_int, 1, 1) + (jday_int - 1) * 86400
            day_end = day_start + 86400
            
            # Determinar si el pick está fuera del rango y qué día adyacente leer
            if pick_time < day_start:
                day_offsets = [-1]
            elif pick_time >= day_end:
                day_offsets = [1]
            else:
                return None  # Pick está dentro del rango del día principal
        
        # Determinar network label del station
        if station in MORIA:
            label = 'MORIA'
        elif station in DPRI:
            label = 'DPRI'
        elif station in GEONET:
            label = 'GEONET'
        else:
            return None
        
        # Intentar cada día offset
        for day_offset in day_offsets:
            try:
                st_adj = read_adjacent_day_data(jday_int, year_int, yearpath, station, label, day_offset)
                if len(st_adj) > 0:
                    st_sel = st_adj.select(station=station, channel=channel_code).copy()
                    if len(st_sel) > 0:
                        picks_recovered_adjacent += 1
                        return st_sel
            except Exception as e:
                pass
        
        return None

    for pick in event.picks:
        sta = pick.waveform_id.station_code

        # Seleccionar canales según red y fase
        if sta in DPRI:
            channel_codes = ['EHZ'] if pick.phase_hint == 'P' else ['EH1', 'EH2']
        elif sta in MORIA:
            channel_codes = ['*HZ'] if pick.phase_hint == 'P' else ['*HE', '*HN', '*H1', '*H2']
        elif sta in GEONET:
            channel_codes = ['HHZ', 'EHZ'] if pick.phase_hint == 'P' else ['HHE', 'HHN', 'EHE', 'EHN']
        else:
            channel_codes = ['HHZ', 'EHZ', '*HZ'] if pick.phase_hint == 'P' else ['HHE', 'HHN', 'EHE', 'EHN', '*HE', '*HN']

        # Buscar canal en inventario
        channel = None
        for ch_code in channel_codes:
            try:
                channel = inv.select(station=sta, channel=ch_code)[0][0].channels[0]
                break
            except:
                continue

        if channel is None:
            try:
                original_channel = pick.waveform_id.channel_code
                channel = inv.select(station=sta, channel=original_channel)[0][0].channels[0]
            except:
                picks_lost_no_channel += 1
                continue

        pick.waveform_id.channel_code = channel.code
        pick.waveform_id.location_code = channel.location_code

        # Ondas S: seleccionar mejor horizontal
        if pick.phase_hint == 'S':
            best_trace = None
            best_channel_code = None
            best_score = 0
            best_critical_completeness = 0

            if sta in DPRI:
                horizontal_channels = ['EH1', 'EH2', 'EHE', 'EHN']
            elif sta in GEONET:
                horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN', 'HH1', 'HH2', 'EH1', 'EH2']
            elif sta in MORIA:
                horizontal_channels = ['HH1', 'HH2', 'HHE', 'HHN']
            else:
                # TGF y otras no clasificadas
                horizontal_channels = ['EH1', 'EH2', 'HH1', 'HH2', 'HHE', 'HHN', 'EHE', 'EHN']

            for ch_code in horizontal_channels:
                try:
                    # ch_inv = inv.select(station=sta, channel=ch_code,
                    #                    starttime=pick.time - length,
                    #                    endtime=pick.time + length)
                    ch_inv = inv.select(station=sta, channel=ch_code,)
                    # print(f'    {sta} {ch_code}: inv={len(ch_inv)}')
                    if len(ch_inv) == 0:
                        continue
                    if len(ch_inv) == 0 and ch_code == horizontal_channels[-1]:
                        # Último canal intentado y ninguno funcionó
                        all_channels = [ch.code for net in inv for sta_obj in net 
                                       if sta_obj.code == sta for ch in sta_obj]
                        # print(f'    {sta}: all channels in inventory: {all_channels}')

                    test_trace = st_copy.select(station=sta, channel=ch_code).copy()
                    
                    # Intento fallback si no hay dato principal
                    if len(test_trace) == 0:
                        test_trace = try_adjacent_day_data(pick.time, sta, ch_code, force_adjacent=True)
                        if test_trace is None:
                            continue

                    s1 = pick.time - length
                    e1 = pick.time + length
                    windowed_trace = test_trace.trim(starttime=s1, endtime=e1).copy()
                    
                    # Intento fallback si el trim fue vacío
                    if len(windowed_trace) == 0:
                        test_trace_adj = try_adjacent_day_data(pick.time, sta, ch_code, force_adjacent=False)
                        if test_trace_adj is not None:
                            windowed_trace = test_trace_adj.trim(starttime=s1, endtime=e1)
                    
                    if len(windowed_trace) == 0:
                        continue

                    trace = windowed_trace[0]
                    sr = trace.stats.sampling_rate
                    expected_samples = int((e1 - s1) * sr)
                    actual_samples = len(trace.data)
                    if expected_samples == 0:
                        continue

                    # Criterio primario: ventana crítica ±0.5s alrededor del pick
                    critical_half = 0.5
                    critical_start_idx = max(0, int((length - critical_half) * sr))
                    critical_end_idx = min(actual_samples, int((length + critical_half) * sr))
                    critical_samples = critical_end_idx - critical_start_idx
                    expected_critical = int(2 * critical_half * sr)
                    critical_completeness = critical_samples / expected_critical if expected_critical > 0 else 0

                    critical_gaps = [g for g in windowed_trace.get_gaps()
                                    if g[4] < pick.time + critical_half
                                    and g[5] > pick.time - critical_half]

                    if critical_completeness < 0.9 or len(critical_gaps) > 0:
                        windowed_trace.clear()
                        del windowed_trace
                        continue

                    # Criterio secundario: calidad de toda la traza
                    completeness = actual_samples / expected_samples
                    gap_penalty = len(windowed_trace.get_gaps()) * 0.1
                    secondary_score = completeness - gap_penalty

                    is_better = (critical_completeness > best_critical_completeness or
                                (critical_completeness == best_critical_completeness and
                                 secondary_score > best_score))

                    if is_better:
                        best_score = secondary_score
                        best_critical_completeness = critical_completeness
                        if best_trace is not None:
                            best_trace.clear()
                        best_trace = windowed_trace.copy()
                        best_channel_code = ch_code

                    windowed_trace.clear()
                    del windowed_trace
                    test_trace.clear()
                    del test_trace

                except Exception:
                    continue

            if best_trace is not None:
                st2 += best_trace
                pick.waveform_id.channel_code = best_channel_code
                valid_picks += 1
                s_waves_optimized += 1
                best_trace.clear()
                del best_trace
            else:
                picks_lost_no_time_window += 1
                if debug_boundary_picks:
                    # Mostrar qué horizontales había disponibles
                    available_h = []
                    for ch_code in horizontal_channels:
                        tr_test = st_copy.select(station=sta, channel=ch_code)
                        if len(tr_test) > 0:
                            available_h.append(f"{ch_code}: {tr_test[0].stats.starttime} to {tr_test[0].stats.endtime}")
                    boundary_picks_lost.append({
                        'station': sta,
                        'phase': 'S',
                        'pick_time': pick.time,
                        'trim_window': f"{pick.time - length} to {pick.time + length}",
                        'available_traces': available_h,
                        'trim_result': 'no_suitable_horizontal'
                    })

        else:
            # Ondas P: lógica original
            st5 = st_copy.select(station=sta, channel=pick.waveform_id.channel_code).copy()
            
            # Intento fallback si no hay dato principal
            if len(st5) == 0:
                st_adj = try_adjacent_day_data(pick.time, sta, pick.waveform_id.channel_code, force_adjacent=True)
                if st_adj is not None:
                    st5 = st_adj.copy()
                else:
                    picks_lost_no_data += 1
                    continue
            
            s1 = pick.time - length
            e1 = pick.time + length
            st6 = st5.trim(starttime=s1, endtime=e1)
            
            if len(st6) == 0:
                # Intento fallback si el trim fue vacío (el pick puede estar en el límite del día)
                st_adj = try_adjacent_day_data(pick.time, sta, pick.waveform_id.channel_code, force_adjacent=False)
                if st_adj is not None:
                    st6 = st_adj.trim(starttime=s1, endtime=e1)
                
                    if len(st6) == 0:
                        if debug_boundary_picks:
                            trace_info = []
                            for tr in st5:
                                trace_info.append(f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}: "
                                                f"{tr.stats.starttime} to {tr.stats.endtime} "
                                                f"({len(tr.data)} samples)")
                            boundary_picks_lost.append({
                                'station': sta,
                                'phase': pick.phase_hint,
                                'pick_time': pick.time,
                                'trim_window': f"{s1} to {e1}",
                                'available_traces': trace_info,
                                'trim_result': 'empty_but_has_data' if len(st5) > 0 else 'empty_no_data'  # <-- distinguir casos
                            })
                        picks_lost_no_time_window += 1
                        st5.clear()
                        continue
            
            st2 += st6
            valid_picks += 1
            st5.clear()
            st6.clear()

    st_copy.clear()
    del st_copy

    # Merge y rellenar gaps
    st2.merge()
    if len(st2) > 0:
        gaps = st2.get_gaps()
        if len(gaps) > 0:
            st2.merge(method=1, fill_value=0)

    stats = {
        'valid_picks': valid_picks,
        'picks_lost_no_channel': picks_lost_no_channel,
        'picks_lost_no_data': picks_lost_no_data,
        'picks_lost_no_time_window': picks_lost_no_time_window,
        'picks_recovered_adjacent': picks_recovered_adjacent,
        's_waves_optimized': s_waves_optimized,
        'boundary_picks_lost': boundary_picks_lost
    }
    return st2, stats

def event_already_processed(event, rpnet_output_dir, station_groups=None, min_coverage=100):
    """
    Verifica si un evento ya tiene waveforms cortadas en el directorio de salida.
    Retorna True si el evento ya fue procesado con cobertura >= min_coverage (default 100%).
    Retorna False si necesita ser procesado o tiene cobertura incompleta.
    Si station_groups se proporciona, muestra estadísticas de cobertura.
    """
    try:
        origin = event.preferred_origin() or event.origins[0]
        origin_time = origin.time
    except (AttributeError, IndexError):
        try:
            origin_time = min([pick.time for pick in event.picks])
        except (AttributeError, ValueError):
            return False
    
    # Formats: YYYY_JDD_HHMMSS
    year = origin_time.year
    jday = origin_time.julday
    time_str = f"{origin_time.hour:02d}{origin_time.minute:02d}{origin_time.second:02d}"
    file_prefix = f"{year}_{jday:03d}_{time_str}"
    event_dir = join(rpnet_output_dir, file_prefix)
    
    # Verificar si el directorio existe y contiene archivos mseed
    if exists(event_dir):
        mseed_files = glob(join(event_dir, '*.mseed'))
        if len(mseed_files) > 0:
            # Siempre calcular cobertura
            stations_with_picks = set()
            for pick in event.picks:
                stations_with_picks.add(pick.waveform_id.station_code)
            
            # Extraer estaciones de los archivos mseed
            stations_processed = set()
            for mseed_file in mseed_files:
                station_name = os.path.basename(mseed_file).replace('.mseed', '')
                stations_processed.add(station_name)
            
            # Calcular estadísticas
            coverage = len(stations_processed) / len(stations_with_picks) * 100 if stations_with_picks else 0
            missing = stations_with_picks - stations_processed
            
            # Mostrar estadísticas si se proporciona station_groups
            if station_groups is not None:
                print(f'  Event {file_prefix}: {len(stations_processed)}/{len(stations_with_picks)} stations ({coverage:.1f}%)')
                if missing and len(missing) <= 10:
                    print(f'    Missing: {sorted(missing)}')
            
            # Solo considerar como "procesado" si tiene cobertura >= min_coverage
            if coverage < min_coverage:
                if station_groups is not None:
                    print(f'    WILL REPROCESS (coverage {coverage:.1f}% < {min_coverage}%)')
                return False
            
            if station_groups is not None:
                print(f'    Complete (coverage {coverage:.1f}%)')
            return True
    
    return False

def estimate_p_time(origin_time, event_lat, event_lon, event_dep_km, sta_lat, sta_lon, sta_elv_m,
                    model='/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.npz'):
    """
    Estima el tiempo de llegada de la onda P usando TauP cuando no hay pick de P.
    Intenta múltiples modelos como fallback.
    """
    from obspy.taup import TauPyModel
    from obspy.geodetics import locations2degrees
    
    # Modelos a intentar en orden
    models_to_try = [model]  # Primero el custom
    if os.path.exists(model):
        models_to_try = [model,'ak135','iasp91']
    else:
        models_to_try = ['ak135','iasp91']
    
    try:
        dist_deg = locations2degrees(event_lat, event_lon, sta_lat, sta_lon)
        depth_km = max(0, event_dep_km)

        for model_name in models_to_try:
            try:
                taup = TauPyModel(model=model_name)
                arrivals = taup.get_travel_times(source_depth_in_km=depth_km,
                                                 distance_in_degree=dist_deg,
                                                 phase_list=['P', 'Pg', 'Pn', 'p', 'pP', 'sP'])
                if arrivals:
                    p_travel_time = arrivals[0].time
                    return origin_time + p_travel_time, model_name
                else:
                    print(f'    TauPy ({model_name}): no arrivals for dist={dist_deg:.3f}° dep={depth_km:.1f}km')
            except Exception as e:
                print(f'    TauPy ({model_name}) error: {e}')
                continue
    except Exception as e:
        pass
    
    return None, None

def estimate_s_time(origin_time, event_lat, event_lon, event_dep_km, sta_lat, sta_lon, sta_elv_m,
                    model='/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.npz'):
    """
    Estima el tiempo de llegada de la onda S usando TauP cuando no hay pick de S.
    Intenta múltiples modelos como fallback.
    """
    from obspy.taup import TauPyModel
    from obspy.geodetics import locations2degrees
    
    # Modelos a intentar en orden
    models_to_try = [model]  # Primero el custom
    if os.path.exists(model):
        models_to_try = [model,'ak135','iasp91']
    else:
        models_to_try = ['ak135','iasp91']
    
    try:
        dist_deg = locations2degrees(event_lat, event_lon, sta_lat, sta_lon)
        depth_km = max(0, event_dep_km)

        for model_name in models_to_try:
            try:
                taup = TauPyModel(model=model_name)
                arrivals = taup.get_travel_times(source_depth_in_km=depth_km,
                                                 distance_in_degree=dist_deg,
                                                 phase_list=['S', 'Sg', 'Sn', 's', 'sS', 'pS'])
                if arrivals:
                    p_travel_time = arrivals[0].time
                    return origin_time + p_travel_time, model_name
                else:
                    print(f'    TauPy ({model_name}): no arrivals for dist={dist_deg:.3f}° dep={depth_km:.1f}km')
            except Exception as e:
                print(f'    TauPy ({model_name}) error: {e}')
                continue
    except Exception as e:
        pass
    
    return None, None


def cut_rpnet_streams(cat, st_day, inv, station_groups, rpnet_output_dir, jday, year,
                      pre_p=3.0, post_s=20.0,
                      taup_model='/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.npz'):
    """
    Corta ventanas desde p_time - pre_p hasta s_time + post_s por estación para RPNet.
    Incluye componente vertical (Z) y ambas horizontales.
    Usa estrategia GrowClust: busca en el Stream ya cargado por estación.
    Guarda un mseed por estación en rpnet_output_dir/YYYY_JDD_HHMMSS/STATION.mseed
    
    Parámetros:
    -----------
    cat : obspy.Catalog
        Catálogo con todos los eventos del día
    st_day : obspy.Stream
        Stream del día completo (pre-cargado)
    inv : obspy.Inventory
        Inventario de estaciones
    station_groups : dict
        Diccionario con estaciones organizadas por red
    rpnet_output_dir : str
        Directorio de salida para archivos MSEED
    jday : int o str
        Día del año (1-366)
    year : int o str
        Año
    pre_p : float
        Segundos antes del pick P
    post_s : float
        Segundos después del pick S
    taup_model : str
        Ruta al modelo de velocidad TauP
    """
    MORIA = station_groups['MORIA']
    DPRI = station_groups['DPRI']
    GEONET = station_groups['GEONET']
    
    jday_str = str(int(jday)).zfill(3)
    year_str = str(year)
    
    total_stations_written = 0
    total_stations_no_s = 0
    total_stations_failed = 0
    
    # Contadores de causas de fallo
    failure_reasons = {
        'no_inventory': 0,
        'distance_out_of_range': 0,
        'no_p_estimate': 0,
        'no_s_estimate': 0,
        'no_data_files': 0,
        'no_traces': 0,
        'no_z_component': 0,
        'z_incomplete': 0,
        'no_h_component': 0,
        'trim_failed': 0,
        'other_error': 0
    }

    failure_reasons.update({
        'p_estimated_npz': 0,
        'p_estimated_ak135': 0,
        'p_estimated_iaspei91': 0,
        'p_estimated_other': 0,
        's_estimated_npz': 0,
        's_estimated_ak135': 0,
        's_estimated_iaspei91': 0,
        's_estimated_other': 0,
    })
    
    if len(st_day) == 0:
        print(f'  No data in stream for day {jday}')
        return {
            'stations_written': 0,
            'stations_no_s': 0,
            'stations_failed': len(set([pick.waveform_id.station_code for event in cat for pick in event.picks]))
        }
    
    print(f'  Using pre-loaded stream with {len(st_day)} traces')
    
    for event in cat:
        try:
            origin = event.preferred_origin() or event.origins[0]
            event_lat = origin.latitude
            event_lon = origin.longitude
            event_dep_km = origin.depth / 1000.0
        except Exception:
            continue

        p_picks = {}
        s_picks = {}
        for pick in event.picks:
            sta = pick.waveform_id.station_code
            if pick.phase_hint == 'P':
                p_picks[sta] = pick
            elif pick.phase_hint == 'S':
                s_picks[sta] = pick
        print(f'  DEBUG {file_prefix if "file_prefix" in dir() else ""}: event.picks={len(event.picks)}, p_picks={len(p_picks)}, s_picks={len(s_picks)}')
        
        try:
            origin_time = event.preferred_origin().time or event.origins[0].time
        except AttributeError:
            origin_time = min([pick.time for pick in event.picks])

        time_str = f"{origin_time.hour:02d}{origin_time.minute:02d}{origin_time.second:02d}"
        file_prefix = f"{year_str}_{jday_str}_{time_str}"
        event_out_dir = join(rpnet_output_dir, file_prefix)
        os.makedirs(event_out_dir, exist_ok=True)

        stations_written = 0
        stations_no_s = 0
        stations_failed = 0
        failed_stations_detail = []

        evt_model_counters = {
            'p_estimated_npz': 0,
            'p_estimated_ak135': 0,
            'p_estimated_iasp91': 0,
            'p_estimated_other': 0,
            's_estimated_npz': 0,
            's_estimated_ak135': 0,
            's_estimated_iasp91': 0,
            's_estimated_other': 0,
        }

        all_picks_stations = set(p_picks.keys()) | set(s_picks.keys())

        for sta in sorted(all_picks_stations):
            in_moria = sta in MORIA
            in_dpri = sta in DPRI
            in_geonet = sta in GEONET
            if not any([in_moria, in_dpri, in_geonet]):
                print(f'    WARNING: {sta} not in any station group')
        
        for sta in sorted(all_picks_stations):
            p_estimated = False
            s_estimated = False
            try:
                # Inventario
                try:
                    sta_inv = inv.select(station=sta)[0][0]
                    sta_lat = sta_inv.latitude
                    sta_lon = sta_inv.longitude
                    sta_elv = sta_inv.elevation
                except Exception:
                    print(f'    DEBUG {sta}: inventory lookup failed: {e}')
                    failure_reasons['no_inventory'] += 1
                    stations_failed += 1
                    failed_stations_detail.append((sta, 'no_inventory'))
                    continue

                # Tiempo P
                if sta in p_picks:
                    p_time = p_picks[sta].time
                    p_model = None
                else:
                    try:
                        p_time, p_model = estimate_p_time(origin_time, event_lat, event_lon, event_dep_km,
                                                           sta_lat, sta_lon, sta_elv, model=taup_model)
                    except Exception as estim_err:
                        p_time = None
                        p_model = None
                        print(f'    DEBUG {sta}: estimate_p_time error: {str(estim_err)[:80]}')

                    if p_time is None:
                        failure_reasons['no_p_estimate'] += 1
                        stations_failed += 1
                        failed_stations_detail.append((sta, 'no_p_estimate'))
                        continue
                    p_estimated = True
                    try:
                        pm = str(p_model)
                        if 'npz' in pm:
                            failure_reasons['p_estimated_npz'] += 1
                            evt_model_counters['p_estimated_npz'] += 1
                        elif pm == 'ak135':
                            failure_reasons['p_estimated_ak135'] += 1
                            evt_model_counters['p_estimated_ak135'] += 1
                        elif pm == 'iaspei91':
                            failure_reasons['p_estimated_iaspei91'] += 1
                            evt_model_counters['p_estimated_iaspei91'] += 1
                        else:
                            failure_reasons['p_estimated_other'] += 1
                            evt_model_counters['p_estimated_other'] += 1
                    except Exception:
                        failure_reasons['p_estimated_other'] += 1

                # Tiempo S
                if sta in s_picks:
                    s_time = s_picks[sta].time
                    s_model = None
                else:
                    s_time, s_model = estimate_s_time(origin_time, event_lat, event_lon, event_dep_km,
                                                      sta_lat, sta_lon, sta_elv, model=taup_model)
                    if s_time is None:
                        failure_reasons['no_s_estimate'] += 1
                        stations_no_s += 1
                        failed_stations_detail.append((sta, 'no_s_estimate'))
                        continue
                    s_estimated = True
                    try:
                        sm = str(s_model)
                        if 'npz' in sm:
                            failure_reasons['s_estimated_npz'] += 1
                            evt_model_counters['s_estimated_npz'] += 1
                        elif sm == 'ak135':
                            failure_reasons['s_estimated_ak135'] += 1
                            evt_model_counters['s_estimated_ak135'] += 1
                        elif sm == 'iaspei91':
                            failure_reasons['s_estimated_iaspei91'] += 1
                            evt_model_counters['s_estimated_iaspei91'] += 1
                        else:
                            failure_reasons['s_estimated_other'] += 1
                            evt_model_counters['s_estimated_other'] += 1
                    except Exception:
                        failure_reasons['s_estimated_other'] += 1

                starttime = p_time - pre_p
                endtime = s_time + post_s

                # ===== BUSCAR DATOS EN EL STREAM O LEER BAJO DEMANDA =====
                st_sta = st_day.select(station=sta).copy()
                
                has_valid_data = any(len(tr.data) > 1 and (tr.stats.endtime - tr.stats.starttime) > 0.1 for tr in st_sta)
                
                if len(st_sta) == 0 or not has_valid_data:
                    if sta in MORIA:
                        net_label = 'MORIA'
                    elif sta in DPRI:
                        net_label = 'DPRI'
                    elif sta in GEONET:
                        net_label = 'GEONET'
                    else:
                        failure_reasons['other_error'] += 1
                        stations_failed += 1
                        continue
                    
                    from glob import glob
                    pattern = join('/Volumes/GeoPhysics_49/users-data/montalca/DATA', f'{year_str}/{net_label}/{sta}/*{year_str}.{jday_str}')
                    data_files = glob(pattern)
                    
                    if not data_files:
                        failure_reasons['no_data_files'] += 1
                        stations_failed += 1
                        failed_stations_detail.append((sta, f'no_data_files: pattern={pattern}'))
                        continue
                    
                    st_sta = Stream()
                    for file_path in data_files:
                        try:
                            st_tmp = read(file_path)
                            if len(st_tmp) > 0:
                                st_sta += st_tmp
                        except Exception:
                            pass
                
                if len(st_sta) == 0:
                    failure_reasons['no_traces'] += 1
                    stations_failed += 1
                    failed_stations_detail.append((sta, 'no_traces'))
                    continue

                # ===== COMPONENTE VERTICAL =====
                if sta in DPRI:
                    z_channels = ['EHZ']
                elif sta in MORIA:
                    z_channels = ['HHZ', 'EHZ', '*HZ']
                elif sta in GEONET:
                    z_channels = ['HHZ', 'EHZ']
                else:
                    z_channels = ['HHZ', 'EHZ', '*HZ']

                z_trace = Stream()
                for ch_code in z_channels:
                    traces = st_sta.select(station=sta, channel=ch_code).copy()
                    if len(traces) > 0:
                        valid_traces = Stream()
                        for tr in traces:
                            duration = tr.stats.endtime - tr.stats.starttime
                            if duration <= 0.1 or len(tr.data) <= 1:
                                continue
                            valid_traces += tr
                        if len(valid_traces) > 0:
                            z_trace += valid_traces
                            break
                
                if len(z_trace) == 0:
                    alt_z_channels = ['HH1', 'HH2', 'EH1', 'EH2']
                    for ch_code in alt_z_channels:
                        traces = st_sta.select(station=sta, channel=ch_code).copy()
                        if len(traces) > 0:
                            duration = traces[0].stats.endtime - traces[0].stats.starttime
                            if duration > 0.1 and len(traces[0].data) > 1:
                                z_trace += traces
                                break
                
                if len(z_trace) == 0:
                    if sta in MORIA:
                        net_label = 'MORIA'
                    elif sta in DPRI:
                        net_label = 'DPRI'
                    elif sta in GEONET:
                        net_label = 'GEONET'
                    else:
                        stations_failed += 1
                        st_sta.clear()
                        continue
                    
                    from glob import glob
                    year_str = str(int(year)).zfill(4)
                    jday_str = str(int(jday)).zfill(3)
                    pattern = join('/Volumes/GeoPhysics_49/users-data/montalca/DATA', f'{year_str}/{net_label}/{sta}/*{year_str}.{jday_str}')
                    data_files = glob(pattern)
                    
                    if data_files:
                        st_fallback = Stream()
                        for file_path in data_files:
                            try:
                                st_tmp = read(file_path)
                                if len(st_tmp) > 0:
                                    st_fallback += st_tmp
                            except Exception:
                                pass
                        
                        if len(st_fallback) > 0:
                            for ch_code in z_channels + alt_z_channels:
                                traces = st_fallback.select(station=sta, channel=ch_code).copy()
                                if len(traces) > 0:
                                    duration = traces[0].stats.endtime - traces[0].stats.starttime
                                    if duration > 0.1 and len(traces[0].data) > 1:
                                        z_trace += traces
                                        break

                if len(z_trace) == 0:
                    failure_reasons['no_z_component'] += 1
                    stations_failed += 1
                    failed_stations_detail.append((sta, 'no_z_component'))
                    st_sta.clear()
                    continue

                z_time_start = min(tr.stats.starttime for tr in z_trace)
                z_time_end = max(tr.stats.endtime for tr in z_trace)
                if (starttime < z_time_start) or (endtime > z_time_end):
                    print(f'    DEBUG {sta}: z_trace does not cover requested window')
                    print(f'      z_trace covers: {z_time_start} to {z_time_end}')
                    print(f'      Need: {starttime} to {endtime}')

                z_trace = z_trace.trim(starttime=starttime, endtime=endtime)
                
                if len(z_trace) == 0:
                    print(f'    DEBUG {sta}: trim_failed after trim.')
                    print(f'      Requested: {starttime} to {endtime}')
                    failure_reasons['trim_failed'] += 1
                    stations_failed += 1
                    failed_stations_detail.append((sta, f'trim_failed: start={starttime}, end={endtime}'))
                    st_sta.clear()
                    continue

                z_trace.merge(method=1, fill_value=0)

                # ===== COMPONENTES HORIZONTALES (AMBAS) =====
                if sta in DPRI:
                    horizontal_channels = ['EH1', 'EH2', 'EHE', 'EHN']
                elif sta in GEONET:
                    horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN']
                elif sta in MORIA:
                    horizontal_channels = ['HHE', 'HHN', 'HH1', 'HH2']
                else:
                    horizontal_channels = ['HHE', 'HHN', 'HH1', 'HH2', 'EHE', 'EHN', 'EH1', 'EH2']

                h_traces = Stream()  # Acumula todas las horizontales válidas

                for ch_code in horizontal_channels:
                    try:
                        ch_inv = inv.select(station=sta, channel=ch_code,
                                            starttime=starttime, endtime=endtime)
                        if len(ch_inv) == 0:
                            continue

                        test_trace = st_sta.select(station=sta, channel=ch_code).copy()
                        if len(test_trace) == 0:
                            continue

                        windowed_trace = test_trace.trim(starttime=starttime, endtime=endtime)
                        if len(windowed_trace) == 0:
                            windowed_trace.clear()
                            test_trace.clear()
                            continue

                        trace = windowed_trace[0]
                        if len(trace.data) <= 1:
                            windowed_trace.clear()
                            test_trace.clear()
                            continue

                        # Evitar duplicar canales ya añadidos
                        already_added = any(tr.stats.channel == ch_code for tr in h_traces)
                        if not already_added:
                            h_traces += windowed_trace.copy()

                        windowed_trace.clear()
                        del windowed_trace
                        test_trace.clear()
                        del test_trace

                    except Exception:
                        continue

                # ===== COMBINAR Y GUARDAR =====
                sta_st = z_trace.copy()
                if len(h_traces) > 0:
                    sta_st += h_traces
                    h_traces.clear()
                # Si no hay horizontales, se guarda solo Z

                out_file = join(event_out_dir, f'{sta}.mseed')
                sta_st.write(out_file, format='MSEED')
                stations_written += 1

                z_trace.clear()
                sta_st.clear()
                st_sta.clear()

            except Exception as e:
                failure_reasons['other_error'] += 1
                stations_failed += 1
                failed_stations_detail.append((sta, f'other_error: {str(e)[:60]}'))
                continue

        # Acumular estadísticas del evento
        total_stations_written += stations_written
        total_stations_no_s += stations_no_s
        total_stations_failed += stations_failed
        
        if len(failed_stations_detail) > 0:
            print(f'  Event {file_prefix}: {stations_written} stations written, '
                  f'{stations_no_s} S estimated, {stations_failed} failed')
            for sta, reason in failed_stations_detail[:10]:
                print(f'    FAILED {sta}: {reason}')
            if len(failed_stations_detail) > 10:
                print(f'    ... and {len(failed_stations_detail) - 10} more')
            try:
                any_p = sum(evt_model_counters[k] for k in evt_model_counters if k.startswith('p_'))
                any_s = sum(evt_model_counters[k] for k in evt_model_counters if k.startswith('s_'))
                if any_p > 0:
                    print(f"    P estimates by model: npz={evt_model_counters['p_estimated_npz']}, "
                          f"ak135={evt_model_counters['p_estimated_ak135']}, "
                          f"iaspei91={evt_model_counters['p_estimated_iasp91']}, "
                          f"other={evt_model_counters['p_estimated_other']}")
                if any_s > 0:
                    print(f"    S estimates by model: npz={evt_model_counters['s_estimated_npz']}, "
                          f"ak135={evt_model_counters['s_estimated_ak135']}, "
                          f"iaspei91={evt_model_counters['s_estimated_iasp91']}, "
                          f"other={evt_model_counters['s_estimated_other']}")
            except Exception:
                pass
        else:
            print(f'  Event {file_prefix}: {stations_written} stations written, '
                  f'{stations_no_s} S estimated, {stations_failed} failed')
            try:
                any_p = sum(evt_model_counters[k] for k in evt_model_counters if k.startswith('p_'))
                any_s = sum(evt_model_counters[k] for k in evt_model_counters if k.startswith('s_'))
                if any_p > 0:
                    print(f"    P estimates by model: npz={evt_model_counters['p_estimated_npz']}, "
                          f"ak135={evt_model_counters['p_estimated_ak135']}, "
                          f"iaspei91={evt_model_counters['p_estimated_iasp91']}, "
                          f"other={evt_model_counters['p_estimated_other']}")
                if any_s > 0:
                    print(f"    S estimates by model: npz={evt_model_counters['s_estimated_npz']}, "
                          f"ak135={evt_model_counters['s_estimated_ak135']}, "
                          f"iaspei91={evt_model_counters['s_estimated_iasp91']}, "
                          f"other={evt_model_counters['s_estimated_other']}")
            except Exception:
                pass

    return {
        'stations_written': total_stations_written,
        'stations_no_s': total_stations_no_s,
        'stations_failed': total_stations_failed,
        'failure_reasons': failure_reasons
    }


def process_single_day(jday, year, yearpath, nll_dir, inv, region,
                       station_groups, streams_output_dir, rpnet_output_dir):
    """
    Procesa un día: lee datos y genera streams para GrowClust y RPNet.
    """
    print(f'Processing day: {jday} (PID: {os.getpid()})')

    # Leer datos del día (compartido)
    cat, st = read_day_data(jday, year, yearpath, nll_dir, station_groups)
    if cat is None or st is None:
        return {
            'day': jday, 
            'events': 0, 
            'picks_processed': 0,
            'picks_valid': 0,
            'gaps_filled': 0,
            'events_with_gaps': 0,
            'picks_lost_no_channel': 0,
            'picks_lost_no_data': 0,
            'picks_lost_no_time_window': 0,
            's_waves_optimized': 0,
            'rpnet_stations_written': 0,
            'rpnet_stations_no_s': 0,
            'rpnet_stations_failed': 0,
            'success': False
        }

    num_events = len(cat)
    streams_file = join(streams_output_dir, f'{year}_{jday}.p')
    stream_dict = {}

    # Filtrar eventos por región y calidad antes de procesarlos
    events_to_process = []
    for event in cat:
        if len(event.picks) == 0:
            continue
        if not (region[2] < event.origins[0].latitude < region[3] and
                region[0] < event.origins[0].longitude < region[1]):
            continue

        depth_uncertainty = event.origins[0].depth_errors.uncertainty if event.origins[0].depth_errors else None
        if depth_uncertainty is not None and depth_uncertainty >= 1000000:
            continue

        # Verificar si el evento ya fue procesado (con cobertura completa)
        # if event_already_processed(event, rpnet_output_dir, station_groups=station_groups, min_coverage=75):
        #     continue
        # print(f'  DEBUG append: picks={len(event.picks)}')
        if event_already_processed(event, streams_output_dir, station_groups=station_groups, min_coverage=75):
            continue

        events_to_process.append(event)

    n_total = len(cat)
    n_region = sum(1 for e in cat 
                   if (region[2] < e.origins[0].latitude < region[3] and
                       region[0] < e.origins[0].longitude < region[1]))
    n_depth_ok = sum(1 for e in cat 
                     if (region[2] < e.origins[0].latitude < region[3] and
                         region[0] < e.origins[0].longitude < region[1]) and
                        (e.origins[0].depth_errors is None or 
                         e.origins[0].depth_errors.uncertainty is None or
                         e.origins[0].depth_errors.uncertainty < 1000000))
    print(f'Day {jday}: {n_total} total → {n_region} in region → {n_depth_ok} depth OK → {len(events_to_process)} to process')

    print(f'Day {jday}: Processing {len(events_to_process)} valid events')

    # Estadísticas (ahora con events_to_process definido)
    total_picks_processed = sum([len(e.picks) for e in events_to_process])
    total_picks_valid = 0
    total_gaps_filled = 0
    events_with_gaps = 0
    picks_lost_no_channel = 0
    picks_lost_no_data = 0
    picks_lost_no_time_window = 0
    s_waves_optimized = 0

    # from collections import Counter
    # pick_counts = Counter()
    # for i, event in enumerate(cat):
    #     n = len(event.picks)
    #     pick_counts[n] += 1
    #     if n == 0:
    #         try:
    #             ot = event.preferred_origin().time
    #         except Exception:
    #             ot = None
    #         print(f'  Event {i}: resource_id={event.resource_id}, picks=0, origin_time={ot}, n_origins={len(event.origins)}')
    
    # print(pick_counts)

    # Crear catálogo con solo los eventos a procesar
    cat_filtered = Catalog(events_to_process)
    # print(f'  DEBUG cat_filtered: n={len(cat_filtered)}, picks_event0={len(cat_filtered.events[0].picks) if len(cat_filtered)>0 else "NA"}')

    # === RPNet: procesa el catálogo completo usando el Stream ya cargado ===
    # rpnet_stats = cut_rpnet_streams(cat_filtered, st, inv, station_groups, rpnet_output_dir, 
    #                                 jday, year)

    # === GrowClust: procesa cada evento y guarda streams en pickle ===
    for event in events_to_process:
        try:
            # Habilitar debug para un evento con muchos picks perdidos
            debug_this_event = True if len(events_to_process) > 0 else False
            st_event, gc_stats = cut_growclust_streams(event, st, inv, station_groups, 
                                                        jday=jday, year=year, yearpath=yearpath,
                                                        length=3, debug_boundary_picks=debug_this_event)
            if st_event is not None and len(st_event) > 0:
                # Usar identificador único del evento
                try:
                    origin_time = event.preferred_origin().time or event.origins[0].time
                except:
                    origin_time = min([pick.time for pick in event.picks])
                
                try:
                    origin_time = event.preferred_origin().time or event.origins[0].time
                except:
                    origin_time = min([pick.time for pick in event.picks])

                event_key = f"{origin_time.year}_{origin_time.julday:03d}_{origin_time.hour:02d}{origin_time.minute:02d}{origin_time.second:02d}"
                stream_dict[event_key] = st_event
                # event_key = str(event.resource_id)
                # stream_dict[event_key] = st_event

                # Acumular estadísticas
                total_picks_valid += gc_stats['valid_picks']
                picks_lost_no_channel += gc_stats['picks_lost_no_channel']
                picks_lost_no_data += gc_stats['picks_lost_no_data']
                picks_lost_no_time_window += gc_stats['picks_lost_no_time_window']
                s_waves_optimized += gc_stats['s_waves_optimized']
                total_gaps_filled += gc_stats.get('gaps_filled', 0)
                
                # Debug: mostrar picks perdidos en boundary
                if gc_stats.get('boundary_picks_lost') and len(gc_stats['boundary_picks_lost']) > 0:
                    print(f"\n  DEBUG Event {event_key}: {len(gc_stats['boundary_picks_lost'])} boundary picks lost:")
                    for i, lost_pick in enumerate(gc_stats['boundary_picks_lost'][:3]):  # Mostrar solo los 3 primeros
                        print(f"    Pick {i+1}: {lost_pick['station']} {lost_pick['phase']}")
                        print(f"      Pick time: {lost_pick['pick_time']}")
                        print(f"      Trim window: {lost_pick['trim_window']}")
                        print(f"      Available traces: {lost_pick['available_traces']}")
        # except Exception as e:
        #     continue
        except Exception as e:
            import traceback
            print(f'  ERROR event {event.resource_id}: {e}')
            traceback.print_exc()
            continue

    # Guardar streams en pickle
    # if len(stream_dict) > 0:
    #     os.makedirs(streams_output_dir, exist_ok=True)
    #     with open(streams_file, 'wb') as f:
    #         pickle.dump(stream_dict, f)
    #     print(f'Day {jday}: Saved {len(stream_dict)} GrowClust event streams to {streams_file}')
    # Guardar streams en pickle (merge con existentes si el archivo ya existe)
    if len(stream_dict) > 0:
        os.makedirs(streams_output_dir, exist_ok=True)
        
        # Cargar pickle existente si existe
        existing_streams = {}
        if exists(streams_file):
            try:
                with open(streams_file, 'rb') as f:
                    existing_streams = pickle.load(f)
                print(f'Day {jday}: Loaded {len(existing_streams)} existing streams from pickle')
            except Exception as e:
                print(f'Day {jday}: Could not load existing pickle: {e}')
        
        # Merge: los nuevos streams tienen prioridad sobre los existentes
        existing_streams.update(stream_dict)
        
        with open(streams_file, 'wb') as f:
            pickle.dump(existing_streams, f)
        print(f'Day {jday}: Saved {len(existing_streams)} total streams ({len(stream_dict)} new) to {streams_file}')

    # Limpiar memoria
    cat.clear()
    st.clear()
    stream_dict.clear()
    del stream_dict
    del cat_filtered
    gc.collect()

    return {
        'day': jday,
        'events': len(events_to_process),
        'picks_processed': total_picks_processed,
        'picks_valid': total_picks_valid,
        'gaps_filled': total_gaps_filled,
        'events_with_gaps': events_with_gaps,
        'picks_lost_no_channel': picks_lost_no_channel,
        'picks_lost_no_data': picks_lost_no_data,
        'picks_lost_no_time_window': picks_lost_no_time_window,
        's_waves_optimized': s_waves_optimized,
        # 'rpnet_stations_written': rpnet_stats['stations_written'],
        # 'rpnet_stations_no_s': rpnet_stats['stations_no_s'],
        # 'rpnet_stations_failed': rpnet_stats['stations_failed'],
        # 'failure_reasons': rpnet_stats.get('failure_reasons', {}),
        'success': True
    }


def main():
    parser = argparse.ArgumentParser(description='Process seismic data for a single day')
    parser.add_argument('--year', type=str, required=True)
    parser.add_argument('--jday', type=str, required=True)
    parser.add_argument('--basedir', type=str, required=True)
    parser.add_argument('--datadir', type=str, required=True)
    parser.add_argument('--region', type=float, nargs=4,
                        default= [171.2,176.1,-43.8,-39.3])
                        # default=[171.2, 176.1, -43.8, -39.3])
    parser.add_argument('--taup_model', type=str,
                    default='/Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/VEL_MODEL/transition_zone_vmodel.npz',
                    help='Path to TauPy velocity model (.npz)')
    args = parser.parse_args()

    codestart = datetime.datetime.now()

    basedir = args.basedir
    ctlgdir = join(basedir, 'CATALOGS')
    nll_dir = join(ctlgdir, 'NLL')
    growclust_dir = join(ctlgdir, 'GROWCLUST')
    rpnet_dir = join(ctlgdir, 'RPNET')
    sta_dir = join(basedir, 'STATIONS')
    # inv = read_inventory(join(sta_dir, 'nll_region_all_stations.xml'))
    inv = read_inventory(join(sta_dir, 'ALL_STATIONS.xml'))

    yearpath = join(args.datadir, args.year)

    if not exists(yearpath):
        print(f'Error: Data directory does not exist: {yearpath}')
        sys.exit(1)
    if not exists(join(nll_dir, args.year)):
        print(f'Error: NLL catalog directory does not exist')
        sys.exit(1)

    station_groups = organize_stations_by_network(inv)
    print(f'Station groups: MORIA={len(station_groups["MORIA"])}, '
          f'DPRI={len(station_groups["DPRI"])}, GEONET={len(station_groups["GEONET"])}')

    streams_output_dir = join(growclust_dir, 'STREAMS')
    rpnet_output_dir = join(rpnet_dir, 'WAVEFORMS')
    os.makedirs(streams_output_dir, exist_ok=True)
    os.makedirs(rpnet_output_dir, exist_ok=True)

    result = process_single_day(
        args.jday, args.year, yearpath, nll_dir, inv,
        args.region, station_groups, streams_output_dir, rpnet_output_dir
    )

    codestop = datetime.datetime.now()
    print(f'\n--- DAY {args.jday} SUMMARY ---')
    print(f'Success: {result.get("success", False)}')
    print(f'Events processed: {result.get("events", 0)}')
    print(f'Picks processed: {result.get("picks_processed", 0)}')
    print(f'Valid picks: {result.get("picks_valid", 0)}')
    picks_processed = result.get("picks_processed", 0)
    picks_valid = result.get("picks_valid", 0)
    if picks_processed > 0:
        print(f'Pick success rate: {picks_valid/picks_processed*100:.1f}%')
    print(f'\nGAP FILLING: {result.get("gaps_filled", 0)} gaps in {result.get("events_with_gaps", 0)} events')
    print(f'S-WAVE OPTIMIZATION: {result.get("s_waves_optimized", 0)} components selected')
    total_lost = (result.get("picks_lost_no_channel", 0) + result.get("picks_lost_no_data", 0) +
                  result.get("picks_lost_no_time_window", 0))
    print(f'\nPICKS LOST: {total_lost} total')
    print(f'  No channel: {result.get("picks_lost_no_channel", 0)}')
    print(f'  No data: {result.get("picks_lost_no_data", 0)}')
    print(f'  No time window: {result.get("picks_lost_no_time_window", 0)}')
    
    # Análisis detallado de picks perdidos
    no_channel = result.get("picks_lost_no_channel", 0)
    no_data = result.get("picks_lost_no_data", 0)
    no_time_window = result.get("picks_lost_no_time_window", 0)
    total_lost = no_channel + no_data + no_time_window
    
    if total_lost > 0:
        print(f'\nPICKS LOST ANALYSIS:')
        if no_time_window > 0:
            pct = (no_time_window/total_lost)*100
            print(f'  {pct:.1f}% ({no_time_window}): Pick outside MSEED time range (too early/late in day)')
        if no_data > 0:
            pct = (no_data/total_lost)*100
            print(f'  {pct:.1f}% ({no_data}): MSEED file not available for station')
        if no_channel > 0:
            pct = (no_channel/total_lost)*100
            print(f'  {pct:.1f}% ({no_channel}): Channel code not found in inventory')
    
    # RPNet is currently disabled - uncomment to enable
    print(f'\nRPNet WAVEFORMS:')
    print(f'  Stations written: {result.get("rpnet_stations_written", 0)}')
    print(f'  S estimated by TauP: {result.get("rpnet_stations_no_s", 0)}')
    print(f'  Failed: {result.get("rpnet_stations_failed", 0)}')
    
    print(f'\nProcessing time: {codestop - codestart}')

    sys.exit(0 if result["success"] else 1)


if __name__ == '__main__':
    main()