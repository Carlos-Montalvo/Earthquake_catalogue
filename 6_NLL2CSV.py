import glob
import pandas as pd
from os.path import join
from datetime import datetime
import os
from obspy import UTCDateTime, read_inventory
from obspy.core.event import Catalog, Event, Origin, Magnitude, OriginQuality, Pick, WaveformStreamID

# Organizes stations by network from ObsPy inventory
def organize_stations_by_network(inventory):
    """
    Extract and organize stations by network from ObsPy inventory.
    Returns dictionaries with stations grouped by data format/location.
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
            
            # Classify stations based on network code and station patterns
            if net_code == '5L':
                stations_by_network['MORIA'].append(sta_code)
            elif net_code == 'DP':
                stations_by_network['DPRI'].append(sta_code)
            elif net_code == 'NZ':
                stations_by_network['GEONET'].append(sta_code)
    
    return stations_by_network

### DIRECTORIES AND FILES ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
out_dir = join(basedir, 'CATALOGS')
nll_out = join(out_dir, 'NLL')
nll_dir = join(basedir, 'NLL')
hyp_files = glob.glob(join(nll_dir,'OUT_JAN25_SEP25/DATA_2025_*.loc.hyp'))

# Define geographic region filter [lon_min, lon_max, lat_min, lat_max]
loc_region = [171, 176.6, -44.3, -39.3]
print(f"Geographic filter applied: Longitude {loc_region[0]}° to {loc_region[1]}° E, Latitude {loc_region[2]}° to {loc_region[3]}° S")

# Load station inventory and organize by network
sta_dir = join(basedir,'STATIONS')
inv = read_inventory(join(sta_dir,'nll_region_all_stations.xml'))
station_groups = organize_stations_by_network(inv)

print(f"Loaded station inventory:")
print(f"  MORIA stations: {len(station_groups['MORIA'])} (e.g., {', '.join(station_groups['MORIA'][:5])})")
print(f"  DPRI stations: {len(station_groups['DPRI'])} (e.g., {', '.join(station_groups['DPRI'][:5])})")
print(f"  GEONET stations: {len(station_groups['GEONET'])} (e.g., {', '.join(station_groups['GEONET'][:5])})")
print(f"  Total stations: {sum(len(v) for v in station_groups.values())}")
print()

### CONVERSION
events = []
event_dates = []  # Para almacenar todas las fechas de eventos
events_outside_region = 0  # Contador de eventos fuera de la región

for f in hyp_files:
    with open(f) as fh:
        ev = {"file": f}
        picks = []  # Lista para almacenar los picks de este evento
        reading_phases = False  # Flag para saber si estamos leyendo la sección de fases
        skip_event = False  # Flag para saltar eventos fuera de la región
        
        for line in fh:
            if line.startswith("GEOGRAPHIC"):
                parts = line.split()
                # OT 2022 01 11  21 25 8.768092
                year, month, day = parts[2], parts[3], parts[4]
                hour, minute, sec = parts[5], parts[6], parts[7]
                ev["datetime"] = f"{year}-{month}-{day} {hour}:{minute}:{sec}"
                ev["latitude"]  = float(parts[9])
                ev["longitude"] = float(parts[11])
                ev["depth_km"]  = float(parts[13])
                
                # Check if event is within the specified region
                lat, lon = ev["latitude"], ev["longitude"]
                if not (loc_region[0] <= lon <= loc_region[1] and loc_region[2] <= lat <= loc_region[3]):
                    events_outside_region += 1
                    # Mark event to be skipped
                    skip_event = True
                    break
                
                # Almacenar la fecha para determinar el rango temporal
                event_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                event_dates.append(event_date)
                
            elif line.startswith("QUALITY"):
                parts = line.split()
                # RMS ... Nphs ... Gap ... Dist ...
                if "RMS" in parts:
                    ev["rms"] = float(parts[parts.index("RMS")+1])
                if "Nphs" in parts:
                    ev["nphases"] = int(float(parts[parts.index("Nphs")+1]))
                if "Gap" in parts:
                    ev["gap"] = float(parts[parts.index("Gap")+1])
                if "Dist" in parts:
                    ev["dist_km"] = float(parts[parts.index("Dist")+1])
                    
            elif line.startswith("PHASE ID"):
                # Empezamos a leer la sección de fases
                reading_phases = True
                continue
                
            elif line.startswith("END_PHASE"):
                # Terminamos de leer las fases
                reading_phases = False
                continue
                
            elif reading_phases and not line.strip() == "":
                # Parsear línea de pick
                try:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        station = parts[0]
                        phase = parts[4]
                        # Fix: Assign appropriate default channel based on phase type
                        if parts[2] != '?':
                            channel = parts[2]
                        else:
                            # For P waves: use vertical channel (Z)
                            # For S waves: use horizontal channel (E as default)
                            if phase == 'P':
                                channel = 'EHZ'  # Vertical for P waves
                            elif phase == 'S':
                                channel = 'EHE'  # Horizontal for S waves (will be refined later based on station type)
                            else:
                                channel = 'EHZ'  # Fallback
                        pick_date = parts[6]
                        pick_hrmn = parts[7]
                        pick_sec = parts[8]
                        
                        # Construir tiempo del pick
                        # pick_date formato: 20220115, pick_hrmn formato: 1955
                        pick_year = pick_date[:4]
                        pick_month = pick_date[4:6]
                        pick_day = pick_date[6:8]
                        pick_hour = pick_hrmn[:2]
                        pick_minute = pick_hrmn[2:4]
                        
                        pick_datetime = f"{pick_year}-{pick_month}-{pick_day} {pick_hour}:{pick_minute}:{pick_sec}"
                        
                        pick_info = {
                            'station': station,
                            'channel': channel,
                            'phase': phase,
                            'datetime': pick_datetime
                        }
                        picks.append(pick_info)
                except Exception as e:
                    # Si hay error parseando el pick, continuar
                    pass
        
        # Agregar picks al evento
        ev["picks"] = picks
        
        # Only add event if it's within the region (skip_event flag not set)
        if not skip_event:
            events.append(ev)

df = pd.DataFrame(events)

print(f"\nRegion filtering results:")
print(f"Events outside region [{loc_region[0]}, {loc_region[1]}, {loc_region[2]}, {loc_region[3]}]: {events_outside_region}")
print(f"Events within region: {len(events)}")
print(f"Total events processed: {len(events) + events_outside_region}")

# Generar archivos XML por día
print("\nGenerating daily XML files...")

# Agrupar eventos por día
events_by_day = {}
negative_time_count = 0  # Contador de eventos con tiempos negativos
parsing_error_count = 0  # Contador de otros errores de parsing

for event in events:
    if 'datetime' in event:
        # Manejar microsegundos en el formato de fecha
        datetime_str = event['datetime']
        
        # Verificar si hay segundos negativos y corregir
        if ':-' in datetime_str:
            negative_time_count += 1
            continue
            
        try:
            # Intentar con microsegundos primero
            event_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            try:
                # Si falla, intentar sin microsegundos
                event_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                parsing_error_count += 1
                continue
        
        year = event_dt.strftime("%Y")
        jday = event_dt.strftime("%j")  # Día juliano
        day_key = f"{year}_{jday}"
        
        if day_key not in events_by_day:
            events_by_day[day_key] = []
        events_by_day[day_key].append(event)

# Reportar estadísticas de parsing
print(f"Events with negative time: {negative_time_count}")
print(f"Events with parsing errors: {parsing_error_count}")
print(f"Valid events processed: {len([e for day_events in events_by_day.values() for e in day_events])}")
print(f"Total events in input: {len(events)}")
print(f"Events passed to XML creation: {sum(len(day_events) for day_events in events_by_day.values())}")

# Crear directorio por año si no existe
xml_negative_time_count = 0  # Contador adicional para la fase de XML
xml_parsing_error_count = 0

# Contadores para fixes aplicados
network_fixes_count = 0
channel_fixes_count = 0

for day_key, day_events in events_by_day.items():
    year = day_key.split('_')[0]
    year_dir = join(nll_out, year)
    
    # Crear directorio del año si no existe
    if not os.path.exists(year_dir):
        os.makedirs(year_dir)
        print(f"Created directory: {year_dir}")
    
    # Crear catálogo de ObsPy para el día
    daily_catalog = Catalog()
    
    for event_data in day_events:
        # Crear evento de ObsPy
        event = Event()
        
        # Crear origen
        origin = Origin()
        # Manejar microsegundos en el tiempo de origen
        datetime_str = event_data['datetime']
        
        # Verificar si hay segundos negativos y corregir
        if ':-' in datetime_str:
            xml_negative_time_count += 1
            continue
            
        try:
            # Intentar con microsegundos primero
            origin.time = UTCDateTime(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f"))
        except ValueError:
            try:
                # Si falla, intentar sin microsegundos
                origin.time = UTCDateTime(datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                xml_parsing_error_count += 1
                continue
        
        origin.latitude = event_data['latitude']
        origin.longitude = event_data['longitude']
        origin.depth = event_data['depth_km'] * 1000  # Convertir a metros
        
        # Inicializar OriginQuality
        origin.quality = OriginQuality()
        
        # Agregar calidad si está disponible
        if 'rms' in event_data:
            origin.time_errors.uncertainty = event_data['rms']
        if 'nphases' in event_data:
            origin.quality.used_phase_count = event_data['nphases']
        if 'gap' in event_data:
            origin.quality.azimuthal_gap = event_data['gap']
        if 'dist_km' in event_data:
            origin.quality.minimum_distance = event_data['dist_km'] * 1000  # Convertir a metros
        
        # Agregar origen al evento
        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id
        
        # Agregar picks al evento
        if 'picks' in event_data:
            for pick_data in event_data['picks']:
                try:
                    # Crear pick de ObsPy
                    pick = Pick()
                    
                    # Parsear tiempo del pick
                    pick_datetime_str = pick_data['datetime']
                    try:
                        pick.time = UTCDateTime(datetime.strptime(pick_datetime_str, "%Y-%m-%d %H:%M:%S.%f"))
                    except ValueError:
                        try:
                            pick.time = UTCDateTime(datetime.strptime(pick_datetime_str, "%Y-%m-%d %H:%M:%S"))
                        except ValueError:
                            continue  # Saltar pick si no se puede parsear el tiempo
                    
                    # Crear WaveformStreamID
                    waveform_id = WaveformStreamID()
                    waveform_id.station_code = pick_data['station']
                    
                    # Get station lists from inventory (loaded at startup)
                    moria_stations = station_groups['MORIA']
                    dpri_stations = station_groups['DPRI']
                    geonet_stations = station_groups['GEONET']
                    
                    # Fix 1: Assign correct network code
                    original_network = 'NZ'  # Default that would have been assigned
                    if pick_data['station'] in moria_stations:
                        waveform_id.network_code = '5L'
                    elif pick_data['station'] in dpri_stations:
                        waveform_id.network_code = 'DP'  # Correct network for DPRI stations
                        if original_network != 'DP':
                            network_fixes_count += 1
                    elif pick_data['station'].startswith('DP'):
                        waveform_id.network_code = 'DP'
                    elif pick_data['station'] in geonet_stations:
                        waveform_id.network_code = 'NZ'  # GeoNet stations
                    else:
                        waveform_id.network_code = 'NZ'  # Default fallback
                    
                    # Fix 2: Assign correct channel based on phase, station type, and inventory
                    original_channel = pick_data['channel']
                    station = pick_data['station']
                    phase = pick_data['phase']
                    
                    # Get available channels from inventory for this station
                    available_channels = []
                    for network in inv:
                        if network.code == waveform_id.network_code:
                            for sta in network:
                                if sta.code == station:
                                    available_channels = [ch.code for ch in sta.channels]
                                    break
                    
                    if station in dpri_stations:
                        # DPRI stations: P->EHZ, S->EH1/EH2
                        if phase == 'P':
                            waveform_id.channel_code = 'EHZ'
                        elif phase == 'S':
                            # Use EH1 for S waves (horizontal component)
                            waveform_id.channel_code = 'EH1'
                            if original_channel == 'EHZ':  # S wave was on vertical channel
                                channel_fixes_count += 1
                        else:
                            waveform_id.channel_code = original_channel
                    elif waveform_id.network_code == 'NZ':
                        # GeoNet stations: use channels based on what's available in inventory
                        if phase == 'P':
                            # For P waves, prefer vertical channels
                            if 'HHZ' in available_channels:
                                waveform_id.channel_code = 'HHZ'
                            elif 'EHZ' in available_channels:
                                waveform_id.channel_code = 'EHZ'
                            else:
                                waveform_id.channel_code = 'HHZ'  # Default fallback
                        elif phase == 'S':
                            # For S waves, prefer horizontal channels
                            if 'HHE' in available_channels:
                                waveform_id.channel_code = 'HHE'  # Broadband horizontal
                            elif 'EHE' in available_channels:
                                waveform_id.channel_code = 'EHE'  # Short period horizontal
                            else:
                                waveform_id.channel_code = 'HHE'  # Default fallback
                            if original_channel.endswith('Z'):  # S wave was on vertical channel
                                channel_fixes_count += 1
                        else:
                            waveform_id.channel_code = original_channel
                    else:
                        # Other networks: keep original channel
                        waveform_id.channel_code = original_channel
                    
                    pick.waveform_id = waveform_id
                    pick.phase_hint = pick_data['phase']
                    
                    # Agregar pick al evento
                    event.picks.append(pick)
                    
                except Exception as e:
                    # Si hay error creando el pick, continuar
                    continue
        
        # Agregar evento al catálogo
        daily_catalog.append(event)
    
    # Guardar archivo XML del día
    xml_filename = f"{day_key}_nll.xml"
    xml_path = join(year_dir, xml_filename)
    daily_catalog.write(xml_path, format="QUAKEML")
    
    print(f"Saved {len(daily_catalog)} events to {xml_path}")

print(f"\nGenerated XML files for {len(events_by_day)} days")
print(f"Additional events skipped during XML creation:")
print(f"  - Events with negative time: {xml_negative_time_count}")
print(f"  - Events with parsing errors: {xml_parsing_error_count}")

print(f"\nFixes applied during XML creation:")
print(f"  - Network code corrections (NZ->DP for DPRI stations): {network_fixes_count}")
print(f"  - Channel corrections (S waves from Z to horizontal): {channel_fixes_count}")
if network_fixes_count > 0 or channel_fixes_count > 0:
    print("✅ Catalog files have been automatically corrected for cross-correlation compatibility!")

df = pd.DataFrame(events)

# Determinar el rango temporal de los eventos para el nombre del catálogo
if event_dates:
    min_date = min(event_dates)
    max_date = max(event_dates)
    
    # Formatear el nombre del catálogo según el formato solicitado
    YYYY = min_date.strftime("%Y")
    SM = min_date.strftime("%m")
    SD = min_date.strftime("%d")
    EM = max_date.strftime("%m")
    ED = max_date.strftime("%d")
    
    catalog_name = f'nll_catalog_{YYYY}_{SM}_{SD}_{EM}_{ED}.csv'
else:
    # Usar nombre por defecto si no hay eventos
    catalog_name = 'nll_catalog.csv'

catalog = f'{out_dir}/{catalog_name}'
df.to_csv(catalog, index=False)
print(f"Saved {len(df)} events to {catalog}")
print(f"Catalog covers period from {min_date.strftime('%Y-%m-%d') if event_dates else 'N/A'} to {max_date.strftime('%Y-%m-%d') if event_dates else 'N/A'}")
