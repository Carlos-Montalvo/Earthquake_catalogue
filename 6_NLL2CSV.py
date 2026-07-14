import glob
import pandas as pd
from os.path import join
from datetime import datetime
import os
from obspy import UTCDateTime, read_inventory, read_events, Catalog
from obspy.core.event import OriginUncertainty, QuantityError


def organize_stations_by_network(inventory):
    stations_by_network = {'MORIA': [], 'DPRI': [], 'GEONET': []}
    for network in inventory:
        net_code = network.code
        for station in network:
            sta_code = station.code
            if net_code == '5L':
                stations_by_network['MORIA'].append(sta_code)
            elif net_code == 'DP':
                stations_by_network['DPRI'].append(sta_code)
            elif net_code == 'NZ':
                stations_by_network['GEONET'].append(sta_code)
    return stations_by_network


def parse_nll_hyp_summary(hyp_path):
    """
    Lee GEOGRAPHIC, QUALITY, STATISTICS y QML_OriginUncertainty directo del
    texto del .hyp, para tener todos los campos del catálogo CSV en las
    unidades nativas de NLL (evita ambigüedades de unidades al pasar por
    los objetos OriginQuality de ObsPy/QuakeML).
    """
    summary = {}
    stats = {}
    qml_unc = {}

    with open(hyp_path, 'r') as fh:
        for line in fh:
            if line.startswith('QUALITY'):
                parts = line.split()
                if 'RMS' in parts:
                    summary['rms'] = float(parts[parts.index('RMS') + 1])
                if 'Nphs' in parts:
                    summary['nphases'] = int(float(parts[parts.index('Nphs') + 1]))
                if 'Gap' in parts:
                    summary['gap'] = float(parts[parts.index('Gap') + 1])
                if 'Dist' in parts:
                    summary['dist_km'] = float(parts[parts.index('Dist') + 1])
            elif line.startswith('STATISTICS'):
                parts = line.split()
                stats = dict(zip(parts[1::2], parts[2::2]))
            elif line.startswith('QML_OriginUncertainty'):
                parts = line.split()
                qml_unc = dict(zip(parts[1::2], parts[2::2]))
            elif line.startswith('PHASE ID'):
                break

    if 'ZZ' in stats:
        try:
            summary['vert_uncert_km'] = float(stats['ZZ']) ** 0.5
        except (ValueError, TypeError):
            pass

    hor_unc = qml_unc.get('horUnc')
    min_hor = qml_unc.get('minHorUnc')
    max_hor = qml_unc.get('maxHorUnc')
    try:
        if hor_unc is not None and float(hor_unc) > 0:
            summary['horz_uncert_km'] = float(hor_unc)
        elif min_hor is not None and max_hor is not None:
            summary['horz_uncert_km'] = (float(min_hor) * float(max_hor)) ** 0.5
    except (ValueError, TypeError):
        pass

    return summary


def parse_nll_phase_geometry(hyp_path):
    """
    Lee RAz (azimut del rayo en la fuente) directo de las líneas PHASE,
    indexado por (estación, fase, hora del pick) para poder emparejar
    con cada Pick/Arrival del objeto ObsPy.
    """
    geometry = {}
    in_phase_block = False
    with open(hyp_path, 'r') as fh:
        for line in fh:
            if line.startswith('PHASE ID'):
                in_phase_block = True
                continue
            if line.startswith('END_PHASE'):
                in_phase_block = False
                continue
            if in_phase_block and line.strip():
                parts = line.split()
                if len(parts) < 26:
                    continue
                sta = parts[0]
                phase = parts[4]
                pick_hrmn = parts[7]
                pick_sec = parts[8]
                r_az = float(parts[23])   # RAz
                r_dip = float(parts[24])  # RDip (ya coincide con lo que da ObsPy)
                key = (sta, phase, pick_hrmn, pick_sec)
                geometry[key] = {'RAz': r_az, 'RDip': r_dip}
    return geometry


### DIRECTORIES AND FILES ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
out_dir = join(basedir, 'CATALOGS')
nll_out = join(out_dir, 'NLL')
nll_dir = join(basedir, 'NLL')
hyp_files = glob.glob(join(nll_dir, 'OUT_JAN24_DEC24/DATA_2024_*.loc.hyp'))

# Filtrar archivos vacíos o casi vacíos (residuos de corridas interrumpidas)
valid_hyp_files = []
n_empty_skipped = 0
for f in hyp_files:
    if os.path.getsize(f) == 0:
        n_empty_skipped += 1
        continue
    valid_hyp_files.append(f)

hyp_files = valid_hyp_files
print(f"Archivos vacíos descartados: {n_empty_skipped}")
print(f"Total archivos a procesar: {len(hyp_files)}")

loc_region = [171, 176.6, -44.3, -39.3]
print(f"Geographic filter applied: Longitude {loc_region[0]}° to {loc_region[1]}° E, Latitude {loc_region[2]}° to {loc_region[3]}° S")

sta_dir = join(basedir, 'STATIONS')
inv = read_inventory(join(sta_dir, 'ALL_STATIONS.xml'))
station_groups = organize_stations_by_network(inv)
moria_stations = station_groups['MORIA']
dpri_stations = station_groups['DPRI']
geonet_stations = station_groups['GEONET']

print("Loaded station inventory:")
print(f"  MORIA: {len(moria_stations)}  DPRI: {len(dpri_stations)}  GEONET: {len(geonet_stations)}")

### CONVERSION ###
events_by_day = {}
events_outside_region = 0
n_parse_errors = 0
n_missing_horz_uncert = 0
n_missing_vert_uncert = 0
n_geom_matched = 0
n_geom_unmatched = 0
n_missing_arrival_geom = 0
network_fixes_count = 0
channel_fixes_count = 0

sanity_check_done = False

event_records = []
event_dates = []

for f in hyp_files:
    try:
        cat = read_events(f)
    except Exception as e:
        print(f"  ERROR leyendo {f}: {e}")
        n_parse_errors += 1
        continue

    summary = parse_nll_hyp_summary(f)
    geom = parse_nll_phase_geometry(f)

    for event in cat:
        origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
        if origin is None or origin.latitude is None or origin.longitude is None:
            continue

        if not (loc_region[0] <= origin.longitude <= loc_region[1] and
                loc_region[2] <= origin.latitude <= loc_region[3]):
            events_outside_region += 1
            continue

        # --- FIX: horz/vert uncertainty leídos directo del .hyp ---
        if 'horz_uncert_km' in summary:
            if origin.origin_uncertainty is None:
                origin.origin_uncertainty = OriginUncertainty()
            origin.origin_uncertainty.horizontal_uncertainty = summary['horz_uncert_km'] * 1000.0
        else:
            n_missing_horz_uncert += 1

        if 'vert_uncert_km' in summary:
            origin.depth_errors = QuantityError(uncertainty=summary['vert_uncert_km'] * 1000.0)
        else:
            n_missing_vert_uncert += 1

        if not sanity_check_done and origin.arrivals:
            a0 = origin.arrivals[0]
            print(f"\n[SANITY CHECK] {f}")
            print(f"  Arrival azimuth={a0.azimuth}  takeoff_angle={a0.takeoff_angle}  distance(deg)={a0.distance}")
            sanity_check_done = True

        # --- Registro para el CSV del catálogo ---
        event_id = str(event.resource_id).split('/')[-1]
        event_dt = origin.time.datetime
        event_dates.append(event_dt)
        event_records.append({
            'event_id':       event_id,
            'file':           f,
            'datetime':       origin.time.isoformat(),
            'latitude':       origin.latitude,
            'longitude':      origin.longitude,
            'depth_km':       origin.depth / 1000.0 if origin.depth is not None else None,
            'rms':            summary.get('rms'),
            'nphases':        summary.get('nphases'),
            'gap':            summary.get('gap'),
            'dist_km':        summary.get('dist_km'),
            'horz_uncert_km': summary.get('horz_uncert_km'),
            'vert_uncert_km': summary.get('vert_uncert_km'),
        })

        arrival_lookup = {str(a.pick_id): a for a in origin.arrivals}

        for pick in event.picks:
            if pick.waveform_id is None:
                continue
            sta_code = pick.waveform_id.station_code
            phase = pick.phase_hint

            arrival = arrival_lookup.get(str(pick.resource_id))
            if arrival is None or arrival.azimuth is None or arrival.takeoff_angle is None:
                n_missing_arrival_geom += 1

            if arrival is not None:
                pick_hrmn = f"{pick.time.hour:02d}{pick.time.minute:02d}"
                pick_sec = f"{pick.time.second + pick.time.microsecond / 1e6:.4f}"
                key = (sta_code, phase, pick_hrmn, pick_sec)
                if key in geom:
                    arrival.azimuth = geom[key]['RAz']
                    n_geom_matched += 1
                else:
                    n_geom_unmatched += 1

            original_network = pick.waveform_id.network_code
            if sta_code in moria_stations:
                pick.waveform_id.network_code = '5L'
            elif sta_code in dpri_stations or sta_code.startswith('DP'):
                pick.waveform_id.network_code = 'DP'
                if original_network != 'DP':
                    network_fixes_count += 1
            elif sta_code in geonet_stations:
                pick.waveform_id.network_code = 'NZ'
            else:
                pick.waveform_id.network_code = 'NZ'

            available_channels = []
            for network in inv:
                if network.code == pick.waveform_id.network_code:
                    for sta in network:
                        if sta.code == sta_code:
                            available_channels = [ch.code for ch in sta.channels]
                            break

            original_channel = pick.waveform_id.channel_code
            if sta_code in dpri_stations:
                if phase == 'P':
                    pick.waveform_id.channel_code = 'EHZ'
                elif phase == 'S':
                    pick.waveform_id.channel_code = 'EH1'
                    if original_channel == 'EHZ':
                        channel_fixes_count += 1
            elif pick.waveform_id.network_code == 'NZ':
                if phase == 'P':
                    pick.waveform_id.channel_code = 'HHZ' if 'HHZ' in available_channels else (
                        'EHZ' if 'EHZ' in available_channels else 'HHZ')
                elif phase == 'S':
                    pick.waveform_id.channel_code = 'HHE' if 'HHE' in available_channels else (
                        'EHE' if 'EHE' in available_channels else 'HHE')
                    if original_channel and original_channel.endswith('Z'):
                        channel_fixes_count += 1

        day_key = f"{event_dt.strftime('%Y')}_{event_dt.strftime('%j')}"
        events_by_day.setdefault(day_key, []).append(event)

print(f"\nEvents outside region: {events_outside_region}")
print(f"Files with parse errors: {n_parse_errors}")
print(f"Events missing horz_uncert_km: {n_missing_horz_uncert}")
print(f"Events missing vert_uncert_km: {n_missing_vert_uncert}")
print(f"Picks missing arrival azimuth/takeoff: {n_missing_arrival_geom}")
print(f"Geometría RAz: {n_geom_matched} matched, {n_geom_unmatched} sin match (quedaron con SAzim)")
print(f"Network fixes (->DP): {network_fixes_count}  Channel fixes: {channel_fixes_count}")

# --- Escribir XML por día ---
for day_key, day_events in events_by_day.items():
    year = day_key.split('_')[0]
    year_dir = join(nll_out, year)
    os.makedirs(year_dir, exist_ok=True)
    daily_catalog = Catalog(events=day_events)
    xml_path = join(year_dir, f"{day_key}_nll.xml")
    daily_catalog.write(xml_path, format="QUAKEML")
    # print(f"Saved {len(daily_catalog)} events to {xml_path}")

# --- Escribir CSV del catálogo ---
df = pd.DataFrame(event_records)

if event_dates:
    min_date = min(event_dates)
    max_date = max(event_dates)
    catalog_name = (f'nll_catalog_{min_date.strftime("%Y_%m_%d")}_'
                     f'{max_date.strftime("%m_%d")}.csv')
else:
    catalog_name = 'nll_catalog_JAN25_SEP25.csv'

catalog_path = f'{out_dir}/{catalog_name}'
df.to_csv(catalog_path, index=False)
print(f"\nSaved {len(df)} events to {catalog_path}")
print(f"Columns: {list(df.columns)}")