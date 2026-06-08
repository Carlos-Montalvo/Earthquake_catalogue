import pandas as pd
from os.path import join
import os
import numpy as np
from obspy import read_events
from obspy.core.event import Catalog
 
### DIRECTORIOS ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
xml_path = join(basedir, 'CATALOGS/MAGNITUDES', 'JAN24_SEP25_MAGNITUDES.xml')
out_dir = join(basedir, 'CATALOGS/MAGNITUDES')
os.makedirs(out_dir, exist_ok=True)
 
### LEER CATÁLOGO ###
print(f'Reading catalog from {xml_path}...')
cat = read_events(xml_path)
print(f'Loaded {len(cat)} events')
 
### CONVERTIR EVENTOS ###
events_by_year = {}
 
for event in cat:
    try:
        origin = event.preferred_origin() or event.origins[0]
    except (IndexError, AttributeError):
        continue
 
    # Tiempo de origen
    ot = origin.time
    year = str(ot.year)
    datetime_str = ot.strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]
 
    # Magnitud preferida + incertidumbre + N estaciones
    magnitude  = None
    uncertainty = None
    usedstations = None
    try:
        mag_obj = event.preferred_magnitude() or event.magnitudes[0]
        magnitude = mag_obj.mag
 
        # Incertidumbre: σ de los residuos de station magnitudes
        residuals = np.array([
            c.residual for c in mag_obj.station_magnitude_contributions
            if c.residual is not None
        ])
        if len(residuals) >= 2:
            uncertainty = float(np.std(residuals))
 
        # Estaciones usadas para calcular la magnitud (= las que tienen residuo)
        usedstations = len(residuals) if len(residuals) > 0 else None
 
    except (IndexError, AttributeError):
        pass
 
    # Calidad
    rms, nphases, gap, dist_km = None, None, None, None
    if origin.time_errors and origin.time_errors.uncertainty:
        rms = origin.time_errors.uncertainty
    if origin.quality:
        nphases = origin.quality.used_phase_count
        gap = origin.quality.azimuthal_gap
        if origin.quality.minimum_distance is not None:
            dist_km = origin.quality.minimum_distance / 1000.0
 
    # Picks
    picks_list = []
    for pick in event.picks:
        if pick.phase_hint not in ('P', 'S'):
            continue
        picks_list.append({
            'station': pick.waveform_id.station_code,
            'channel': pick.waveform_id.channel_code,
            'phase': pick.phase_hint,
            'datetime': pick.time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-2]
        })
 
    row = {
        'datetime':     datetime_str,
        'latitude':     origin.latitude,
        'longitude':    origin.longitude,
        'depth_km':     origin.depth / 1000.0 if origin.depth is not None else None,
        'rms':          rms,
        'nphases':      nphases,
        'gap':          gap,
        'dist_km':      dist_km,
        'picks':        picks_list,
        'magnitude':    magnitude,
        'uncertainty':  uncertainty,
        'usedstations': usedstations,
    }
 
    if year not in events_by_year:
        events_by_year[year] = []
    events_by_year[year].append(row)

### GUARDAR CSV POR AÑO ###
for year, rows in sorted(events_by_year.items()):
    df = pd.DataFrame(rows, columns=[
        'datetime', 'latitude', 'longitude', 'depth_km',
        'rms', 'nphases', 'gap', 'dist_km', 'picks',
        'magnitude', 'uncertainty', 'usedstations'
    ])
    df = df.sort_values('datetime').reset_index(drop=True)

    out_path = join(out_dir, f'catalog_{year}.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved {len(df)} events to {out_path}')

### GUARDAR CSV ÚNICO ###
all_rows = [row for rows in events_by_year.values() for row in rows]
df = pd.DataFrame(all_rows, columns=[
    'datetime', 'latitude', 'longitude', 'depth_km',
    'rms', 'nphases', 'gap', 'dist_km', 'picks',
    'magnitude', 'uncertainty', 'usedstations'
])
df = df.sort_values('datetime').reset_index(drop=True)
 
out_path = join(out_dir, 'catalog_all.csv')
df.to_csv(out_path, index=False)
print(f'Saved {len(df)} events to {out_path}')
print(f'\nDone. {len(events_by_year)} years processed.')