# Code adapted from Finnigan Illsley Kemp, edited by Stephen/Cedric
# Adapted for daily XML catalogs and pickle files by montalca

import pickle
import shutil
import logging
import datetime
import os
import sys
import numpy as np
from multiprocessing import cpu_count
from pandas import date_range
from os.path import join, exists
from obspy import *
from eqcorrscan import *
from eqcorrscan.core import *
# Import the modified version instead of original eqcorrscan.utils
sys.path.append('/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON')
import catalog_to_dd_edited_SK as catalog_to_dd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

# Confirm which version is being used
print(f"✓ Using modified catalog_to_dd from: {catalog_to_dd.__file__}")
print("  This version includes SK's fixes for stream ID compatibility")

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

# Transforms two dates to a date range and julian dates
def dates(s_date,e_date):
    year,sm,sd = s_date.split('-')
    yy,em,ed = e_date.split('-')
    date_format = '%Y-%m-%d'
    tperiod = date_range(start=s_date,end=e_date).strftime(date_format)
    jperiod = date_range(start=s_date,end=e_date).strftime('%j')
    time_period = ((tperiod,jperiod))
    return time_period,year,sm,sd,em,ed

def get_event_key(event):
    try:
        origin_time = event.preferred_origin().time or event.origins[0].time
    except:
        origin_time = min([pick.time for pick in event.picks])
    return f"{origin_time.year}_{origin_time.julday:03d}_{origin_time.hour:02d}{origin_time.minute:02d}{origin_time.second:02d}"

def combine_daily_catalogs_and_streams(nll_dir, streams_dir, time_period, year):
    """
    Combine daily XML catalogs and pickle streams into single objects for correlation processing.
    This ensures the final .cc file contains correlations for the ENTIRE time period.
    """
    print("Loading daily files...")
    combined_catalog = Catalog()
    combined_stream_dict = {}
    
    successful_days = 0
    failed_days = []
    total_events = 0
    total_streams = 0
    
    for jday in time_period[1]:
        catalog_file = join(nll_dir, f'{year}/{year}_{jday}_nll.xml')
        streams_file = join(streams_dir, f'{year}_{jday}.p')
        
        try:
            if exists(catalog_file) and exists(streams_file):
                daily_cat = read_events(catalog_file)
                with open(streams_file, 'rb') as fp:
                    daily_streams = pickle.load(fp)
                
                combined_catalog.extend(daily_cat)
                combined_stream_dict.update(daily_streams)
                
                successful_days += 1
                total_events += len(daily_cat)
                total_streams += len(daily_streams)
                
            else:
                failed_days.append(jday)
                
        except Exception as e:
            failed_days.append(jday)
    
    print(f"✓ Loaded {successful_days}/{len(time_period[1])} days | {len(combined_catalog)} events | {len(combined_stream_dict)} streams")
    if failed_days:
        print(f"✗ Failed days: {failed_days}")

    def get_event_key(event):
        try:
            origin_time = event.preferred_origin().time or event.origins[0].time
        except:
            origin_time = min([pick.time for pick in event.picks])
        return f"{origin_time.year}_{origin_time.julday:03d}_{origin_time.hour:02d}{origin_time.minute:02d}{origin_time.second:02d}"

    catalog_event_ids = set(get_event_key(event) for event in combined_catalog)
    stream_event_ids = set(combined_stream_dict.keys())

    # # Check for mismatch between catalog and streams
    # catalog_event_ids = set(str(event.resource_id) for event in combined_catalog)
    # stream_event_ids = set(combined_stream_dict.keys())
    
    missing_streams = catalog_event_ids - stream_event_ids
    extra_streams = stream_event_ids - catalog_event_ids
    
    if missing_streams:
        print(f"⚠ {len(missing_streams)} events in catalog have no streams")
    if extra_streams:
        print(f"⚠ {len(extra_streams)} streams have no corresponding events")
    
    # --- DIAGNÓSTICO DE EVENT IDs ---

    # Toma muestras de cada lado
    catalog_ids_sample = [str(get_event_key(event)) for event in combined_catalog[:5]]
    stream_ids_sample = list(combined_stream_dict.keys())[:5]
    
    print("\n--- IDs del catálogo (XML) ---")
    for cid in catalog_ids_sample:
        print(repr(cid))
    
    print("\n--- IDs de los streams (pickle) ---")
    for sid in stream_ids_sample:
        print(repr(sid))
    
    # Compara longitudes y estructura
    print("\n--- Estructura ---")
    print(f"Tipo catalog id: {type(catalog_ids_sample[0])}, len: {len(catalog_ids_sample[0])}")
    print(f"Tipo stream id: {type(stream_ids_sample[0])}, len: {len(stream_ids_sample[0])}")
    
    # Intenta ver si uno está contenido en el otro (substring)
    test_cat_id = catalog_ids_sample[0]
    matches = [sid for sid in combined_stream_dict.keys() if sid in test_cat_id or test_cat_id in sid]
    print(f"\n¿Algún stream id es substring (o viceversa) del primer catalog id?")
    print(matches[:3] if matches else "Ninguno encontrado")
    
    # Revisa si el problema es un prefijo tipo 'smi:local/' o 'quakeml:'
    import re
    prefixes_cat = set(re.match(r'^[a-zA-Z]+:[^/]*/?', cid).group(0) if re.match(r'^[a-zA-Z]+:[^/]*/?', cid) else 'NONE' for cid in catalog_ids_sample)
    print(f"\nPrefijos detectados en catalog ids: {prefixes_cat}")

    # ¿Los streams huérfanos son de días específicos o dispersos?
    orphan_ids = stream_event_ids - catalog_event_ids
    orphan_days = {}
    for sid in orphan_ids:
        # Buscar en qué pickle vive este stream
        for jday in time_period[1]:
            streams_file = join(streams_dir, f'{year}_{jday}.p')
            if exists(streams_file):
                with open(streams_file, 'rb') as fp:
                    d = pickle.load(fp)
                if sid in d:
                    orphan_days[sid] = jday
                    break
    print(f"Orphan streams por día: {set(orphan_days.values())}")

    return combined_catalog, combined_stream_dict

# ------------------------------------------------------------------------------
# Main script starts here
# ------------------------------------------------------------------------------

### DIRECTORIES AND FILES ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
# Catalogues
ctlgdir = join(basedir,'CATALOGS')
nll_dir = join(ctlgdir,'NLL')
growclust_dir = join(ctlgdir,'GROWCLUST')
streams_dir = join(growclust_dir,'STREAMS')
corrdir = join(growclust_dir,'CORRELATIONS')
# Stations
sta_dir = join(basedir,'STATIONS')
inv = read_inventory(join(sta_dir,'ALL_STATIONS.xml'))

print('Directory structure: basedir/datadir/year/stations/data')
print('')
basedir = input('Write basedir path: ')
datapth = input('Write datadir path: ')
datadir = join(basedir,datapth)
print('')

# Time period
print('Time period has to be from the same year')
s_date = input('Start date (format yyyy-mm-dd): ')
e_date = input('End date (format yyyy-mm-dd): ')
time_period,year,sm,sd,em,ed = dates(s_date,e_date)
yearpath = join(datadir,year)

# Create output directory for correlations
if not exists(corrdir):
    os.makedirs(corrdir)
    print(f'Created correlations directory: {corrdir}')

### CORRELATION SETTINGS ###
setup_required = True
adding_existing_eventmapper = False

# Configuration
correlation_params = {
    'extract_len': 1,      # Length in seconds to extract around the pick
    'pre_pick': 0.3,       # Time before the pick to start the correlation window
    'shift_len': 0.3,      # Time to allow pick to vary in seconds
    'lowcut': 1,           # Lowcut in Hz
    'highcut': 30,         # Highcut in Hz
    'max_sep': 25,         # Maximum separation between event pairs in km
    'min_link': 4,         # Minimum links for an event to be paired
    'min_cc': 0.2,        # Threshold to include cross-correlation results
    'tdiff_thresh': 10     # Maximum time difference to keep
}

# Multiprocessing configuration
available_cpus = cpu_count()
while True:
    try:
        max_workers = int(input(f'Workers for correlations (1-{available_cpus}): '))
        if 1 <= max_workers <= available_cpus:
            break
        else:
            print(f'Enter 1-{available_cpus}')
    except ValueError:
        print('Enter a valid number')

# File names
period_str = f"{year}_{sm}{sd}_{em}{ed}"
eventidmapper_file = join(corrdir, f'event_id_mapper_{period_str}.p')
eventlist_file = join(corrdir, f'event_list_{period_str}.dat')
correlation_output_file = join(corrdir, f'correlations_{period_str}.cc')

codestart = datetime.datetime.now()

if setup_required:
    print('\n1. Loading data...')
    
    # Combine daily catalogs and streams
    cat, stream_dict = combine_daily_catalogs_and_streams(nll_dir, streams_dir, time_period, year)

    # Re-indexar streams por resource_id (estable mientras cat esté en memoria)
    resource_stream_dict = {}
    matched, unmatched = 0, 0
    for event in cat:
        key = get_event_key(event)
        if key in stream_dict:
            resource_stream_dict[str(event.resource_id)] = stream_dict[key]
            matched += 1
        else:
            unmatched += 1
    
    print(f"✓ {matched}/{len(cat)} events re-matched to streams by resource_id")
    stream_dict = resource_stream_dict
    
    if len(cat) == 0:
        print("ERROR: No events found!")
        exit(1)
    
    if len(stream_dict) == 0:
        print("ERROR: No streams found!")
        exit(1)
    
    # Check event-stream matching
    matched_events = len(set(str(e.resource_id) for e in cat) & set(stream_dict.keys()))
    print(f"✓ {matched_events}/{len(cat)} events have matching streams")
    
    if matched_events == 0:
        print("ERROR: No events have matching streams!")
        print("Check that event IDs match between XML files and pickle files")
        exit(1)
    
    print('\n2. Creating station file...')
    # Produce station file
    stafile_path = join(corrdir, f'stations_{period_str}.dat')
    stafile = open(stafile_path, 'w')
    done = []
    for net in inv:
        for sta in net:
            if "%s_%s" % (net.code,sta.code) not in done:
                stafile.write("%s %s %s %s\n" % (sta.code, sta.latitude, sta.longitude, sta.elevation))
                done.append("%s_%s" % (net.code,sta.code))
    stafile.close()
    print(f"✓ {len(done)} stations")
    
    print('\n3. Mapping event IDs...')
    if adding_existing_eventmapper and exists(eventidmapper_file):
        with open(eventidmapper_file, 'rb') as fp:
            event_id_mapper = pickle.load(fp)
    else:
        event_id_mapper = None
    
    # Make dt.ct file and get dictionary of event_id mapping
    event_id_mapper = catalog_to_dd.write_catalog(cat, event_id_mapper=event_id_mapper)

    # Save event_id_mapper dictionary to file for future reference
    with open(eventidmapper_file,'wb') as fp:
        pickle.dump(event_id_mapper,fp,protocol=pickle.HIGHEST_PROTOCOL)

    # Write event.dat using previous event_id_mapper
    catalog_to_dd.write_event(cat, event_id_mapper=event_id_mapper)
    shutil.copy('event.dat', eventlist_file)
    print(f"✓ {len(event_id_mapper)} event IDs mapped")

print('\n4. Computing correlations...')
# Re-read the event mapper for correlation phase
with open(eventidmapper_file, 'rb') as fp:
    event_id_mapper = pickle.load(fp)

# Write event.dat again
catalog_to_dd.write_event(cat, event_id_mapper=event_id_mapper)
shutil.copy('event.dat', eventlist_file)

print(f"Events: {len(cat)} | Workers: {max_workers} | Max distance: {correlation_params['max_sep']}km")

# Preprocess streams to handle masked data and gaps
print("Preprocessing streams to handle gaps and masked data...")
processed_stream_dict = {}
total_events = len(stream_dict)
processed_count = 0
removed_count = 0

for event_id, stream in stream_dict.items():
    processed_count += 1
    if processed_count % 100 == 0:
        print(f"  Processed {processed_count}/{total_events} events...")
    
    if len(stream) == 0:
        removed_count += 1
        continue
    
    try:
        # Create a copy of the stream for processing
        st_clean = stream.copy()
        
        # Merge traces to fill gaps with linear interpolation or zeros
        try:
            st_clean.merge(method=1, fill_value=0)  # Fill gaps with zeros
        except:
            try:
                st_clean.merge(method=0)  # Simple merge without filling
            except:
                pass  # If merge fails, continue with original traces
        
        # Remove traces with masked data or gaps
        traces_to_remove = []
        for i, tr in enumerate(st_clean):
            # Check for masked data
            if hasattr(tr.data, 'mask') and tr.data.mask.any():
                traces_to_remove.append(i)
                continue
            
            # Check for NaN or inf values
            if np.isnan(tr.data).any() or np.isinf(tr.data).any():
                traces_to_remove.append(i)
                continue
            
            # Check if trace is too short
            if len(tr.data) < 100:  # Minimum 100 samples
                traces_to_remove.append(i)
                continue
                
        # Remove problematic traces in reverse order
        for i in reversed(traces_to_remove):
            st_clean.pop(i)
        
        # Only keep events with at least some clean traces
        if len(st_clean) > 0:
            processed_stream_dict[event_id] = st_clean
        else:
            removed_count += 1
            
    except Exception as e:
        print(f"  Warning: Failed to process event {event_id}: {e}")
        removed_count += 1
        continue

print(f"✓ Stream preprocessing complete: {len(processed_stream_dict)} events kept, {removed_count} removed")

# Compute and write correlation picks for ALL events in the period
catalog_to_dd.write_correlations(
    cat, processed_stream_dict, 
    correlation_params['extract_len'], 
    correlation_params['pre_pick'], 
    correlation_params['shift_len'],
    lowcut=correlation_params['lowcut'],
    highcut=correlation_params['highcut'],
    max_sep=correlation_params['max_sep'],
    min_link=correlation_params['min_link'],
    event_id_mapper=event_id_mapper,
    max_workers=max_workers,
    min_cc=correlation_params['min_cc']
)

print('\n5. Filtering correlations...')
# Filter correlations by time difference threshold

outfile = open(correlation_output_file, "w")
rem_count = 0
line_count = 0
kept_count = 0

# Check if dt.cc file exists and has content
if not exists('dt.cc'):
    print("✗ No correlations generated - dt.cc file not found")
    print("This usually means no event pairs met the correlation criteria")
    exit(1)

with open('dt.cc', 'r') as infile:
    for line in infile:
        line_count += 1
        columns = line.split()
        if len(columns) == 0:  # Skip empty lines
            continue
        if columns[0] == "#" or abs(float(columns[1])) < correlation_params['tdiff_thresh']:
            outfile.write(line)
            kept_count += 1
        else:
            rem_count += 1

outfile.close()

if line_count == 0:
    print("✗ No correlations found in dt.cc file")
    print("Possible reasons:")
    print("  - Events too far apart (max_sep too small)")
    print("  - Not enough shared picks (min_link too high)")
    print("  - Missing stream data for events")
    print("  - Correlation threshold too high (min_cc)")
else:
    retention_rate = kept_count/line_count*100 if line_count > 0 else 0
    print(f"✓ Kept {kept_count}/{line_count} correlations ({retention_rate:.1f}%)")
    print(f"✓ Final file: {correlation_output_file}")

codestop = datetime.datetime.now()
execution_time = codestop - codestart

print(f'\n=== SUMMARY ===')
print(f'Period: {s_date} to {e_date}')
print(f'Events: {len(cat)} | Streams: {len(stream_dict)} | Workers: {max_workers}')
print(f'Time: {execution_time}')
print(f'Output: {correlation_output_file}')