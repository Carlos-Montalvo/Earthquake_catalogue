#!/usr/bin/env python3
"""
EQT Detection for Single Day
Processes seismic data for a single Julian day using EQTransformer
Based on the original EQT_detection.py but optimized for single-day processing
"""

# Standard library imports
import os
import argparse
import sys
import warnings
import gc
import torch
from glob import glob
from os import makedirs
from os.path import join, exists
from datetime import datetime
from pandas import date_range

# Add path for resource monitor
sys.path.append('/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON')

# Third-party imports
import obspy
import seisbench.models as sbm
import pandas as pd

# Scientific/seismic imports
from obspy import read, read_inventory
from pandas import DataFrame

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------

# Transforms two dates to a date range and julian dates
def dates(s_date,e_date):
    year,sm,sd = s_date.split('-')
    yy,em,ed = e_date.split('-')
    date_format = '%Y-%m-%d'
    tperiod = date_range(start=s_date,end=e_date).strftime(date_format)
    jperiod = date_range(start=s_date,end=e_date).strftime('%j')
    time_period = ((tperiod,jperiod))
    return time_period,year


def save_probability_curves_sac(probs_max, probs_avg, jday, year, probspath):
    """Save probability curves as SAC files"""
    # Create directories if they don't exist
    yearpath = join(probspath, year)
    if not exists(yearpath):
        makedirs(yearpath)
    
    print(f"Saving probability curves to: {yearpath}")
    
    # Get P and S probabilities - try different channel naming patterns
    p_traces_max = [tr for tr in probs_max if 'P' in tr.stats.channel]
    s_traces_max = [tr for tr in probs_max if 'S' in tr.stats.channel]
    p_traces_avg = [tr for tr in probs_avg if 'P' in tr.stats.channel]
    s_traces_avg = [tr for tr in probs_avg if 'S' in tr.stats.channel]
    
    # Save max probabilities
    for tr in p_traces_max:
        station = tr.stats.station
        filename_max = join(yearpath, f'{station}_P_max_{jday}_{year}.sac')
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'P_PROB_MAX'
        tr_sac.stats.sac.knetwk = getattr(tr.stats, 'network', 'XX')
        try:
            tr_sac.write(filename_max, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_max}: {e}')
    
    for tr in s_traces_max:
        station = tr.stats.station
        filename_max = join(yearpath, f'{station}_S_max_{jday}_{year}.sac')
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'S_PROB_MAX'
        tr_sac.stats.sac.knetwk = getattr(tr.stats, 'network', 'XX')
        try:
            tr_sac.write(filename_max, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_max}: {e}')
    
    # Save avg probabilities
    for tr in p_traces_avg:
        station = tr.stats.station
        filename_avg = join(yearpath, f'{station}_P_avg_{jday}_{year}.sac')
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'P_PROB_AVG'
        tr_sac.stats.sac.knetwk = getattr(tr.stats, 'network', 'XX')
        try:
            tr_sac.write(filename_avg, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_avg}: {e}')
    
    for tr in s_traces_avg:
        station = tr.stats.station
        filename_avg = join(yearpath, f'{station}_S_avg_{jday}_{year}.sac')
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'S_PROB_AVG'
        tr_sac.stats.sac.knetwk = getattr(tr.stats, 'network', 'XX')
        try:
            tr_sac.write(filename_avg, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_avg}: {e}')

def process_single_day(year, jday, datadir, mode, pickspath, probspath):
    """Process a single day's data with EQTransformer"""
    
    start_time = datetime.now()
    print(f"Starting EQT processing for year {year}, day {jday}")
    print(f"Data directory: {datadir}")
    print(f"Mode: {mode}")
    
        # Build year path
    yearpath = join(datadir, year)
    
    if not exists(yearpath):
        print(f"Error: Year directory {yearpath} does not exist")
        return False
    
    # Reading data
    print(f'Reading data for day {jday}')
    search_pattern = join(yearpath, f'*/*/*.{jday}')
    print(f'Searching for files with pattern: {search_pattern}')
    traces = glob(search_pattern)
    print(f'Found {len(traces)} waveform files for day {jday}')
    
    if len(traces) == 0:
        print(f'Warning: No files found matching pattern {search_pattern}')
        print(f'Please verify your directory structure: datadir/YYYY/NETWORK/STATION/waveforms')
        return False
    
    # Load traces into stream
    st = obspy.Stream()
    loaded_count = 0
    for trace in traces:
        try:
            tr = read(trace)  # Use direct read function like original
            st += tr
            loaded_count += 1
        except Exception as e:
            print(f'Could not read {trace}: {e}')
    
    print(f'📈 Loaded {len(st)} traces from {loaded_count} files for day {jday}')
    
    # Verify sufficient data
    if len(st) == 0:
        print(f'⚠️ No data found for day {jday}, skipping...')
        return False
    
    # Load EQTransformer model
    model_start = datetime.now()
    print('Loading EQTransformer model...')
    model = sbm.EQTransformer.from_pretrained('original')
    if torch.cuda.is_available():
        model.cuda()
        print(f'Using GPU: {torch.cuda.get_device_name()}')
    else:
        print('Using CPU')
    model_end = datetime.now()
    print(f'Model loaded in {(model_end - model_start).total_seconds():.1f} seconds')
    
    batch_size = 10000  # Optimal batch size based on performance testing  # Restore original larger batch size for efficiency
    
    # Create output directories
    year_picks_path = join(pickspath, year)
    year_probs_path = join(probspath, year)
    makedirs(year_picks_path, exist_ok=True)
    makedirs(year_probs_path, exist_ok=True)
    
    picks_max_count = 0
    picks_avg_count = 0
    probs_max = []
    probs_avg = []
    
    try:
        if mode in [1, 3]:  # Max picking
            print('Extracting max probabilities...')
            annotate_start = datetime.now()
            probs_max = model.annotate(st, stacking="max", overlap=5500, 
                                      batch_size=batch_size)
            annotate_end = datetime.now()
            print(f'Max probabilities extracted in {(annotate_end - annotate_start).total_seconds():.1f} seconds')
            
            # Extract picks from probabilities
            print('Extracting picks from max probabilities...')
            try:
                picks_max = model.classify_aggregate(
                    probs_max, 
                    argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
                ).picks
            except (AttributeError, TypeError) as e:
                print(f'Using fallback classify method: {e}')
                picks_max = model.classify(st, stacking="max", overlap=5500, 
                                         P_threshold=0.05, S_threshold=0.05, 
                                         batch_size=batch_size)
            
            # Save picks
            picks_max_dicts = [pick.__dict__ for pick in picks_max]
            picks_max_df = pd.DataFrame(picks_max_dicts)
            picks_max_df.to_csv(join(year_picks_path, f'picks_max_{year}_{jday}.csv'), index=False)
            picks_max_count = len(picks_max)
            print(f'Saved {picks_max_count} max picks')
        
        if mode in [2, 3]:  # Avg picking
            print('Extracting avg probabilities...')
            probs_avg = model.annotate(st, stacking="avg", overlap=5500, 
                                      batch_size=batch_size)
            
            # Extract picks from probabilities
            print('Extracting picks from avg probabilities...')
            try:
                picks_avg = model.classify_aggregate(
                    probs_avg, 
                    argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
                ).picks
            except (AttributeError, TypeError) as e:
                print(f'Using fallback classify method: {e}')
                picks_avg = model.classify(st, stacking="avg", overlap=5500, 
                                         P_threshold=0.05, S_threshold=0.05, 
                                         batch_size=batch_size)
            
            # Save picks
            picks_avg_dicts = [pick.__dict__ for pick in picks_avg]
            picks_avg_df = pd.DataFrame(picks_avg_dicts)
            picks_avg_df.to_csv(join(year_picks_path, f'picks_avg_{year}_{jday}.csv'), index=False)
            picks_avg_count = len(picks_avg)
            print(f'Saved {picks_avg_count} avg picks')
        
        # Save probability curves
        # if mode in [1, 2, 3]:
        #     print('Saving probability curves as SAC files...')
        #     save_probability_curves_sac(probs_max, probs_avg, jday, year, probspath)
        
    except Exception as e:
        print(f'Error during processing: {e}')
        return False
    
    finally:
        # Cleanup memory
        del st, model
        if mode in [1, 3] and 'picks_max' in locals():
            del picks_max, picks_max_df
        if mode in [2, 3] and 'picks_avg' in locals():
            del picks_avg, picks_avg_df
        if probs_max:
            del probs_max
        if probs_avg:
            del probs_avg
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f'✅ Day {jday} completed successfully in {duration:.1f}s')
    print(f'   Max picks: {picks_max_count}, Avg picks: {picks_avg_count}')
    
    return True

# def process_single_day(year, jday, datadir, mode, pickspath, probspath, stations_per_batch=10):
    
#     start_time = datetime.now()
#     print(f"Starting EQT processing for year {year}, day {jday}")
    
#     yearpath = join(datadir, year)
#     if not exists(yearpath):
#         print(f"Error: Year directory {yearpath} does not exist")
#         return False

#     # Encontrar archivos y agrupar por estación
#     search_pattern = join(yearpath, f'*/*/*.{jday}')
#     traces = glob(search_pattern)
#     print(f'Found {len(traces)} waveform files for day {jday}')
    
#     if len(traces) == 0:
#         return False

#     # Agrupar archivos por estación (carpeta padre)
#     from itertools import groupby
#     def get_station_key(filepath):
#         return os.sep.join(filepath.split(os.sep)[-3:-1])  # NETWORK/STATION
    
#     traces_sorted = sorted(traces, key=get_station_key)
#     station_groups = {k: list(v) for k, v in groupby(traces_sorted, key=get_station_key)}
#     station_list = list(station_groups.keys())
#     total_stations = len(station_list)
#     print(f'Found {total_stations} stations, processing in batches of {stations_per_batch}')

#     # Crear directorios de salida
#     year_picks_path = join(pickspath, year)
#     makedirs(year_picks_path, exist_ok=True)

#     # Cargar modelo una sola vez antes de los batches
#     print('Loading EQTransformer model...')
#     model_start = datetime.now()
#     model = sbm.EQTransformer.from_pretrained('original')
#     if torch.cuda.is_available():
#         model.cuda()
#         print(f'Using GPU: {torch.cuda.get_device_name()}')
#     else:
#         print('Using CPU')
#     print(f'Model loaded in {(datetime.now() - model_start).total_seconds():.1f}s')

#     # Acumuladores de picks para todo el día
#     all_picks_max = []
#     all_picks_avg = []

#     # Procesar en batches de estaciones
#     for batch_idx in range(0, total_stations, stations_per_batch):
#         batch_stations = station_list[batch_idx:batch_idx + stations_per_batch]
#         batch_num = batch_idx // stations_per_batch + 1
#         total_batches = (total_stations + stations_per_batch - 1) // stations_per_batch
#         print(f'\nBatch {batch_num}/{total_batches}: {len(batch_stations)} stations')

#         # Leer solo las estaciones de este batch
#         st = obspy.Stream()
#         for station_key in batch_stations:
#             for f in station_groups[station_key]:
#                 try:
#                     st += read(f)
#                 except Exception as e:
#                     print(f'Could not read {f}: {e}')

#         if len(st) == 0:
#             print(f'No data for batch {batch_num}, skipping')
#             continue

#         print(f'Loaded {len(st)} traces for batch {batch_num}')

#         try:
#             if mode in [1, 3]:
#                 probs_max = model.annotate(st, stacking="max", overlap=5500, batch_size=10000)
#                 try:
#                     picks_max = model.classify_aggregate(
#                         probs_max,
#                         argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
#                     ).picks
#                 except (AttributeError, TypeError):
#                     picks_max = model.classify(st, stacking="max", overlap=5500,
#                                               P_threshold=0.05, S_threshold=0.05,
#                                               batch_size=10000)
#                 all_picks_max.extend(picks_max)
#                 del probs_max, picks_max

#             if mode in [2, 3]:
#                 probs_avg = model.annotate(st, stacking="avg", overlap=5500, batch_size=10000)
#                 try:
#                     picks_avg = model.classify_aggregate(
#                         probs_avg,
#                         argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
#                     ).picks
#                 except (AttributeError, TypeError):
#                     picks_avg = model.classify(st, stacking="avg", overlap=5500,
#                                               P_threshold=0.05, S_threshold=0.05,
#                                               batch_size=10000)
#                 all_picks_avg.extend(picks_avg)
#                 del probs_avg, picks_avg

#         except Exception as e:
#             print(f'Error processing batch {batch_num}: {e}')

#         finally:
#             # Liberar memoria del batch inmediatamente
#             st.clear()
#             del st
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

#     # Guardar todos los picks acumulados
#     del model
#     gc.collect()

#     picks_max_count = 0
#     picks_avg_count = 0

#     if mode in [1, 3] and all_picks_max:
#         picks_max_dicts = [pick.__dict__ for pick in all_picks_max]
#         picks_max_df = pd.DataFrame(picks_max_dicts)
#         picks_max_df.to_csv(join(year_picks_path, f'picks_max_{year}_{jday}.csv'), index=False)
#         picks_max_count = len(all_picks_max)
#         print(f'\nSaved {picks_max_count} max picks for day {jday}')

#     if mode in [2, 3] and all_picks_avg:
#         picks_avg_dicts = [pick.__dict__ for pick in all_picks_avg]
#         picks_avg_df = pd.DataFrame(picks_avg_dicts)
#         picks_avg_df.to_csv(join(year_picks_path, f'picks_avg_{year}_{jday}.csv'), index=False)
#         picks_avg_count = len(all_picks_avg)
#         print(f'Saved {picks_avg_count} avg picks for day {jday}')

#     end_time = datetime.now()
#     duration = (end_time - start_time).total_seconds()
#     print(f'\n✅ Day {jday} completed in {duration:.1f}s')
#     print(f'   Max picks: {picks_max_count}, Avg picks: {picks_avg_count}')

#     return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='EQTransformer detection for single day')
    parser.add_argument('--year', required=True, help='Year to process (YYYY)')
    parser.add_argument('--jday', required=True, help='Julian day to process (001-366)')
    parser.add_argument('--basedir', required=True, help='Base directory path')
    parser.add_argument('--datadir', required=True, help='Data directory path')
    parser.add_argument('--mode', type=int, default=3, 
                       help='Mode: 1=max picks, 2=avg picks, 3=both (default: 3)')
    # parser.add_argument('--stations_per_batch', type=int, default=10,
    #                 help='Number of stations to process per batch (default: 10)')
    
    args = parser.parse_args()
    
    # Setup paths
    montalca = r'/Volumes/GeoPhysics_49/users-data/montalca'
    catalogs = join(montalca, 'CATALOGS/EQT_PICKS')
    pickspath = join(catalogs, 'PICKS')
    probspath = join(catalogs, 'PROBS')
    
    # Ensure output directories exist
    makedirs(pickspath, exist_ok=True)
    makedirs(probspath, exist_ok=True)
    
    print("="*60)
    print(f"EQT Detection - Single Day Processing")
    print("="*60)
    print(f"Year: {args.year}")
    print(f"Julian Day: {args.jday}")
    print(f"Base directory: {args.basedir}")
    print(f"Data directory: {args.datadir}")
    print(f"Mode: {args.mode}")
    print(f"Picks output: {pickspath}")
    print(f"Probabilities output: {probspath}")
    print("="*60)
    
    # Process the day
    success = process_single_day(
        args.year, 
        args.jday, 
        args.datadir, 
        args.mode, 
        pickspath, 
        probspath,
        # stations_per_batch=args.stations_per_batch
    )
    
    if success:
        print(f"✅ Successfully processed day {args.jday}")
        sys.exit(0)
    else:
        print(f"❌ Failed to process day {args.jday}")
        sys.exit(1)

if __name__ == "__main__":
    main()
