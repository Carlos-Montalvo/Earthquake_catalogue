# Standard library imports
import gc
import warnings
import sys
from glob import glob
from os import makedirs
from os.path import join, exists
from datetime import datetime

# Add path for resource monitor
sys.path.append('/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/OUT_LOGS')
from resource_monitor import start_resource_monitoring, stop_resource_monitoring

# Third-party imports
import pandas as pd
import torch

# Scientific/seismic imports
import obspy
import seisbench.models as sbm
from obspy import read, read_inventory
from pandas import DataFrame, date_range

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Reducir warnings innecesarios

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

# Save probability curves as SAC files
def save_probability_curves_sac(probs_max, probs_avg, jday, year):
    # Create directories if they don't exist
    yearpath = join(probspath,year)
    if not exists(yearpath):
        makedirs(yearpath)
    
    # print(f"Debug: probs_max has {len(probs_max)} traces")
    # print(f"Debug: probs_avg has {len(probs_avg)} traces")
    
    # # Debug: Print channel names to understand the structure
    # if len(probs_max) > 0:
    #     print(f"Debug: First few channel names in probs_max:")
    #     for i, tr in enumerate(probs_max[:5]):  # Show first 5 channels
    #         print(f"  {i}: {tr.stats.channel} - Station: {tr.stats.station}")
    
    # Get P and S probabilities - try different channel naming patterns
    p_traces_max = [tr for tr in probs_max if 'P' in tr.stats.channel]
    s_traces_max = [tr for tr in probs_max if 'S' in tr.stats.channel]
    p_traces_avg = [tr for tr in probs_avg if 'P' in tr.stats.channel]
    s_traces_avg = [tr for tr in probs_avg if 'S' in tr.stats.channel]
    
    # print(f"Debug: Found {len(p_traces_max)} P traces and {len(s_traces_max)} S traces in probs_max")
    # print(f"Debug: Found {len(p_traces_avg)} P traces and {len(s_traces_avg)} S traces in probs_avg")
    for tr in p_traces_max:
        station = tr.stats.station
        filename_max = join(yearpath,f'{station}_P_{jday}_{year}.sac')
        tr.write(filename_max, format='SAC')
        # Creates a copy to modify the metadata
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'P_PROB_MAX'
        tr_sac.stats.sac.knetwk = tr.stats.network if hasattr(tr.stats, 'network') else 'XX'
        try:
            tr_sac.write(filename_max, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_max}: {e}')
    for tr in s_traces_max:
        station = tr.stats.station
        filename_max = join(yearpath, f'{station}_S_{jday}_{year}.sac')
        tr.write(filename_max, format='SAC')
        # Creates a copy to modify the metadata
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'S_PROB_MAX'
        tr_sac.stats.sac.knetwk = tr.stats.network if hasattr(tr.stats, 'network') else 'XX'
        try:
            tr_sac.write(filename_max, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_max}: {e}')
    for tr in p_traces_avg:
        station = tr.stats.station
        filename_avg = join(yearpath, f'{station}_P_{jday}_{year}.sac')
        tr.write(filename_avg, format='SAC')
        # Creates a copy to modify the metadata
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'P_PROB_AVG'
        tr_sac.stats.sac.knetwk = tr.stats.network if hasattr(tr.stats, 'network') else 'XX'
        try:
            tr_sac.write(filename_avg, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_avg}: {e}')
    for tr in s_traces_avg:
        station = tr.stats.station
        filename_avg = join(yearpath, f'{station}_S_{jday}_{year}.sac')
        tr.write(filename_avg, format='SAC')
        # Creates a copy to modify the metadata
        tr_sac = tr.copy()
        tr_sac.stats.sac = {}
        tr_sac.stats.sac.kstnm = station
        tr_sac.stats.sac.kcmpnm = 'S_PROB_AVG'
        tr_sac.stats.sac.knetwk = tr.stats.network if hasattr(tr.stats, 'network') else 'XX'
        try:
            tr_sac.write(filename_avg, format='SAC')
        except Exception as e:
            print(f'Error saving {filename_avg}: {e}')
    print(f'Probability curves saved to: {yearpath}')

# Pick phases using EQTransformer model
def pick_phases(time_period,yearpath):
    # Extraer el año del path
    year = yearpath.split('/')[-1]
    for i,jday in enumerate(time_period[1][:]):
        # Reading data
        print(f'Reading data from {time_period[0][i]}')
        # Updated pattern to match datadir/YYYY/NETWORK/STATION/waveforms structure
        search_pattern = join(yearpath,f'*/*/*.{jday}')
        print(f'Searching for files with pattern: {search_pattern}')
        traces = glob(search_pattern)
        print(f'Found {len(traces)} waveform files for day {jday}')
        if len(traces) == 0:
            print(f'Warning: No files found matching pattern {search_pattern}')
            print(f'Please verify your directory structure: datadir/YYYY/NETWORK/STATION/waveforms')
        st = obspy.Stream()
        for trace in traces:
            try:
                tr = read(trace)
                st += tr
            except Exception as e:
                print(f'Could not read file {trace}: {e}')
        print(f'📈 Loaded {len(st)} traces for day {jday}')
        # Verificar si hay datos suficientes para procesar
        if len(st) == 0:
            print(f'⚠️ No data found for day {jday}, skipping...')
            continue
        # Picking model
        print('Loading EQTransformer model...')
        model = sbm.EQTransformer.from_pretrained('original')
        if torch.cuda.is_available():
            model.cuda()
        
        batch_size = 10000
        
        if mode == 1:
            print('Extracting continuous probabilities...')
            probs_max = model.annotate(st, stacking="max", overlap=5500, 
                                      batch_size=batch_size)
            # Extraer picks desde las probabilidades
            print('Extracting picks from probabilities...')
            try:
                # Usar classify_aggregate con la sintaxis correcta
                picks_max = model.classify_aggregate(
                    probs_max, 
                    argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
                ).picks
            except (AttributeError, TypeError) as e:
                print(f'classify_aggregate not available or wrong syntax: {e}')
                # Fallback: usar classify()
                print('Using model.classify() as fallback...')
                picks_max = model.classify(st, stacking="max", overlap=5500, 
                                          P_threshold=0.05, S_threshold=0.05, 
                                          batch_size=batch_size)
            
            print('Saving probability curves as SAC files...')
            save_probability_curves_sac(probs_max, [], jday, year)
            if not exists(join(pickspath,year)):
                makedirs(join(pickspath,year))
            # Save picks to CSV
            print(f'Saving picks for day {jday}...')
            picks_max_dicts = [pick.__dict__ for pick in picks_max]
            picks_max_df = pd.DataFrame(picks_max_dicts)
            picks_max_df.to_csv(join(pickspath, year, f'picks_max_{year}_{jday}.csv'), index=False)
                
        if mode == 2:
            print('Extracting continuous probabilities...')
            probs_avg = model.annotate(st, stacking="avg", overlap=5500, 
                                      batch_size=batch_size)
            # Extraer picks desde las probabilidades
            print('Extracting picks from probabilities...')
            try:
                # Usar classify_aggregate con la sintaxis correcta
                picks_avg = model.classify_aggregate(
                    probs_avg, 
                    argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
                ).picks
            except (AttributeError, TypeError) as e:
                print(f'classify_aggregate not available or wrong syntax: {e}')
                # Fallback: usar classify()
                print('Using model.classify() as fallback...')
                picks_avg = model.classify(st, stacking="avg", overlap=5500, 
                                          P_threshold=0.05, S_threshold=0.05, 
                                          batch_size=batch_size)
            
            print('Saving probability curves as SAC files...')
            save_probability_curves_sac([], probs_avg, jday, year)
            if not exists(join(pickspath,year)):
                makedirs(join(pickspath,year))
            # Save picks to CSV
            print(f'Saving picks for day {jday}...')
            picks_avg_dicts = [pick.__dict__ for pick in picks_avg]
            picks_avg_df = pd.DataFrame(picks_avg_dicts)
            picks_avg_df.to_csv(join(pickspath, year, f'picks_avg_{year}_{jday}.csv'), index=False)
        
        # Clear memory after each day
        del st, model
        if mode == 1:
            del picks_max_df
        if mode == 2:
            del picks_avg_df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Return the results from the last day processed
    if mode == 1:
        return picks_max, probs_max
    else:  # mode == 2
        return picks_avg, probs_avg

def pick_phases_testing(time_period,yearpath):
    # max_picks = DataFrame()
    # avg_picks = DataFrame()
    # Extraer el año del path
    year = yearpath.split('/')[-1]
    for i,jday in enumerate(time_period[1][:]):
        # Reading data
        print(f'Reading data from {time_period[0][i]}')
        # Updated pattern to match datadir/YYYY/NETWORK/STATION/waveforms structure
        search_pattern = join(yearpath,f'*/*/*.{jday}')
        print(f'Searching for files with pattern: {search_pattern}')
        traces = glob(search_pattern)
        print(f'Found {len(traces)} waveform files for day {jday}')
        if len(traces) == 0:
            print(f'Warning: No files found matching pattern {search_pattern}')
            print(f'Please verify your directory structure: datadir/YYYY/NETWORK/STATION/waveforms')
        st = obspy.Stream()
        for trace in traces:
            try:
                tr = read(trace)
                st += tr
                #st = read(join(yearpath,f'*/*.{jday}'))
            except Exception as e:
                print(f'Could not read file {trace}: {e}')
        print(f'📈 Loaded {len(st)} traces for day {jday}')
        # Verificar si hay datos suficientes para procesar
        if len(st) == 0:
            print(f'⚠️ No data found for day {jday}, skipping...')
            continue
        # Picking model
        print('Loading EQTransformer model...')
        model = sbm.EQTransformer.from_pretrained('original')
        if torch.cuda.is_available():
            model.cuda()
        # Phase picking
        # print('Picking phases...')
        # Estrategias de picking: "max" (máximo pico) y "avg" (promedio)
        # batch_size = min(5000, len(st) * 100)  # Reducir batch size dinámicamente
        batch_size = 10000 
        # Use annotate() to get continuous probabilities
        print('Extracting continuous probabilities...')
        probs_max = model.annotate(st, stacking="max", overlap=5500, 
                                  batch_size=batch_size)
        probs_avg = model.annotate(st, stacking="avg", overlap=5500, 
                                  batch_size=batch_size)
        # Extract picks from probabilities
        print('Extracting picks from probabilities...')
        try:
            # Usar classify_aggregate con la sintaxis correcta
            picks_max = model.classify_aggregate(
                probs_max, 
                argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
            ).picks
            # print("Inspecting pick object attributes:")
            # print(dir(picks_max[0]))
            # print("Pick content:", picks_max[0])
            picks_avg = model.classify_aggregate(
                probs_avg, 
                argdict={'P_threshold': 0.05, 'S_threshold': 0.05}
            ).picks
        except (AttributeError, TypeError) as e:
            print(f'classify_aggregate not available or wrong syntax: {e}')
            # Fallback: usar classify() 
            print('Using model.classify() as fallback...')
            picks_max = model.classify(st, stacking="max", overlap=5500, 
                                      P_threshold=0.05, S_threshold=0.05, 
                                      batch_size=batch_size)
            picks_avg = model.classify(st, stacking="avg", overlap=5500, 
                                      P_threshold=0.05, S_threshold=0.05, 
                                      batch_size=batch_size)
        print('Saving P probability curves as SAC files...')
        save_probability_curves_sac(probs_max, probs_avg, jday, year)
        # Save each day's picks dataframes
        print(f'Saving picks for day {jday}...')
        # Save picks to CSV files
        if not exists(join(pickspath,year)):
            makedirs(join(pickspath,year))
        # Max stacking picks
        picks_max_dicts = [pick.__dict__ for pick in picks_max]
        picks_max_df = pd.DataFrame(picks_max_dicts)
        picks_max_df.to_csv(join(pickspath, year, f'picks_max_{year}_{jday}.csv'), index=False)
        # Avg stacking picks
        picks_avg_dicts = [pick.__dict__ for pick in picks_avg]
        picks_avg_df = pd.DataFrame(picks_avg_dicts)
        picks_avg_df.to_csv(join(pickspath, year, f'picks_avg_{year}_{jday}.csv'), index=False)

        # Clear memory after each day
        del st, model, picks_max_df, picks_avg_df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Return the results from the last day processed
    return picks_max, picks_avg, probs_max, probs_avg

# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------

if __name__ == "__main__":
    codestart = datetime.now()
    
    # Start resource monitoring
    log_file = '/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/OUT_LOGS/EQT_detection.out'
    start_resource_monitoring(log_file, interval=10)  # Log every 10 seconds
    
    try:
        ### DIRECTORIES & FILES ###
        montalca = r'/Volumes/GeoPhysics_49/users-data/montalca'
        catalogs = join(montalca,'CATALOGS/EQT_PICKS')
        pickspath = join(catalogs,'PICKS')
        probspath = join(catalogs,'PROBS')
        print('')
        basedir = input('Write basedir path: ')
        datapth = input('Write datadir path: ')
        datadir = join(basedir,datapth)
        print('')

        ### FLAGS ###
        # 1 = Gets max picks and probs; 2 = Gets avg picks and probs;
        # 3 = Testing mode (Gets max and avg picks and probs)
        mode = int(input('Select mode (1: max picks and probs, 2: avg picks and probs, 3: test (both methods)): '))

        ### PICKING (EQTransformer)###
        # Time period
        print('')
        print('Time period has to be from the same year')
        s_date = input('Start date (format yyyy-mm-dd): ')
        e_date = input('End date (format yyyy-mm-dd): ')
        print('')
        time_period,year = dates(s_date,e_date)
        yearpath = join(datadir,year)
        final_catalog = DataFrame()
        # Picking phases
        if mode == 1 or mode == 2:
            picks,probs = pick_phases(time_period, yearpath)
        if mode == 3:
            picks_max,picks_avg,probs_max,probs_avg = pick_phases_testing(time_period, yearpath)
        
        codestop = datetime.now()
        print(f'Code execution time: {codestop - codestart}')
        
    except Exception as e:
        print(f"Error during execution: {e}")
        raise
    finally:
        # Stop resource monitoring
        stop_resource_monitoring()