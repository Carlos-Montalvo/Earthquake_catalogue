# Standard library imports
import datetime
import gc
import warnings
from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import Union

# Third-party imports
import pandas as pd
import pytz

# Scientific/seismic imports
import pyocto
from obspy import read_inventory, UTCDateTime
from pandas import DataFrame, date_range, concat, read_excel, read_csv

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Reducir warnings innecesarios

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

# Transforms two dates to a date range and julian dates
def dates(s_date,e_date):
    year,sm,sd = s_date.split('-')
    yy,em,ed = e_date.split('-')
    date_format = '%Y-%m-%d'
    tperiod = date_range(start=s_date,end=e_date).strftime(date_format)
    jperiod = date_range(start=s_date,end=e_date).strftime('%j')
    time_period = ((tperiod,jperiod))
    return time_period,year,sm,sd,em,ed

# Converts UTC time to NZT (New Zealand Time)
def utc_to_nzt(utc_time):
    # Define NZ timezone
    nz_tz = pytz.timezone('Pacific/Auckland')
    # Convert UTCDateTime to datetime if needed
    if isinstance(utc_time, UTCDateTime):
        utc_dt = utc_time.datetime
    else:
        utc_dt = utc_time
    # Ensure the datetime is timezone-aware (UTC)
    if utc_dt.tzinfo is None:
        utc_dt = pytz.UTC.localize(utc_dt)
    # Convert to NZ time
    nzt_dt = utc_dt.astimezone(nz_tz)
    return nzt_dt

# Add UTC and NZT columns to events dataframe
def add_time_columns(events_df):
    # Create a copy to avoid modifying original
    df = events_df.copy()
    # Add UTC time column (formatted string)
    df['time_utc'] = df['time'].apply(lambda x: UTCDateTime(x).strftime('%Y-%m-%d,%H:%M:%S.%f')[:-3])
    # Add NZT time column (formatted string)
    df['time_nzt'] = df['time'].apply(lambda x: utc_to_nzt(UTCDateTime(x)).strftime('%Y-%m-%d,%H:%M:%S.%f')[:-3])
    # # Add timezone indicators
    # df['timezone_utc'] = 'UTC'
    # df['timezone_nzt'] = df['time'].apply(lambda x: utc_to_nzt(UTCDateTime(x)).strftime('%Z'))
    return df

# Save assignments to NonLinLoc format (modified version of the pyocto.associator.to_nonlinloc 17/06/2025)
# sta; instrument; channel; P phase onset; ph; 1st motion; yymmdd; hhmm; ss.ssss; error; errMag; coda duration; amp; period; priorWt
# e.g. ABC   ?   Z   ?   P   U   20220110    2359    43.500  GAU -1.00e+00   -1.00e+00   -1.00e+00   -1.00e+00   -1.00e+00
def assignments_to_nonlinloc(assignments: DataFrame, path: Union[str, Path]):
    # Read station information to determine channel type
    basedir = '/Volumes/GeoPhysics_49/users-data/montalca'
    stafile = read_excel(join(basedir, 'STATIONS/STATIONS.xlsx'), sheet_name='GEONET', header=0,
                        usecols=[2, 6], names=['code', 'type'])
    
    # Create dictionary for station types
    station_types = dict(zip(stafile.code, stafile.type))
    
    with open(path, "w") as f:
        for event_idx, event_catalog in assignments.groupby("event_idx"):
            f.write(f"PUBLIC_ID E{event_idx:08d}\n")
            for _, pick in event_catalog.iterrows():
                # Example: GRX    ?    ?    ? P      U 19940217 2216   44.9200 GAU  2.00e-02 -1.00e+00 -1.00e+00 -1.00e+00
                if isinstance(pick["time"], datetime.datetime):
                    time = pick["time"]
                else:
                    time = datetime.datetime.fromtimestamp(pick["time"], datetime.timezone.utc)
                phase = pick["phase"].upper()
                station = pick["station"]
                probability = pick["probability"]
                ntwk,sta,loc = station.split('.')
                daystr = time.strftime("%Y%m%d")
                hourmin = time.strftime("%H%M")
                second = time.strftime("%S.%f")[:-2]
                
                # Determine channel based on network and station type
                if phase == 'P' and ntwk == "DP":
                    f.write(
                        f"{sta} ? EHZ ? {phase} ? {daystr} {hourmin} {second} "
                        f"GAU {probability:.2e} -1.00e+00 -1.00e+00 -1.00e+00\n"
                    )
                elif phase == 'P' and ntwk == "NZ":
                    # Check station type for NZ network
                    station_type = station_types.get(sta, 'Unknown')
                    if station_type == 'Broadband':
                        channel = 'HHZ'
                    elif station_type == 'Short Period':
                        channel = 'EHZ'
                    else:
                        channel = 'HHZ'  # Default to HHZ if type unknown
                    
                    f.write(
                        f"{sta} ? {channel} ? {phase} ? {daystr} {hourmin} {second} "
                        f"GAU {probability:.2e} -1.00e+00 -1.00e+00 -1.00e+00\n")
                elif phase == 'P' and ntwk == "5L":
                    f.write(
                        f"{sta} ? HHZ ? {phase} ? {daystr} {hourmin} {second} "
                        f"GAU {probability:.2e} -1.00e+00 -1.00e+00 -1.00e+00\n")
                else:               
                    f.write(
                        f"{sta} ? ? ? {phase} ? {daystr} {hourmin} {second} "
                        f"GAU {probability:.2e} -1.00e+00 -1.00e+00 -1.00e+00\n")
            f.write("\n")

# Creates the PyOcto 1D velocity model and associate events with stations
def vel_model(vel_file,region):
    # try:
    print('Creating 1D velocity model...')
    # Getting vp and vs mean for each depth
    filtered_velfile = vel_file[vel_file.depth >= 0]
    
    # Additional filtering by region bounds (vectorized operation)
    region_mask = (
        (filtered_velfile.lat >= region[0]) & 
        (filtered_velfile.lat <= region[1]) & 
        (filtered_velfile.lon >= region[2]) & 
        (filtered_velfile.lon <= region[3])
    )
    filtered_velfile = filtered_velfile[region_mask]
    
    depths = filtered_velfile.depth.unique()
    vp_mean = []
    vs_mean = []
    
    for dpt in depths:
        # Use vectorized operations to filter by depth
        depth_mask = filtered_velfile.depth == dpt
        depth_data = filtered_velfile[depth_mask]
        
        if len(depth_data) > 0:
            vp_mean.append(depth_data.vp.mean())
            vs_mean.append(depth_data.vs.mean())
        else:
            # Handle case where no data exists for this depth
            vp_mean.append(float('nan'))
            vs_mean.append(float('nan'))
    # Velocity model
    model = DataFrame({'depth':depths,'vp':vp_mean,'vs':vs_mean})
    pyocto.VelocityModel1D.create_model(model=model,delta=0.5,xdist=300,zdist=750,
        path=join(vel_path,'1D_VModel.in'))
    velocity_model = pyocto.VelocityModel1D(path=join(vel_path,'1D_VModel.in'),
        surface_p_velocity=4.3,surface_s_velocity=2.5,
        tolerance=1.0,association_cutoff_distance=250)
    # except:
    #     print('Error creating 1D velocity model, using 0D velocity model...')
    #     velocity_model = pyocto.VelocityModel0D(p_velocity=6.5,s_velocity=3.7,
    #     tolerance=1.0,association_cutoff_distance=250)   
    return velocity_model

# Associates picks with stations
def associate_picks(time_period,yearpath,picks_type:str,sm,sd,em,ed):
    # Region for velocity model
    region = [-45.3, -38.3, 170, 177.6]  # (lat_min, lat_max, lon_min, lon_max)
    nll_region = [-43.8,-39.3,171.2,176.1]  # (lat_min, lat_max, lon_min, lon_max)
    interest_region = [-43,-40.5,172,175.6]  # (lat_min, lat_max, lon_min, lon_max)

    final_event_catalog = DataFrame()
    final_assignment_catalog = DataFrame()
    year = yearpath.split('/')[-1]
    global_event_idx = 0  # Global counter for continuous event indexing

    for i,jday in enumerate(time_period[1][:]):
        print('')
        # Reading picks
        print(f'Reading picks for day: {jday}')    
        # Avg picks
        if picks_type == 'avg' and exists(join(pickspath,year,f'picks_avg_{year}_{jday}.csv')):
            picks_file_path = join(pickspath,year,f'picks_avg_{year}_{jday}.csv')
            #print(f'Reading avg picks from: {picks_file_path}')
            picks = read_csv(picks_file_path,
                header=0,usecols=[0,3,4,5],
                names=['station','time','probability','phase'])
            #print(f'Loaded {len(picks)} picks')
            #print(f'Sample picks:\n{picks.head()}')
        # Max picks
        elif picks_type == 'max' and exists(join(pickspath,year,f'picks_max_{year}_{jday}.csv')):
            picks_file_path = join(pickspath,year,f'picks_max_{year}_{jday}.csv')
            #print(f'Reading max picks from: {picks_file_path}')
            picks = read_csv(picks_file_path,
                header=0,sep=',',usecols=[0,3,4,5],
                names=['station','time','probability','phase'])
            #print(f'Loaded {len(picks)} picks')
            #print(f'Sample picks:\n{picks.head()}')
        # Kurtosis picks
        elif picks_type == 'kurtosis' and exists(join(krts_path,year,f'picks_max_{year}_{jday}_retained.csv')):
            picks_raw = read_csv(join(krts_path,year,f'picks_max_{year}_{jday}_retained.csv'),
                header=0,sep=',',usecols=[1,2,4,5,6,7],
                names=['network','station','location','phase','time','probability'])
            # Combine network.station.location into station column for PyOcto compatibility
            picks = picks_raw.copy()
            # Convert to string and handle NaN values
            picks_raw['network'] = picks_raw['network'].fillna('').astype(str)
            picks_raw['station'] = picks_raw['station'].fillna('').astype(str)
            # For location, convert float to int first to avoid .0 suffix
            picks_raw['location'] = picks_raw['location'].fillna('').apply(
                lambda x: str(int(float(x))) if x != '' and pd.notna(x) else str(x)
            )
            picks['station'] = picks_raw['network'] + '.' + picks_raw['station'] + '.' + picks_raw['location']
            # Keep only the columns needed for PyOcto
            picks = picks[['station','time','probability','phase']]
        else:
            print(f'No picks available for {year} day {jday}')
            continue
        
        # Convert time column to timestamp for PyOcto compatibility
        print(f'Converting time format for PyOcto...')
        picks['time'] = pd.to_datetime(picks['time']).apply(lambda x: x.timestamp())
        print(f'Sample converted times: {picks["time"].head().tolist()}')
        
        # Setting the velocity model
        # velocity_model = vel_model(vel_file,region)
        # velocity_model = vel_model(vel_file,nll_region)
        velocity_model = vel_model(vel_file,interest_region)
        # Associator
        associator = pyocto.OctoAssociator.from_area(lat=(region[0],region[1]),lon=(region[2],region[3]),zlim=(0,750),
            time_before=300,velocity_model=velocity_model,
            n_picks=12,n_p_and_s_picks=5,
            min_node_size=10, # increases runtime, e.g. 10 -> 1 min, 5 -> 5 min
            min_node_size_location=0.1, # improves location in order of 0.001 degrees, increases runtime e.g. 0.1 -> 1.5 min
            pick_match_tolerance=1.5, # max diff time between predicted and observed phase time, increases runtime
            refinement_iterations=10, # improves accuracy, increases runtime
            time_slicing=350, # increases memory, reduces runtime
            location_split_depth=6, # increases accuracy, increases runtime dramatically e.g. 6 -> 1 min, 12 -> 17.5 min
            location_split_return=5,
            n_threads=20, # Number of cores to be use
            )
        # Stations
        stations = associator.inventory_to_df(inv)
        # Associate picks with stations
        print('Associating picks with stations...')
        events,assignments = associator.associate(picks,stations)
        associator.transform_events(events)
        events_df = DataFrame(events)
        assignments_df = DataFrame(assignments)

        # Saving events and assignments per day
        if picks_type == 'avg' or picks_type == 'max':
            if not exists(join(pyocto_dir,f'RAW_EVENTS/{year}')):
                makedirs(join(pyocto_dir,f'RAW_EVENTS/{year}'))
            if not exists(join(pyocto_dir,f'RAW_ASSIGNMENTS/{year}')):
                makedirs(join(pyocto_dir,f'RAW_ASSIGNMENTS/{year}'))
            events_df.to_csv(join(pyocto_dir,f'RAW_EVENTS/{year}/{datapth}_{year}_EVENTS_{jday}.csv'), index=False)
            assignments_df.to_csv(join(pyocto_dir,f'RAW_ASSIGNMENTS/{year}/{datapth}_{year}_ASSIGNMENTS_{jday}.csv'), index=False)
        else:
            if not exists(join(pyocto_dir,f'EVENTS/{year}')):
                makedirs(join(pyocto_dir,f'EVENTS/{year}'))
            if not exists(join(pyocto_dir,f'ASSIGNMENTS/{year}')):
                makedirs(join(pyocto_dir,f'ASSIGNMENTS/{year}'))
            events_df.to_csv(join(pyocto_dir,f'EVENTS/{year}/{datapth}_{year}_EVENTS_{jday}.csv'), index=False)
            assignments_df.to_csv(join(pyocto_dir,f'ASSIGNMENTS/{year}/{datapth}_{year}_ASSIGNMENTS_{jday}.csv'), index=False)

        print(f'Events in day {jday}: {len(events_df.idx)}')
        
        # Update event indices to continue global numbering
        if len(events_df) > 0:
            # Update events dataframe
            events_df['idx'] = range(global_event_idx, global_event_idx + len(events_df))
            
            # Update assignments dataframe - map old event_idx to new global event_idx
            if len(assignments_df) > 0:
                # Create mapping from old event_idx to new global event_idx
                old_to_new_idx = {}
                for i, old_idx in enumerate(events_df['idx'].tolist()):
                    # The old event_idx starts from 0 for each day
                    old_to_new_idx[i] = global_event_idx + i
                
                # Update event_idx in assignments using the mapping
                assignments_df['event_idx'] = assignments_df['event_idx'].map(old_to_new_idx)
            
            global_event_idx += len(events_df)
        
        events_df_wtime = add_time_columns(events_df)  # Add time columns
        final_event_catalog = concat([final_event_catalog, events_df_wtime], ignore_index=True)
        final_assignment_catalog = concat([final_assignment_catalog, assignments_df], ignore_index=True)
                   
        # Clear memory
        print('Cleaning up memory after processing day...')
        del events, assignments, stations, picks
        gc.collect()  # Forzar limpieza de memoria
    
    # Saving assignments to NonLinLoc format
    path = join(nll_path,f'{datapth}_{year}_{sm}{sd}_{em}{ed}.obs')
    assignments_to_nonlinloc(final_assignment_catalog,path)
    # Save final catalog
    event_catalog_path = join(catalogs,f'{datapth}_{year}_CATALOG.csv')
    assignment_catalog_path = join(catalogs,f'{datapth}_{year}_ASSIGNMENTS.csv')
    final_event_catalog.to_csv(event_catalog_path, index=False)
    final_assignment_catalog.to_csv(assignment_catalog_path, index=False)
    # Display summary of time columns
    if len(final_event_catalog) > 0:
        print(f"Final catalog summary:")
        print(f"Total events: {len(final_event_catalog)}")

    return None

# ------------------------------------------------------------------------------
# Main script starts here
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    ### DIRECTORIES & FILES ###
    codestart = datetime.datetime.now()
    montalca = r'/Volumes/GeoPhysics_49/users-data/montalca'
    nll_path = join(montalca,'NLL')
    catalogs = join(montalca,'CATALOGS')
    eqt_picks = join(catalogs,'EQT_PICKS')
    pyocto_dir = join(catalogs,'PYOCTO')
    pickspath = join(eqt_picks,'PICKS')
    probspath = join(eqt_picks,'PROBS')
    krts_path = join(eqt_picks,'KURTOSIS')
    vel_path = join(montalca,'VEL_MODEL')
    # All stations
    inv = read_inventory(join(montalca,'STATIONS/all_stations.xml'))
    # DPRI stations
    # inv = read_inventory(join(montalca,'STATIONS/DP_DPRI.xml'))
    vel_file = read_excel(join(vel_path,'VELOCITY_MODEL.xlsx'),header=0,
        usecols=[0,2,8,9,10],names=['vp','vs','depth','lat','lon'])
    print('Directory structure: basedir/datadir/year/stations/data')
    print('')
    basedir = input('Write basedir path: ')
    datapth = input('Write datadir path: ')
    datadir = join(basedir,datapth)
    print('')

    ### ASSOCIATING (PyOcto) ###
    # Time period
    print('Time period has to be from the same year')
    s_date = input('Start date (format yyyy-mm-dd): ')
    e_date = input('End date (format yyyy-mm-dd): ')
    time_period,year,sm,sd,em,ed = dates(s_date,e_date)
    yearpath = join(datadir,year)
    final_catalog = DataFrame()
    # Associationg and ploting events
    associate_picks(time_period,yearpath,"kurtosis",sm,sd,em,ed)
    # associate_picks(time_period,yearpath,"max",sm,sd,em,ed)
    codestop = datetime.datetime.now()
    print(f'Code execution time: {codestop - codestart}')