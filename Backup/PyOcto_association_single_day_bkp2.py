#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PYOCTO ASSOCIATION - Single Day Processing Version

This script performs seismic event association using PyOcto for a single day.
Adapted from the original PyOcto_association.py to work with individual days
and can be called from shell scripts.

WORKFLOW:
1. Reads EQT picks from CSV files (max, avg, or kurtosis)
2. Creates 1D velocity model for the region
3. Associates picks using PyOcto algorithm
4. Saves events and assignments for the day
5. Exports results to NonLinLoc format

USAGE:
    python PyOcto_association_single_day.py <basedir> <datadir> <year> <julian_day> <picks_type>

EXAMPLE:
    python PyOcto_association_single_day.py /Volumes/GeoPhysics_49/users-data/montalca DATA 2025 032 kurtosis

PICKS_TYPE options:
    - "max": Uses picks_max_YYYY_JDD.csv files
    - "avg": Uses picks_avg_YYYY_JDD.csv files  
    - "kurtosis": Uses picks_max_YYYY_JDD_retained.csv files from kurtosis processing

@author: montalca (adapted from original PyOcto_association.py)
@date: October 2025
@version: 1.0 - Single Day Processing Edition
"""

# Standard library imports
import datetime
import gc
import warnings
from os import makedirs
from os.path import join, exists
from pathlib import Path
from typing import Union
import sys
import os

# Third-party imports
import pandas as pd
import pytz

# Scientific/seismic imports
import pyocto
from obspy import read_inventory, UTCDateTime
from pandas import DataFrame, date_range, concat, read_excel, read_csv

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Reducir warnings innecesarios

# Simple color codes for terminal output
class colours:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    PURPLE = '\033[95m'
    ENDC = '\033[0m'

#-------------------------------------------------------------------------------
# FUNCTIONS
#-------------------------------------------------------------------------------

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
    return df

# Save assignments to NonLinLoc format (modified version of the pyocto.associator.to_nonlinloc 17/06/2025)
def assignments_to_nonlinloc(assignments: DataFrame, path: Union[str, Path], basedir: str):
    # Read station information to determine channel type
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

# Removes picks that have already been associated in previous iterations
def pick_removal(catalog, assignments, pick_df):
    """
    Remove picks that have been associated to events in previous iterations.
    
    Parameters:
    -----------
    catalog : DataFrame
        Events from the association
    assignments : DataFrame
        Pick assignments from the association
    pick_df : DataFrame
        Original picks dataframe
        
    Returns:
    --------
    DataFrame : Filtered picks dataframe with associated picks removed
    """
    if len(catalog) == 0 or len(assignments) == 0:
        return pick_df
    
    pick_df_new = pick_df.copy()
    deleted_indices = []
    
    # Get all pick indices that were associated to events
    for i in range(len(catalog)):
        event_index = catalog["idx"].iloc[i]
        associated_picks = assignments.loc[assignments["event_idx"] == event_index]
        
        for pick in range(len(associated_picks)):
            pick_index = associated_picks["pick_idx"].iloc[pick]
            if pick_index not in deleted_indices:
                deleted_indices.append(pick_index)
    
    # Sort indices in reverse order for proper deletion
    deleted_indices = sorted(deleted_indices, reverse=True)
    
    # Drop the associated picks
    pick_df_new = pick_df_new.drop(deleted_indices, errors='ignore')
    
    # Reset indices for the next iteration
    pick_df_new = pick_df_new.reset_index(drop=True)
    
    print(f'{colours.YELLOW}Removed {len(deleted_indices)} associated picks, {len(pick_df_new)} picks remaining{colours.ENDC}')
    
    return pick_df_new

# Creates the PyOcto 1D velocity model and associate events with stations
def vel_model(vel_file, region, vel_path):
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
    
    return velocity_model

# Associates picks for a single day
def associate_picks_single_day(julian_day, year, picks_type, basedir, datadir, 
                              vel_file, vel_path, inv, pickspath, krts_path, 
                              pyocto_dir, nll_path, catalogs):
    """
    Associate picks for a single day using PyOcto.
    
    Parameters:
    -----------
    julian_day : str
        Julian day to process (e.g., "032")
    year : str
        Year to process (e.g., "2025")
    picks_type : str
        Type of picks to use ("max", "avg", or "kurtosis")
    basedir : str
        Base directory path
    datadir : str
        Data directory name
    vel_file : DataFrame
        Velocity model data
    vel_path : str
        Velocity model path
    inv : obspy.Inventory
        Station inventory
    pickspath : str
        Path to picks directory
    krts_path : str
        Path to kurtosis results directory
    pyocto_dir : str
        PyOcto output directory
    nll_path : str
        NonLinLoc output path
    catalogs : str
        Catalogs directory
        
    Returns:
    --------
    dict : Processing results summary
    """
    
    # Region definitions
    region = [-45.3, -38.3, 170, 177.6]  # (lat_min, lat_max, lon_min, lon_max)
    nll_region = [-43.8, -39.3, 171.2, 176.1]  # (lat_min, lat_max, lon_min, lon_max)
    interest_region = [-43, -40.5, 172, 175.6]  # (lat_min, lat_max, lon_min, lon_max)

    print(f'{colours.CYAN}Reading picks for day: {julian_day}{colours.ENDC}')
    
    # Load picks based on type
    picks = None
    picks_file_path = None
    
    if picks_type == 'avg':
        picks_file_path = join(pickspath, year, f'picks_avg_{year}_{julian_day}.csv')
        if exists(picks_file_path):
            picks = read_csv(picks_file_path,
                header=0, usecols=[0,3,4,5],
                names=['station','time','probability','phase'])
    
    elif picks_type == 'max':
        picks_file_path = join(pickspath, year, f'picks_max_{year}_{julian_day}.csv')
        if exists(picks_file_path):
            picks = read_csv(picks_file_path,
                header=0, sep=',', usecols=[0,3,4,5],
                names=['station','time','probability','phase'])
    
    elif picks_type == 'kurtosis':
        picks_file_path = join(krts_path, year, f'picks_max_{year}_{julian_day}_retained.csv')
        if exists(picks_file_path):
            picks_raw = read_csv(picks_file_path,
                header=0, sep=',', usecols=[1,2,4,5,6,7],
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
    
    if picks is None or len(picks) == 0:
        error_msg = f'No picks available for {year} day {julian_day} with type {picks_type}'
        print(f'{colours.YELLOW}{error_msg}{colours.ENDC}')
        return {'success': False, 'error': error_msg}
    
    print(f'{colours.GREEN}Loaded {len(picks)} picks from {picks_file_path}{colours.ENDC}')
    
    # Convert time column to timestamp for PyOcto compatibility
    print(f'Converting time format for PyOcto...')
    picks['time'] = pd.to_datetime(picks['time']).apply(lambda x: x.timestamp())
    print(f'Sample converted times: {picks["time"].head().tolist()}')
    
    # Setting the velocity model
    velocity_model = vel_model(vel_file, interest_region, vel_path)
    
    # Initialize lists to accumulate all events and assignments across iterations
    all_events = []
    all_assignments = []
    current_picks = picks.copy()
    iteration_count = 0
    
    # Get stations from initial associator (needed for all iterations)
    initial_associator = pyocto.OctoAssociator.from_area(
        lat=(region[0],region[1]), lon=(region[2],region[3]), zlim=(0,750),
        time_before=300, velocity_model=velocity_model,
        n_picks=150, n_p_picks=75, n_s_picks=75,
        min_node_size=10, min_node_size_location=0.1,
        pick_match_tolerance=5, refinement_iterations=10,
        time_slicing=350, location_split_depth=6,
        location_split_return=5, n_threads=10,
    )
    stations = initial_associator.inventory_to_df(inv)
    
    # =========================================================================
    # PHASE 1: n_picks from 150 to 22, step -2 (n_p_picks=n_s_picks=n_picks/2)
    # pick_match_tolerance = 5
    # =========================================================================
    print(f'{colours.CYAN}=== PHASE 1: Large events (n_picks 150->22) ==={colours.ENDC}')
    
    for n_picks in range(150, 20, -2):
        if len(current_picks) == 0:
            print(f'{colours.YELLOW}No more picks remaining, stopping iterations{colours.ENDC}')
            break
            
        n_p_picks = n_picks // 2
        n_s_picks = n_picks // 2
        iteration_count += 1
        
        iter_start_time = datetime.datetime.now()
        print(f'{colours.CYAN}Iteration {iteration_count}: n_picks={n_picks}, n_p={n_p_picks}, n_s={n_s_picks}, tolerance=5{colours.ENDC}')
        
        associator = pyocto.OctoAssociator.from_area(
            lat=(region[0],region[1]), lon=(region[2],region[3]), zlim=(0,750),
            time_before=300, velocity_model=velocity_model,
            n_picks=n_picks, n_p_picks=n_p_picks, n_s_picks=n_s_picks,
            min_node_size=10, min_node_size_location=0.1,
            pick_match_tolerance=5, refinement_iterations=10,
            time_slicing=350, location_split_depth=6,
            location_split_return=5, n_threads=10,
        )
        
        events, assignments = associator.associate(current_picks, stations)
        
        if len(events) > 0:
            associator.transform_events(events)
            events_df_iter = DataFrame(events)
            assignments_df_iter = DataFrame(assignments)
            
            print(f'{colours.GREEN}  Found {len(events_df_iter)} events{colours.ENDC}')
            
            # Accumulate events and assignments
            all_events.append(events_df_iter)
            all_assignments.append(assignments_df_iter)
            
            # Remove associated picks for next iteration
            current_picks = pick_removal(events_df_iter, assignments_df_iter, current_picks)
        else:
            print(f'{colours.YELLOW}  No events found in this iteration{colours.ENDC}')
        
        iter_end_time = datetime.datetime.now()
        iter_duration = iter_end_time - iter_start_time
        print(f'{colours.PURPLE}  Iteration time: {iter_duration}{colours.ENDC}')
    
    # =========================================================================
    # PHASE 2: n_p_and_s_picks from 20 to 5, step -1
    # pick_match_tolerance = 1.5
    # =========================================================================
    print(f'{colours.CYAN}=== PHASE 2: Medium events (n_p_and_s_picks 20->5) ==={colours.ENDC}')
    
    for n_p_and_s in range(20, 4, -1):
        if len(current_picks) == 0:
            print(f'{colours.YELLOW}No more picks remaining, stopping iterations{colours.ENDC}')
            break
            
        iteration_count += 1
        
        iter_start_time = datetime.datetime.now()
        print(f'{colours.CYAN}Iteration {iteration_count}: n_p_and_s_picks={n_p_and_s}, tolerance=1.5{colours.ENDC}')
        
        associator = pyocto.OctoAssociator.from_area(
            lat=(region[0],region[1]), lon=(region[2],region[3]), zlim=(0,750),
            time_before=300, velocity_model=velocity_model,
            n_picks=n_p_and_s, n_p_and_s_picks=n_p_and_s,
            min_node_size=10, min_node_size_location=0.1,
            pick_match_tolerance=1.5, refinement_iterations=10,
            time_slicing=350, location_split_depth=6,
            location_split_return=5, n_threads=10,
        )
        
        events, assignments = associator.associate(current_picks, stations)
        
        if len(events) > 0:
            associator.transform_events(events)
            events_df_iter = DataFrame(events)
            assignments_df_iter = DataFrame(assignments)
            
            print(f'{colours.GREEN}  Found {len(events_df_iter)} events{colours.ENDC}')
            
            # Accumulate events and assignments
            all_events.append(events_df_iter)
            all_assignments.append(assignments_df_iter)
            
            # Remove associated picks for next iteration
            current_picks = pick_removal(events_df_iter, assignments_df_iter, current_picks)
        else:
            print(f'{colours.YELLOW}  No events found in this iteration{colours.ENDC}')
        
        iter_end_time = datetime.datetime.now()
        iter_duration = iter_end_time - iter_start_time
        print(f'{colours.PURPLE}  Iteration time: {iter_duration}{colours.ENDC}')
    
    # =========================================================================
    # PHASE 3: Final iteration with n_picks=4, n_p_picks=4
    # pick_match_tolerance = 1.5
    # =========================================================================
    print(f'{colours.CYAN}=== PHASE 3: Small events (n_picks=4, n_p_picks=4) ==={colours.ENDC}')
    
    if len(current_picks) > 0:
        iteration_count += 1
        
        iter_start_time = datetime.datetime.now()
        print(f'{colours.CYAN}Iteration {iteration_count}: n_picks=4, n_p_picks=4, tolerance=1.5{colours.ENDC}')
        
        associator = pyocto.OctoAssociator.from_area(
            lat=(region[0],region[1]), lon=(region[2],region[3]), zlim=(0,750),
            time_before=300, velocity_model=velocity_model,
            n_picks=4, n_p_picks=4,
            min_node_size=10, min_node_size_location=0.1,
            pick_match_tolerance=1.5, refinement_iterations=10,
            time_slicing=350, location_split_depth=6,
            location_split_return=5, n_threads=10,
        )
        
        events, assignments = associator.associate(current_picks, stations)
        
        if len(events) > 0:
            associator.transform_events(events)
            events_df_iter = DataFrame(events)
            assignments_df_iter = DataFrame(assignments)
            
            print(f'{colours.GREEN}  Found {len(events_df_iter)} events{colours.ENDC}')
            
            # Accumulate events and assignments
            all_events.append(events_df_iter)
            all_assignments.append(assignments_df_iter)
        else:
            print(f'{colours.YELLOW}  No events found in final iteration{colours.ENDC}')
        
        iter_end_time = datetime.datetime.now()
        iter_duration = iter_end_time - iter_start_time
        print(f'{colours.PURPLE}  Iteration time: {iter_duration}{colours.ENDC}')
    
    # =========================================================================
    # Combine all events and assignments
    # =========================================================================
    print(f'{colours.CYAN}=== Combinando resultados de {iteration_count} iteraciones ==={colours.ENDC}')
    
    if len(all_events) > 0:
        events_df = concat(all_events, ignore_index=True)
        assignments_df = concat(all_assignments, ignore_index=True)
        # Re-index events para IDs únicos
        events_df['idx'] = range(len(events_df))
        # Actualizar event_idx en assignments_df para que coincida con el nuevo idx global
        # Primero, obtener el mapeo de los índices originales a los nuevos
        original_indices = []
        for df in all_events:
            if 'idx' in df.columns:
                original_indices.extend(df['idx'].tolist())
            else:
                original_indices.extend([i for i in range(len(df))])
        new_indices = list(range(len(events_df)))
        idx_mapping = dict(zip(original_indices, new_indices))
        if 'event_idx' in assignments_df.columns:
            assignments_df['event_idx'] = assignments_df['event_idx'].map(idx_mapping)
    else:
        events_df = DataFrame()
        assignments_df = DataFrame()
    
    print(f'{colours.GREEN}Total de eventos encontrados para el día {julian_day}: {len(events_df)}{colours.ENDC}')
    
    # Create output directories
    if picks_type == 'avg' or picks_type == 'max':
        events_dir = join(pyocto_dir, f'RAW_EVENTS/{year}')
        assignments_dir = join(pyocto_dir, f'RAW_ASSIGNMENTS/{year}')
    else:
        events_dir = join(pyocto_dir, f'EVENTS/{year}')
        assignments_dir = join(pyocto_dir, f'ASSIGNMENTS/{year}')
    
    makedirs(events_dir, exist_ok=True)
    makedirs(assignments_dir, exist_ok=True)
    
    # Save events and assignments for this day
    datapth = os.path.basename(datadir)
    events_file = join(events_dir, f'{datapth}_{year}_EVENTS_{julian_day}_iter.csv')
    assignments_file = join(assignments_dir, f'{datapth}_{year}_ASSIGNMENTS_{julian_day}_iter.csv')
    
    events_df.to_csv(events_file, index=False)
    assignments_df.to_csv(assignments_file, index=False)
    
    print(f'{colours.GREEN}Events saved to: {events_file}{colours.ENDC}')
    print(f'{colours.GREEN}Assignments saved to: {assignments_file}{colours.ENDC}')
    
    # Add time columns for final output (only if there are events)
    if len(events_df) > 0:
        events_df_wtime = add_time_columns(events_df)
    
    # Save to NonLinLoc format
    nll_file = None
    if len(assignments_df) > 0:
        nll_file = join(nll_path, f'{datapth}_{year}_{julian_day}.obs')
        assignments_to_nonlinloc(assignments_df, nll_file, basedir)
        print(f'{colours.GREEN}NonLinLoc file saved to: {nll_file}{colours.ENDC}')
    
    # Clean up memory
    print('Cleaning up memory after processing day...')
    del all_events, all_assignments, current_picks, stations, picks
    gc.collect()
    
    return {
        'success': True,
        'events_count': len(events_df),
        'assignments_count': len(assignments_df),
        'events_file': events_file,
        'assignments_file': assignments_file,
        'nll_file': nll_file
    }

# ------------------------------------------------------------------------------
# MAIN PROCESSING FUNCTION FOR SINGLE DAY
# ------------------------------------------------------------------------------

def process_single_day(basedir, datadir, year, julian_day, picks_type):
    """
    Process a single day using PyOcto association.
    
    Parameters:
    -----------
    basedir : str
        Base directory path
    datadir : str
        Data directory name
    year : str
        Year to process
    julian_day : str
        Julian day to process
    picks_type : str
        Type of picks to use ("max", "avg", or "kurtosis")
        
    Returns:
    --------
    dict : Processing results summary
    """
    
    print(f"{colours.CYAN}=== PYOCTO ASSOCIATION SINGLE DAY PROCESSING ==={colours.ENDC}")
    print(f"Processing day {julian_day} for year {year}")
    print(f"Base directory: {basedir}")
    print(f"Data directory: {datadir}")
    print(f"Picks type: {picks_type}")
    
    # Setup paths
    datadir_full = join(basedir, datadir)
    nll_path = join(basedir, 'NLL')
    catalogs = join(basedir, 'CATALOGS')
    eqt_picks = join(catalogs, 'EQT_PICKS')
    pyocto_dir = join(catalogs, 'PYOCTO')
    pickspath = join(eqt_picks, 'PICKS')
    probspath = join(eqt_picks, 'PROBS')
    krts_path = join(eqt_picks, 'KURTOSIS')
    vel_path = join(basedir, 'VEL_MODEL')
    
    # Validate paths
    if not os.path.exists(datadir_full):
        print(f'{colours.RED}ERROR: Data directory does not exist: {datadir_full}{colours.ENDC}')
        return {'success': False, 'error': f'Data directory not found: {datadir_full}'}
    
    if not os.path.exists(vel_path):
        print(f'{colours.RED}ERROR: Velocity model directory does not exist: {vel_path}{colours.ENDC}')
        return {'success': False, 'error': f'Velocity model directory not found: {vel_path}'}
    
    # Create output directories
    makedirs(nll_path, exist_ok=True)
    makedirs(catalogs, exist_ok=True)
    makedirs(pyocto_dir, exist_ok=True)
    
    # Load required files
    try:
        print(f'{colours.CYAN}Loading station inventory...{colours.ENDC}')
        inv = read_inventory(join(basedir, 'STATIONS/nll_region_all_stations.xml'))
        
        print(f'{colours.CYAN}Loading velocity model...{colours.ENDC}')
        vel_file = read_excel(join(vel_path, 'VELOCITY_MODEL.xlsx'), header=0,
            usecols=[0,2,8,9,10], names=['vp','vs','depth','lat','lon'])
        
    except Exception as e:
        error_msg = f'Error loading required files: {e}'
        print(f'{colours.RED}ERROR: {error_msg}{colours.ENDC}')
        return {'success': False, 'error': error_msg}
    
    # Validate picks type
    if picks_type not in ['max', 'avg', 'kurtosis']:
        error_msg = f'Invalid picks type: {picks_type}. Must be one of: max, avg, kurtosis'
        print(f'{colours.RED}ERROR: {error_msg}{colours.ENDC}')
        return {'success': False, 'error': error_msg}
    
    try:
        # Process the day
        results = associate_picks_single_day(
            julian_day=julian_day,
            year=year,
            picks_type=picks_type,
            basedir=basedir,
            datadir=datadir_full,
            vel_file=vel_file,
            vel_path=vel_path,
            inv=inv,
            pickspath=pickspath,
            krts_path=krts_path,
            pyocto_dir=pyocto_dir,
            nll_path=nll_path,
            catalogs=catalogs
        )
        
        return results
        
    except Exception as e:
        error_msg = f'Error processing day {julian_day}: {e}'
        print(f'{colours.RED}ERROR: {error_msg}{colours.ENDC}')
        return {'success': False, 'error': error_msg}

# ------------------------------------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------------------------------------

def main():
    """
    Main function to handle command line arguments and process single day.
    """
    if len(sys.argv) != 6:
        print(f"{colours.RED}Usage: python PyOcto_association_single_day.py <basedir> <datadir> <year> <julian_day> <picks_type>{colours.ENDC}")
        print("Example: python PyOcto_association_single_day.py /Volumes/GeoPhysics_49/users-data/montalca DATA 2025 032 kurtosis")
        print("Picks types: max, avg, kurtosis")
        sys.exit(1)
    
    basedir = sys.argv[1]
    datadir = sys.argv[2]
    year = sys.argv[3]
    julian_day = sys.argv[4]
    picks_type = sys.argv[5]
    
    # Validate inputs
    try:
        julian_int = int(julian_day)
        if julian_int < 1 or julian_int > 366:
            print(f"{colours.RED}ERROR: Julian day must be between 1 and 366{colours.ENDC}")
            sys.exit(1)
    except ValueError:
        print(f"{colours.RED}ERROR: Julian day must be a number{colours.ENDC}")
        sys.exit(1)
    
    # Ensure julian day is zero-padded
    julian_day = julian_day.zfill(3)
    
    print(f"{colours.GREEN}Starting PyOcto association for single day{colours.ENDC}")
    print(f"Parameters: basedir={basedir}, datadir={datadir}, year={year}, julian_day={julian_day}, picks_type={picks_type}")
    
    start_time = datetime.datetime.now()
    
    # Process the single day
    results = process_single_day(basedir, datadir, year, julian_day, picks_type)
    
    end_time = datetime.datetime.now()
    processing_time = end_time - start_time
    
    print(f"\n{colours.CYAN}=== PROCESSING SUMMARY ==={colours.ENDC}")
    print(f"Processing time: {processing_time}")
    
    if results['success']:
        print(f"{colours.GREEN}Processing completed successfully!{colours.ENDC}")
        print(f"Events found: {results['events_count']}")
        print(f"Assignments made: {results['assignments_count']}")
        print(f"Files created:")
        print(f"  Events: {results['events_file']}")
        print(f"  Assignments: {results['assignments_file']}")
        if results.get('nll_file'):
            print(f"  NonLinLoc: {results['nll_file']}")
        sys.exit(0)
    else:
        print(f"{colours.RED}Processing failed: {results['error']}{colours.ENDC}")
        sys.exit(1)

if __name__ == '__main__':
    main()