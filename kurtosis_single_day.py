#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KURTOSIS PICKER - EQT Pick Refinement Tool (SINGLE DAY VERSION)

This script refines EQT (Earthquake Transformer) picks using kurtosis analysis
for a single day processing. Adapted from the multiprocessing version to work
with individual days and can be called from shell scripts.

WORKFLOW:
1. Reads EQT picks from CSV files in the format:
   trace_id,start_time,end_time,peak_time,peak_value,phase
   
2. Groups nearby picks (within 4 seconds) for the same station and phase

3. For groups with multiple picks:
   - Applies kurtosis analysis using HOST picker
   - Calculates first derivative of kurtosis
   - Finds peaks in the derivative
   - Retains the EQT pick closest to kurtosis-derived time
   
4. For single picks: retains them without modification

5. Outputs two CSV files:
   - *_kurtosis.csv: Pure kurtosis-derived picks
   - *_retained.csv: Refined EQT picks

USAGE:
    python kurtosis_single_day.py <basedir> <datadir> <year> <julian_day>

EXAMPLE:
    python kurtosis_single_day.py /Volumes/GeoPhysics_49/users-data/montalca DATA 2023 001

@author: montalca (adapted from multiprocessing version)
@date: October 2025
@version: 1.0 - Single Day Processing Edition
"""

# Import packages
import numpy as np
from os.path import join, exists
import os
from glob import glob
from obspy import Stream, read
from obspy.core import Trace
from obspy.core import UTCDateTime
from obspy.core.event import Pick, WaveformStreamID, QuantityError
from obspy.core.event.base import Comment, CreationInfo
from obspy.core.event.resourceid import ResourceIdentifier
from host.picker import Host
from host import scaffold as HS
from eqcorrscan.utils.findpeaks import find_peaks_compiled
import pandas as pd
from pandas import DataFrame, date_range
from datetime import datetime, timedelta
import csv
import gc  # Garbage collector for memory management
import sys
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
# Import plot utilities for EQT probabilities
sys.path.append('/Volumes/GeoPhysics_49/users-data/montalca/PROGRAMS/PYTHON/PLOTS')
from plot_utils import read_eqt_probabilities

# Simple color codes for terminal output
class colours:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'

# -----------------------------------------------------------------------
# FUNCTIONS FROM MULTIPROCESSING VERSION
# -----------------------------------------------------------------------

def dates(s_date, e_date):
    year, sm, sd = s_date.split('-')
    yy, em, ed = e_date.split('-')
    date_format = '%Y-%m-%d'
    tperiod = date_range(start=s_date, end=e_date).strftime(date_format)
    jperiod = date_range(start=s_date, end=e_date).strftime('%j')
    time_period = ((tperiod, jperiod))
    return time_period, year

def discover_stations_for_day(yearpath, julian_day):
    """
    Discover available stations for a specific day efficiently.
    
    Parameters:
    -----------
    yearpath : str
        Path to year directory containing station subdirectories
    julian_day : str
        Julian day to search for
        
    Returns:
    --------
    list : List of available station codes
    """
    stations = set()
    
    # Pattern to find all waveform files for the day
    # Structure: yearpath/NETWORK/STATION/files.jday
    waveform_pattern = join(yearpath, f'*/*/*.{julian_day.zfill(3)}')
    waveform_files = glob(waveform_pattern)
    
    # Extract station names from file paths
    for file_path in waveform_files:
        # Get station directory name (parent of the file)
        station_dir = os.path.basename(os.path.dirname(file_path))
        stations.add(station_dir)
    
    return sorted(list(stations))

def load_station_day(station, julian_day, yearpath):
    """
    Load waveform data for a specific station and day.
    
    Parameters:
    -----------
    station : str
        Station code
    julian_day : str
        Julian day to load
    yearpath : str
        Path to year directory
        
    Returns:
    --------
    obspy.Stream : Stream containing station data for the day
    """
    if not julian_day:
        return Stream()
    
    st = Stream()
    # Pattern: yearpath/NETWORK/STATION/files.jday
    station_pattern = join(yearpath, f'*/{station}/*.{julian_day.zfill(3)}')
    station_files = glob(station_pattern)
    
    for file_path in station_files:
        try:
            tr = read(file_path, format="MSEED" if file_path.endswith('.mseed') else None)
            st += tr
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
    
    return st

def load_picks_for_station(station, julian_day, pickspath, year):
    """
    Load EQT picks for a specific station and day.
    
    Parameters:
    -----------
    station : str
        Station code
    julian_day : str
        Julian day
    pickspath : str
        Path to picks directory
    year : str
        Year
        
    Returns:
    --------
    list : List of ObsPy Pick objects for the station
    """
    picks_file = join(pickspath, year, f'picks_max_{year}_{julian_day}.csv')
    
    if not exists(picks_file):
        return []
    
    try:
        # Load all picks for the day
        all_picks = csv_2_ObsPy(picks_file)
        
        # Filter picks for this specific station
        station_picks = [pick for pick in all_picks 
                        if pick.waveform_id.station_code == station]
        
        return station_picks
        
    except Exception as e:
        print(f"Warning: Could not load picks for station {station}: {e}")
        return []

def save_station_results(station, julian_day, retained_picks, kurtosis_picks, ktsispath, year):
    """
    Save results for a specific station to individual CSV files (temporary).
    These will be consolidated later by consolidate_station_results().
    
    Parameters:
    -----------
    station : str
        Station code
    julian_day : str
        Julian day
    retained_picks : list
        Refined EQT picks
    kurtosis_picks : list
        Pure kurtosis picks
    ktsispath : str
        Output directory path
    year : str
        Year
    """
    if not retained_picks and not kurtosis_picks:
        return
    
    # Create temporary directory for individual station files
    temp_dir = join(ktsispath, "temp", year)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate temporary filenames for this station
    # Format: picks_max_YYYY_JDD_STATION_kurtosis.csv (temporary)
    base_name = f"picks_max_{year}_{julian_day.zfill(3)}_{station}"
    output_base = join(temp_dir, base_name)
    
    # Save results - picks_2_CSV will add the suffix automatically
    if kurtosis_picks:
        picks_2_CSV(kurtosis_picks, output_base + ".csv", "kurtosis")
    
    if retained_picks:
        picks_2_CSV(retained_picks, output_base + ".csv", "retained")

def process_single_station_mp(args):
    """
    Process a single station using multiprocessing - optimized version.
    
    Each process only loads and processes data for ONE station, dramatically
    reducing I/O and memory usage compared to loading all stations.
    
    Parameters:
    -----------
    args : tuple
        (station, julian_day, yearpath, pickspath, ktsispath, year, prev_julian, next_julian, basedir, enable_plotting, interactive_mode)
        
    Returns:
    --------
    tuple : (station, num_retained, num_kurtosis, success, message)
    """
    station, julian_day, yearpath, pickspath, ktsispath, year, prev_julian, next_julian, basedir, enable_plotting, interactive_mode = args
    
    print(f"DEBUG: Processing station {station} with plotting={enable_plotting}")
    
    try:
        # Load waveform data ONLY for this station
        st = load_station_day(station, julian_day, yearpath)
        
        if len(st) == 0:
            return (station, 0, 0, False, f"No waveform data found for station {station}")
        
        # Load adjacent day data ONLY for this station
        prev_st = load_station_day(station, prev_julian, yearpath) if prev_julian else Stream()
        next_st = load_station_day(station, next_julian, yearpath) if next_julian else Stream()
        
        # Load picks ONLY for this station
        station_picks = load_picks_for_station(station, julian_day, pickspath, year)
        
        if not station_picks:
            return (station, 0, 0, False, f"No picks found for station {station}")
        
        # Process this station using the existing optimized function
        retained_picks, kurtosis_picks = process_station_optimized(
            station, station_picks, st, prev_st, next_st, year, julian_day, ktsispath, basedir, enable_plotting, interactive_mode)
        
        # Save results for this station
        save_station_results(station, julian_day, retained_picks, kurtosis_picks, 
                           ktsispath, year)
        
        # Clean up memory
        st.clear()
        prev_st.clear()
        next_st.clear()
        del st, prev_st, next_st, station_picks
        gc.collect()
        
        return (station, len(retained_picks), len(kurtosis_picks), True, 
                f"Success: {len(retained_picks)} retained, {len(kurtosis_picks)} kurtosis")
        
    except Exception as e:
        return (station, 0, 0, False, f"Error processing station {station}: {e}")

def consolidate_station_results(julian_day, ktsispath, year):
    """
    Consolidate individual station results into day-level CSV files.
    
    In multiprocessing mode, each station creates separate CSV files.
    This function combines them into the standard daily format.
    
    Parameters:
    -----------
    julian_day : str
        Julian day
    ktsispath : str
        Output directory path
    year : str
        Year
    """
    
    # Set up directories
    year_dir = join(ktsispath, year)
    temp_dir = join(ktsispath, "temp", year)
    os.makedirs(year_dir, exist_ok=True)
    
    # Find all station-specific result files for this day in temp directory
    jday_padded = julian_day.zfill(3)
    kurtosis_pattern = join(temp_dir, f"picks_max_{year}_{jday_padded}_*_kurtosis.csv")
    retained_pattern = join(temp_dir, f"picks_max_{year}_{jday_padded}_*_retained.csv")
    
    kurtosis_files = glob(kurtosis_pattern)
    retained_files = glob(retained_pattern)
    
    # Consolidate kurtosis picks
    if kurtosis_files:
        all_kurtosis_picks = []
        for file_path in kurtosis_files:
            try:
                df = pd.read_csv(file_path)
                all_kurtosis_picks.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if all_kurtosis_picks:
            consolidated_kurtosis = pd.concat(all_kurtosis_picks, ignore_index=True)
            output_file = join(year_dir, f"picks_max_{year}_{jday_padded}_kurtosis.csv")
            consolidated_kurtosis.to_csv(output_file, index=False)
            print(f"Consolidated {len(consolidated_kurtosis)} kurtosis picks to {output_file}")
            
            # Remove individual station files
            for file_path in kurtosis_files:
                try:
                    os.remove(file_path)
                except:
                    pass
    
    # Consolidate retained picks
    if retained_files:
        all_retained_picks = []
        for file_path in retained_files:
            try:
                df = pd.read_csv(file_path)
                all_retained_picks.append(df)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if all_retained_picks:
            consolidated_retained = pd.concat(all_retained_picks, ignore_index=True)
            output_file = join(year_dir, f"picks_max_{year}_{jday_padded}_retained.csv")
            consolidated_retained.to_csv(output_file, index=False)
            print(f"Consolidated {len(consolidated_retained)} retained picks to {output_file}")
            
            # Remove individual station files
            for retained_file in retained_files:
                try:
                    os.remove(retained_file)
                except:
                    pass
    
    # Clean up temporary directory if empty
    try:
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except:
        pass

def process_day_multiprocessing(julian_day, yearpath, pickspath, ktsispath, year, basedir,
                               prev_julian=None, next_julian=None, max_workers=None, enable_plotting=False, interactive_mode=False):
    """
    Process a complete day using optimized multiprocessing by station.
    
    Each process handles one station completely, reading only the data it needs.
    This dramatically reduces I/O overhead and memory usage.
    
    Parameters:
    -----------
    julian_day : str
        Julian day to process
    yearpath : str
        Path to year directory
    pickspath : str
        Path to picks directory
    ktsispath : str
        Output directory
    year : str
        Year
    prev_julian : str, optional
        Previous julian day for boundary analysis
    next_julian : str, optional
        Next julian day for boundary analysis
    max_workers : int, optional
        Maximum number of worker processes
        
    Returns:
    --------
    dict : Summary of processing results
    """
    
    # Discover available stations for this day
    stations = discover_stations_for_day(yearpath, julian_day)
    
    if not stations:
        return {
            'total_stations': 0,
            'successful_stations': 0,
            'total_retained': 0,
            'total_kurtosis': 0,
            'errors': ['No stations found for this day']
        }
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(len(stations), mp.cpu_count())
    
    print(f"Processing {len(stations)} stations using {max_workers} parallel processes")
    
    # Prepare arguments for each station
    process_args = [
        (station, julian_day, yearpath, pickspath, ktsispath, year, prev_julian, next_julian, basedir, enable_plotting, interactive_mode)
        for station in stations
    ]
    
    # Process stations in parallel
    results = []
    errors = []
    
    with Pool(processes=max_workers) as pool:
        # Use tqdm to show progress of multiprocessing
        with tqdm(total=len(stations), 
                 desc=f"Processing Stations (Day {julian_day})", 
                 unit="station",
                 leave=True,
                 position=0,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}',
                 ncols=120) as pbar:
            
            # Process in batches to show progress
            for result in pool.imap(process_single_station_mp, process_args):
                station, num_retained, num_kurtosis, success, message = result
                
                if success:
                    results.append(result)
                    pbar.set_postfix_str(f"Station: {station}, Retained: {num_retained}, Kurtosis: {num_kurtosis}")
                else:
                    errors.append(f"Station {station}: {message}")
                    pbar.set_postfix_str(f"Station: {station}, Status: ERROR")
                
                pbar.update(1)
    
    # Calculate summary statistics
    successful_stations = len(results)
    total_retained = sum(r[1] for r in results)
    total_kurtosis = sum(r[2] for r in results)
    
    # Consolidate individual station files into day-level files
    if successful_stations > 0:
        try:
            consolidate_station_results(julian_day, ktsispath, year)
        except Exception as e:
            errors.append(f"Error consolidating results: {e}")
    
    return {
        'total_stations': len(stations),
        'successful_stations': successful_stations,
        'total_retained': total_retained,
        'total_kurtosis': total_kurtosis,
        'errors': errors,
        'station_results': results
    }

def eqcorrscan_picks_from_kurtosis(kurt_trace, threshold, phase, trig_int):
    """
    Extract discrete picks from kurtosis derivative trace using EQcorrscan triggering.
    
    This function applies peak detection algorithms to identify arrival times
    from the first derivative of the kurtosis characteristic function.
    
    Parameters:
    -----------
    kurt_trace : obspy.Trace
        Trace containing first derivative of kurtosis values
    threshold : float
        Minimum amplitude threshold for peak detection
    phase : str
        Phase type ('P' or 'S') for labeling
    trig_int : float
        Minimum interval between triggers in seconds
    
    Returns:
    --------
    tuple : (out_picks, start_picks, end_picks, peak_values)
        out_picks: list of UTCDateTime objects for peak times
        start_picks: list of UTCDateTime objects for pick start times
        end_picks: list of UTCDateTime objects for pick end times
        peak_values: list of float peak amplitudes
        
    Notes:
    ------
    Uses EQcorrscan's find_peaks_compiled for robust peak detection.
    Only considers positive peaks above the threshold.
    """
    print("    Using EQcorrscan findpeaks for triggering")
    out_picks = []
    start_picks = []
    end_picks = []
    peak_values = []
    trace = kurt_trace
    _trig_int = trig_int * trace.stats.sampling_rate
    
    # Find peaks using EQcorrscan compiled function
    triggers = find_peaks_compiled(arr=trace.data, thresh=threshold, trig_int=_trig_int)
    triggers = [t for t in triggers if t[0] > 0]  # Only positive peaks
    times = trace.times()
    
    for peak_value, s_peak in triggers:
        # Find start of peak (backwards)
        s0 = s_peak
        while s0 > 0:
            if trace.data[s0] < threshold:
                break
            s0 -= 1
        
        # Find end of peak (forwards)
        s1 = s_peak
        while s1 < len(trace.data):
            if trace.data[s1] < threshold:
                break
            s1 += 1

        # Convert sample indices to absolute times
        t0 = trace.stats.starttime + times[s0]
        try:
            t1 = trace.stats.starttime + times[s1]
        except:
            t1 = trace.stats.starttime + times[s1-1]

        t_peak = trace.stats.starttime + times[s_peak]

        out_picks.append(t_peak)
        start_picks.append(t0)
        end_picks.append(t1)
        peak_values.append(peak_value)

    return out_picks, start_picks, end_picks, peak_values

def csv_2_ObsPy(csv_file_path):
    """
    Convert EQT picks from CSV format to ObsPy Pick objects.
    OPTIMIZED VERSION with faster pandas reading and vectorized operations.
    """
    picks = []
    
    # Optimized CSV reading with specific dtypes for speed
    try:
        df = pd.read_csv(csv_file_path, 
                        dtype={'trace_id': 'string', 'phase': 'string'},
                        parse_dates=['start_time', 'end_time', 'peak_time'])
    except:
        # Fallback to normal reading if dtype optimization fails
        df = pd.read_csv(csv_file_path)
    
    print(f"Loaded {len(df)} picks from {csv_file_path}")
    
    # Vectorized trace_id parsing for better performance
    trace_parts = df['trace_id'].str.split('.', expand=True)
    df['network'] = trace_parts[0]
    df['station'] = trace_parts[1] 
    df['location'] = trace_parts[2].fillna('')
    
    for _, row in df.iterrows():
        # Use pre-parsed columns for better performance
        network = row['network']
        station = row['station'] 
        location = row['location']
        
        if pd.isna(network) or pd.isna(station):
            print(f"Warning: Invalid trace_id format: {row['trace_id']}")
            continue
        
        # Create ObsPy Pick object with full metadata
        pick = Pick(
            resource_id=ResourceIdentifier(prefix="pick"),
            waveform_id=WaveformStreamID(
                network_code=network,
                station_code=station,
                location_code=location,
                channel_code="HHZ"  # Assuming vertical channel
            ),
            phase_hint=row['phase'],
            time=UTCDateTime(row['peak_time']),
            time_errors=QuantityError(
                confidence_level=row['peak_value'],
                lower_uncertainty=UTCDateTime(row['peak_time']) - UTCDateTime(row['start_time']),
                upper_uncertainty=UTCDateTime(row['end_time']) - UTCDateTime(row['peak_time']),
                uncertainty=(UTCDateTime(row['end_time']) - UTCDateTime(row['start_time'])) / 2
            ),
            evaluation_mode="automatic",
            evaluation_status="preliminary",
            creation_info=CreationInfo(
                agency_id="EQT",
                author="EQTransformer",
                creation_time=UTCDateTime.now()
            ),
            method_id=ResourceIdentifier("EQTransformer"),
            filter_id=ResourceIdentifier("Bandpass"),
            comments=[Comment(
                text=f"EQT pick with confidence {row['peak_value']}",
                resource_id=ResourceIdentifier(prefix="comment")
            )]
        )
        
        picks.append(pick)
    
    return picks

def kurt_picks_2_ObsPy(short_tr, kurt_picks, start_picks, end_picks, threshold, peak_values):
    """
    Convert kurtosis-derived picks to ObsPy Pick objects.
    
    Transforms raw kurtosis detection results into standardized ObsPy Pick
    objects with proper metadata and uncertainty information.
    
    Parameters:
    -----------
    short_tr : obspy.Trace
        Source trace used for kurtosis analysis (for metadata)
    kurt_picks : list
        List of UTCDateTime objects for kurtosis pick times
    start_picks : list
        List of UTCDateTime objects for pick start times
    end_picks : list  
        List of UTCDateTime objects for pick end times
    threshold : float
        Threshold value used in kurtosis detection
    peak_values : list
        List of peak amplitude values
        
    Returns:
    --------
    list : List of obspy.core.event.Pick objects
    
    Notes:
    ------
    - Inherits station/network metadata from input trace
    - Assumes P-phase picks (can be modified for S-phase)
    - Calculates uncertainties from pick time windows
    - Adds kurtosis-specific metadata and comments
    """
    picks_out = []
    
    for i in range(len(kurt_picks)):
        pick_t = kurt_picks[i]
        start = start_picks[i]
        end = end_picks[i]
        peak = peak_values[i]
        
        # Create descriptive comment about the picking method
        comment = (f"Picked by kurtosis picker using HOST method. "
                  f"Threshold: {threshold:.3f}, Peak value: {peak:.3f}")
        
        creation_info = CreationInfo(
            agency_id="Kurtosis Picker",
            author="Automated HOST-picker",
            creation_time=UTCDateTime.now()
        )
        
        # Create ObsPy Pick with comprehensive metadata
        pick = Pick(
            resource_id=ResourceIdentifier(prefix="pick"),
            time=pick_t,
            waveform_id=WaveformStreamID(
                station_code=short_tr.stats.station,
                channel_code=short_tr.stats.channel,
                network_code=short_tr.stats.network,
                location_code=short_tr.stats.location
            ),
            filter_id=ResourceIdentifier("Bandpass 2-45Hz"),
            phase_hint="P",  # Kurtosis analysis typically used for P-waves
            evaluation_mode="automatic",
            method_id=ResourceIdentifier("Kurtosis HOST-picker"),
            evaluation_status="preliminary",
            creation_info=creation_info,
            comments=[Comment(
                text=comment,
                resource_id=ResourceIdentifier(prefix="comment")
            )],
            time_errors=QuantityError(
                confidence_level=peak,
                lower_uncertainty=pick_t - start,
                upper_uncertainty=end - pick_t,
                uncertainty=(end - start) / 2
            )
        )
        
        picks_out.append(pick)
    
    return picks_out

def picks_2_CSV(picks, output_path, suffix):
    """
    Export ObsPy Pick objects to CSV format.
    
    Writes standardized pick information to CSV files for easy analysis
    and integration with other seismic processing workflows.
    
    Parameters:
    -----------
    picks : list
        List of obspy.core.event.Pick objects
    output_path : str
        Base path for output file (will be modified with suffix)
    suffix : str
        Suffix to add to filename (e.g., 'kurtosis', 'retained')
        
    Output CSV format:
    ------------------
    Comprehensive format with all pick metadata including:
    - Resource_id, Network, Station, Channel, Location
    - Phase, Time, Confidence, Uncertainty information
    - Evaluation details and method information
    - Creation info and comments
    
    Notes:
    ------
    Creates files with pattern: <base_name>_<suffix>.csv
    Overwrites existing files with same name
    """
    csv_filename = output_path.replace('.csv', f'_{suffix}.csv')
    
    with open(csv_filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        
        # Comprehensive header with all pick metadata
        header = ["Resource_id", "Network", "Station", "Channel", "Location",
                 "Phase", "Time", "Confidence", "Uncertainty",
                 "Lower_Uncertainty", "Upper_Uncertainty", "Evaluation_mode",
                 "Evaluation_status", "Agency_id", "Author", "creation_time",
                 "Method_id", "Filter_id", "Comment", "Comment_id"]
        
        writer.writerow(header)
        
        # Write data rows with full pick information
        for pick in picks:
            row = [
                pick.resource_id,
                pick.waveform_id.network_code,
                pick.waveform_id.station_code,
                pick.waveform_id.channel_code,
                pick.waveform_id.location_code,
                pick.phase_hint,
                pick.time,
                pick.time_errors.confidence_level,
                pick.time_errors.uncertainty,
                pick.time_errors.lower_uncertainty,
                pick.time_errors.upper_uncertainty,
                pick.evaluation_mode,
                pick.evaluation_status,
                pick.creation_info.agency_id,
                pick.creation_info.author,
                pick.creation_info.creation_time,
                pick.method_id,
                pick.filter_id,
                pick.comments[0].text,
                pick.comments[0].resource_id
            ]
            writer.writerow(row)
    
    print(f"Saved {len(picks)} picks to {csv_filename}")

def plot_kurtosis_analysis_original(tr_filtered, kurtosis_trace, der_1_trace, short_tr, 
                          station_picks, kurt_picks, retained_pick, station, 
                          save_path=None, show_plot=True, interactive=False):
    """
    COMENTADA: Plot kurtosis analysis results following original style.
    
    Esta función ha sido comentada y reemplazada por plot_kurtosis_analysis_three_panel()
    que integra las probabilidades EQT en un formato de 3 paneles.
    
    Para usar la función original, descomenta el código a continuación.
    """
    print(f"Warning: Using original plotting function for station {station} - consider using three_panel version")
    return None

def plot_kurtosis_analysis_three_panel(tr_filtered, kurtosis_trace, der_1_trace, 
                                      station_picks, retained_pick, station, 
                                      year, julian_day, basedir, save_path=None, 
                                      show_plot=True, interactive=False):
    """
    Plot kurtosis analysis results with three panels:
    1. Filtered waveform with EQT picks
    2. EQT probability curves (P and S phases)
    3. Kurtosis function and first derivative with retained picks
    
    This function is specifically designed for ALMA station and reads EQT probability files.
    
    Parameters:
    -----------
    tr_filtered : obspy.Trace
        Filtered waveform trace
    kurtosis_trace : obspy.Trace
        Kurtosis function trace
    der_1_trace : obspy.Trace
        First derivative of kurtosis
    station_picks : list
        List of EQT picks for the station
    retained_pick : obspy.Pick
        The retained EQT pick after kurtosis analysis
    station : str
        Station code
    year : str
        Year for file paths
    julian_day : str
        Julian day for file paths
    basedir : str
        Base directory path for finding probability files
    save_path : str, optional
        Path to save figure (if None, not saved)
    show_plot : bool
        Whether to display the plot
    interactive : bool
        Whether to prompt user to continue
        
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    
    # Only process ALMA station as requested
    if station != 'ALMA':
        print(f"Skipping plotting for station {station} (only ALMA is processed)")
        return None
    
    print(f"DEBUG: Starting plotting for ALMA station with {len(station_picks)} picks")
    if retained_pick:
        print(f"DEBUG: Retained pick time: {retained_pick.time}")
    else:
        print("DEBUG: No retained pick found!")
        return None
    
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(16, 10))
    
    # Create time arrays for waveform
    x_waveform = np.arange(0, tr_filtered.stats.npts)
    x_waveform = x_waveform * tr_filtered.stats.delta
    x_waveform = [(tr_filtered.stats.starttime + _).datetime for _ in x_waveform]
    
    # Panel 1: Filtered waveform with EQT picks
    axes[0].plot(x_waveform, tr_filtered.data, 'k', linewidth=1, label='Filtered waveform')
    axes[0].set_ylabel('Amplitude\n(counts)', fontsize=20)
    # axes[0].set_title(f'Station {station} - Kurtosis Analysis', fontsize=14, fontweight='bold')
    
    # Add EQT picks as vertical lines
    for pick in station_picks:
        if pick.time >= tr_filtered.stats.starttime and pick.time <= tr_filtered.stats.endtime:
            pick_time = pick.time.datetime
            if pick.phase_hint == "P":
                axes[0].axvline(pick_time, color="blue", linewidth=1.5, alpha=0.8, 
                              label="EQT P pick" if pick == station_picks[0] else "")
            else:
                axes[0].axvline(pick_time, color="darkorange", linewidth=1.5, alpha=0.8,
                              label="EQT S pick" if pick.phase_hint == "S" else "")
    
    # # Add retained pick
    # if retained_pick:
    #     retained_time = retained_pick.time.datetime
    #     axes[0].axvline(retained_time, color="red", linewidth=2, alpha=0.9, 
    #                    label="Retained pick")
    
    axes[0].legend(loc='upper left',fontsize=20)
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: EQT Probability curves
    probspath = join(basedir, 'CATALOGS/EQT_PICKS/PROBS')
    
    try:
        # Read EQT probabilities for this station
        probabilities = read_eqt_probabilities(probspath, year, julian_day.zfill(3), 
                                             tr_filtered.stats.network, station)
        
        if probabilities:
            # Plot P probabilities
            if 'P' in probabilities:
                prob_p = probabilities['P']
                times_p = np.arange(0, prob_p.stats.npts) * prob_p.stats.delta
                times_p = [(prob_p.stats.starttime + _).datetime for _ in times_p]
                axes[1].plot(times_p, prob_p.data, color='blue', linewidth=1.5, 
                           label='P-phase probability', alpha=0.8)            
            # # Plot S probabilities
            # if 'S' in probabilities:
            #     prob_s = probabilities['S']
            #     times_s = np.arange(0, prob_s.stats.npts) * prob_s.stats.delta
            #     times_s = [(prob_s.stats.starttime + _).datetime for _ in times_s]
            #     axes[1].plot(times_s, prob_s.data, color='darkorange', linewidth=1.5, 
            #                label='S-phase probability', alpha=0.8)
            axes[1].set_ylabel('EQT Probability',fontsize=20)
            axes[1].legend(loc='upper right', fontsize=20)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
            
        else:
            # No probabilities found - show placeholder
            axes[1].text(0.5, 0.5, f'No EQT probability files found for {station}\nDay {julian_day}', 
                        transform=axes[1].transAxes, ha='center', va='center', fontsize=20,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1].set_ylabel('EQT Probability', fontsize=20)
            axes[1].set_ylim(0, 1)
            axes[1].grid(True, alpha=0.3)
            
    except Exception as e:
        print(f"Warning: Could not read EQT probabilities for {station}: {e}")
        axes[1].text(0.5, 0.5, f'Error reading EQT probabilities:\n{str(e)[:50]}...', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=20,
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        axes[1].set_ylabel('EQT Probability', fontsize=20)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Kurtosis function and first derivative
    # Kurtosis time array
    x_kurtosis = np.arange(0, kurtosis_trace.stats.npts)
    x_kurtosis = x_kurtosis * kurtosis_trace.stats.delta
    x_kurtosis = [(kurtosis_trace.stats.starttime + _).datetime for _ in x_kurtosis]
    
    # Derivative time array
    x_derivative = np.arange(0, der_1_trace.stats.npts)
    x_derivative = x_derivative * der_1_trace.stats.delta
    x_derivative = [(der_1_trace.stats.starttime + _).datetime for _ in x_derivative]
    
    # Plot kurtosis (left y-axis)
    axes[2].plot(x_kurtosis, kurtosis_trace.data, color='tab:blue', linewidth=2, 
                label='Kurtosis', alpha=0.8)
    axes[2].set_ylabel('Kurtosis', color='tab:blue', fontsize=20)
    axes[2].tick_params(axis='y', labelcolor='tab:blue')
    
    # Create second y-axis for derivative
    ax2_twin = axes[2].twinx()
    ax2_twin.plot(x_derivative, der_1_trace.data, color='tab:red', linewidth=1.5, 
                 label='1st Derivative', alpha=0.8)
    ax2_twin.set_ylabel('Kurtosis 1st Derivative', color='tab:red', fontsize=20)
    ax2_twin.tick_params(axis='y', labelcolor='tab:red', labelsize=16)
    
    # Add retained pick to kurtosis panel
    if retained_pick:
        retained_time = retained_pick.time.datetime
        axes[2].axvline(retained_time, color="red", linewidth=2, alpha=0.9, 
                       label="Retained pick", linestyle='--')
        ax2_twin.axvline(retained_time, color="red", linewidth=2, alpha=0.9, 
                        linestyle='--')
    
    # Add legends
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=20)
    
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (UTC)', fontsize=20)
    
    # Set common time limits for all panels - 5 seconds before and after retained pick
    if retained_pick:
        retained_time = retained_pick.time.datetime
        # Calculate 5 seconds before and after the retained pick
        time_start = retained_time - timedelta(seconds=5)
        time_end = retained_time + timedelta(seconds=5)
        
        # Make sure we don't go beyond the actual data limits
        data_start = tr_filtered.stats.starttime.datetime
        data_end = tr_filtered.stats.endtime.datetime
        
        time_start = max(time_start, data_start)
        time_end = min(time_end, data_end)
        
        print(f"Setting time window to 5 seconds around retained pick: {retained_time.strftime('%H:%M:%S')}")
        print(f"Window: {time_start.strftime('%H:%M:%S')} - {time_end.strftime('%H:%M:%S')}")
    else:
        # Fallback to original data limits if no retained pick
        time_start = tr_filtered.stats.starttime.datetime
        time_end = tr_filtered.stats.endtime.datetime
        print(f"No retained pick found, using full data window")
    
    for ax in axes:
        ax.set_xlim([time_start, time_end])
    
    # Format time axis with finer resolution for the shorter window
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axes[2].xaxis.set_major_locator(mdates.SecondLocator(interval=1))  # 1-second intervals for 10-second window
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, fontsize=16)
    
    # Adjust y-axis tick label size
    for i in range(len(axes)):
        axes[i].tick_params(axis='y', labelsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall title with date info and time window information
    if retained_pick:
        window_info = f"(±5s around retained pick at {retained_pick.time.strftime('%H:%M:%S')})"
    else:
        window_info = "(full window)"
    
    fig.suptitle(f'Kurtosis Analysis - Station {station} {window_info}\n'
                f'Date: {time_start.strftime("%Y-%m-%d")} - Julian Day {julian_day}', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.92)
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            
            plt.savefig(save_path, dpi=400, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Figure saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Could not save figure to {save_path}: {e}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    # Interactive prompt
    if interactive:
        inp = input("Continue? (press Enter or 'y' to continue): ")
        if inp and inp.lower() != 'y':
            plt.close(fig)
            return None
    
    return fig

# Alias for backward compatibility and ease of use
def plot_kurtosis_analysis(tr_filtered, kurtosis_trace, der_1_trace, short_tr=None,
                          station_picks=None, kurt_picks=None, retained_pick=None, 
                          station=None, save_path=None, show_plot=True, interactive=False,
                          year=None, julian_day=None, basedir=None):
    """
    Wrapper function that calls the three-panel plotting function.
    Maintains compatibility with existing code while using new functionality.
    """
    if station and year and julian_day and basedir:
        print(f"DEBUG: Calling three-panel plotting for station {station}")
        return plot_kurtosis_analysis_three_panel(
            tr_filtered=tr_filtered,
            kurtosis_trace=kurtosis_trace,
            der_1_trace=der_1_trace,
            station_picks=station_picks or [],
            retained_pick=retained_pick,
            station=station,
            year=year,
            julian_day=julian_day,
            basedir=basedir,
            save_path=save_path,
            show_plot=show_plot,
            interactive=interactive
        )
    else:
        print(f"Warning: Missing parameters for three-panel plotting. Station: {station}")
        return None

def process_station_optimized(station, station_picks, station_st, prev_st_station, next_st_station, year, julian_day=None, ktsispath=None, basedir=None, enable_plotting=False, interactive_mode=False):
    """
    Optimized single station processing for kurtosis analysis.
    
    This function processes one station's picks using the original algorithm
    but with pre-processed waveform data for better performance.
    
    Parameters:
    -----------
    station : str
        Station code
    station_picks : list
        EQT picks for this station
    station_st : obspy.Stream
        Pre-processed waveform stream for this station
    prev_st_station : obspy.Stream
        Pre-processed previous day stream for this station
    next_st_station : obspy.Stream
        Pre-processed next day stream for this station
    year : str
        Year filter
    julian_day : str, optional
        Julian day (zero-padded) used for saving per-station kurtosis traces
    ktsispath : str, optional
        Path to kurtosis output directory (used for temporary per-station mseed files)
        
    Returns:
    --------
    tuple : (retained_picks, kurtosis_picks)
        - retained_picks: list, refined EQT picks
        - kurtosis_picks: list, pure kurtosis picks
    """
    
    retained_picks = []  # Refined EQT picks for this station
    kurtosis_picks = []  # Pure kurtosis picks for this station
    
    if not station_picks:
        return retained_picks, kurtosis_picks
    
    # Quick check if we have waveform data for this station
    if not station_st or len(station_st) == 0:
        # No waveform data - retain all picks without modification
        retained_picks.extend(station_picks)
        return retained_picks, kurtosis_picks
    
    # ====================================================================
    # STEP 1: ENHANCED GROUPING ALGORITHM
    # ====================================================================
    handled_picks = []
    
    for i, pick in enumerate(station_picks): #<-------------------
        if pick in handled_picks:
            continue
        
        pick_time = pick.time
        pick_phase = pick.phase_hint
        handled_picks.append(pick)
        
        # Start with the initial pick
        neighboring_picks = [pick]
        
        # Enhanced grouping: cascading algorithm
        for j in range(i, len(station_picks)): #<-------------------
            if station_picks[j] not in handled_picks:
                # Primary condition: within 4 seconds of original pick, same phase
                if (abs(station_picks[j].time - pick_time) <= 4 and 
                    station_picks[j].phase_hint == pick_phase):
                    neighboring_picks.append(station_picks[j])
                    handled_picks.append(station_picks[j])
                
                # Secondary condition: within 1 second of last pick in group, same phase
                elif (len(neighboring_picks) > 1 and
                      abs(station_picks[j].time - neighboring_picks[-1].time) <= 1 and 
                      station_picks[j].phase_hint == pick_phase):
                    neighboring_picks.append(station_picks[j])
                    handled_picks.append(station_picks[j])

                print(neighboring_picks)
        
        # ================================================================
        # STEP 2: PROCESS PICK GROUPS
        # ================================================================
        
        # Single pick: retain without analysis
        if len(neighboring_picks) == 1:
            retained_picks.append(pick)
        
        # Multiple picks: apply kurtosis analysis
        elif len(neighboring_picks) > 1:
            try:
                # ========================================================
                # STEP 3: USE PRE-PROCESSED WAVEFORM DATA
                # ========================================================
                # Use pre-processed station stream (already filtered to vertical channels)
                sub_st = station_st.copy()
                
                # ====================================================
                # STEP 4: BOUNDARY HANDLING
                # ====================================================
                pick_times = [p.time for p in neighboring_picks]
                earliest_pick = min(pick_times)
                latest_pick = max(pick_times)
                
                # Check boundaries and add adjacent day data if needed
                if (earliest_pick < sub_st[0].stats.starttime + 60 and 
                    prev_st_station and len(prev_st_station) > 0):
                    for tr in prev_st_station:
                        sub_st.append(tr.copy())
                
                # Check if any pick falls beyond current day
                day_end_boundary = any(pt > sub_st[0].stats.endtime for pt in pick_times)
                
                if ((day_end_boundary or latest_pick > sub_st[0].stats.endtime - 60) and 
                    next_st_station and len(next_st_station) > 0):
                    for tr in next_st_station:
                        sub_st.append(tr.copy())
                
                # Merge traces efficiently
                sub_st.merge(fill_value=0)
                
                # ====================================================
                # STEP 5: KURTOSIS ANALYSIS (P-PHASE ONLY)
                # ====================================================
                if neighboring_picks[0].phase_hint == "P":
                    try:
                        # Extract and filter focused time window
                        start_time = min(pick_times) - 10
                        end_time = max(pick_times) + 50
                        
                        # Work with vertical channel
                        tr_filtered = sub_st[0].copy()
                        tr_filtered.filter("bandpass", freqmin=2, freqmax=45)
                        tr_filtered = tr_filtered.slice(start_time, end_time)
                        
                        # Apply HOST picker for kurtosis
                        HP = Host(trace=tr_filtered,
                                time_windows=1,
                                hos_method="kurtosis",
                                transform_cf={},
                                detection_method="min")
                        
                        HP.work()
                        kurtosis_data = HP.hos_arr["1"]
                        
                        # Create kurtosis trace
                        kurtosis_trace = Trace(data=kurtosis_data)
                        kurtosis_trace.stats.starttime = tr_filtered.stats.starttime + 1
                        kurtosis_trace.stats.delta = tr_filtered.stats.delta
                        kurtosis_trace.stats.station = tr_filtered.stats.station
                        kurtosis_trace.stats.network = tr_filtered.stats.network
                        kurtosis_trace.stats.location = tr_filtered.stats.location
                        kurtosis_trace.stats.channel = tr_filtered.stats.channel
                        
                        # ================================================
                        # STEP 6: ROBUST NaN HANDLING
                        # ================================================
                        # Replace NaN values efficiently using numpy
                        kurt_data_clean = np.nan_to_num(kurtosis_data, nan=0.0)
                        
                        # Apply simple correction for zero values
                        for k in range(1, len(kurt_data_clean)):
                            if kurt_data_clean[k] == 0:
                                kurt_data_clean[k] = kurt_data_clean[k-1]
                        
                        # ================================================
                        # STEP 7: CALCULATE FIRST DERIVATIVE
                        # ================================================
                        kurt_der_1_data = np.diff(kurt_data_clean)
                        der_1_trace = Trace(data=kurt_der_1_data)
                        der_1_trace.stats.starttime = kurtosis_trace.stats.starttime
                        der_1_trace.stats.delta = kurtosis_trace.stats.delta
                        der_1_trace.stats.station = tr_filtered.stats.station
                        der_1_trace.stats.network = tr_filtered.stats.network
                        der_1_trace.stats.location = tr_filtered.stats.location
                        der_1_trace.stats.channel = tr_filtered.stats.channel
                        
                        # ================================================
                        # STEP 8: DEFINE ANALYSIS WINDOW
                        # ================================================
                        pick_start = min(pick_times)
                        pick_end = max(pick_times)
                        
                        # Adaptive window sizing
                        if pick_end - pick_start <= 1:
                            window_start = pick_start - 0.5
                            window_end = pick_end + 0.5
                        else:
                            window_half = (pick_end - pick_start) / 2
                            window_start = pick_start - window_half
                            window_end = pick_end + window_half
                        
                        short_tr = der_1_trace.slice(window_start, window_end)
                        
                        # ================================================
                        # STEP 9: PEAK DETECTION
                        # ================================================
                        if len(short_tr.data) > 0:
                            # Efficient threshold calculation
                            der_max = np.max(short_tr.data)
                            threshold = 0.1 * der_max if der_max > 1 else 0.5 * der_max
                            
                            # Find peaks in kurtosis derivative
                            kurt_picks_times, picks_start, picks_end, peaks = eqcorrscan_picks_from_kurtosis(
                                short_tr, threshold, "P", 0.01)
                            
                            if kurt_picks_times:
                                # Convert kurtosis picks to ObsPy format
                                kurtosis_obspy_picks = kurt_picks_2_ObsPy(
                                    tr_filtered, kurt_picks_times, picks_start, picks_end, threshold, peaks)
                                kurtosis_picks.extend(kurtosis_obspy_picks)
                                
                                # Find EQT pick closest to first kurtosis pick
                                time_diffs = [abs(p.time - kurt_picks_times[0]) for p in neighboring_picks]
                                min_diff_idx = np.argmin(time_diffs)
                                retained_pick = neighboring_picks[min_diff_idx]
                                retained_picks.append(retained_pick)
                                
                                # ================================================
                                # OPTIONAL PLOTTING (Three panel figure with EQT probabilities)
                                # ================================================
                                if enable_plotting:
                                    try:
                                        print(f"DEBUG: Plotting enabled for station {station}")
                                        # Create save path using new FIGURES directory structure
                                        save_path = None
                                        if basedir:
                                            # New path: /Volumes/GeoPhysics_49/users-data/montalca/FIGURES/KURTOSIS/YEAR
                                            plot_dir = join(basedir, "FIGURES", "KURTOSIS", year)
                                            os.makedirs(plot_dir, exist_ok=True)
                                            save_path = join(plot_dir, 
                                                           f"kurtosis_{station}_{julian_day}_{retained_pick.time.strftime('%H%M%S')}.png")
                                            print(f"DEBUG: Save path created: {save_path}")
                                        
                                        # Create the three-panel plot with EQT probabilities
                                        fig = plot_kurtosis_analysis(
                                            tr_filtered=tr_filtered,
                                            kurtosis_trace=kurtosis_trace, 
                                            der_1_trace=der_1_trace,
                                            station_picks=neighboring_picks,
                                            retained_pick=retained_pick,
                                            station=station,
                                            year=year,
                                            julian_day=julian_day,
                                            basedir=basedir,
                                            save_path=save_path,
                                            show_plot=interactive_mode,  # Show plot only in interactive mode
                                            interactive=interactive_mode  # Interactive prompt only in interactive mode
                                        )
                                        
                                        # Close figure to save memory
                                        if fig:
                                            plt.close(fig)
                                            
                                    except Exception as plot_error:
                                        print(f"Warning: Plotting failed for station {station}: {plot_error}")
                                
                            else:
                                # No kurtosis picks found - use highest confidence pick
                                best_pick = max(neighboring_picks, 
                                              key=lambda p: p.time_errors.confidence_level)
                                retained_picks.append(best_pick)
                        else:
                            # Empty window - use highest confidence pick
                            best_pick = max(neighboring_picks, 
                                          key=lambda p: p.time_errors.confidence_level)
                            retained_picks.append(best_pick)
                    
                    except Exception as kurtosis_error:
                        # Kurtosis analysis failed - use highest confidence pick
                        best_pick = max(neighboring_picks, 
                                      key=lambda p: p.time_errors.confidence_level)
                        retained_picks.append(best_pick)
                else:
                    # Non-P phase - retain highest confidence pick
                    best_pick = max(neighboring_picks, 
                                  key=lambda p: p.time_errors.confidence_level)
                    retained_picks.append(best_pick)
                
            except Exception as e:
                # General error - use highest confidence pick
                best_pick = max(neighboring_picks, 
                              key=lambda p: p.time_errors.confidence_level)
                retained_picks.append(best_pick)
    
    return retained_picks, kurtosis_picks

# -----------------------------------------------------------------------
# MAIN PROCESSING FUNCTION FOR SINGLE DAY
# -----------------------------------------------------------------------

def process_single_day(basedir, datadir, year, julian_day, enable_plotting=False, interactive_mode=False, max_workers=None):
    """
    Process a single day using multiprocessing optimized for shell script calling.
    
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
        
    Returns:
    --------
    dict : Processing results summary
    """
    
    print(f"{colours.CYAN}=== KURTOSIS SINGLE DAY PROCESSING ==={colours.ENDC}")
    print(f"Processing day {julian_day} for year {year}")
    print(f"Base directory: {basedir}")
    print(f"Data directory: {datadir}")
    
    # Setup paths
    datadir_full = join(basedir, datadir)
    yearpath = join(datadir_full, year)
    pickspath = join(basedir, 'CATALOGS/EQT_PICKS/PICKS')
    ktsispath = join(basedir, 'CATALOGS/EQT_PICKS/KURTOSIS')
    
    # Validate paths
    if not os.path.exists(datadir_full):
        print(f'{colours.RED}ERROR: Data directory does not exist: {datadir_full}{colours.ENDC}')
        return {'success': False, 'error': f'Data directory not found: {datadir_full}'}
    
    if not os.path.exists(yearpath):
        print(f'{colours.RED}ERROR: Year directory does not exist: {yearpath}{colours.ENDC}')
        return {'success': False, 'error': f'Year directory not found: {yearpath}'}
    
    # Create output directory
    os.makedirs(ktsispath, exist_ok=True)
    
    # Check if picks file exists
    picks_file = join(pickspath, year, f'picks_max_{year}_{julian_day}.csv')
    if not exists(picks_file):
        print(f'{colours.YELLOW}Warning: No picks file found for day {julian_day}: {picks_file}{colours.ENDC}')
        return {'success': False, 'error': f'No picks file found: {picks_file}'}
    
    print(f"Found picks file: {picks_file}")
    
    # Determine adjacent days for boundary analysis
    try:
        julian_int = int(julian_day)
        prev_julian = str(julian_int - 1).zfill(3) if julian_int > 1 else None
        next_julian = str(julian_int + 1).zfill(3) if julian_int < 365 else None
    except:
        prev_julian = None
        next_julian = None
    
    # Use multiprocessing approach from the original script
    if max_workers is None:
        # If interactive plotting is enabled, use single process for interactivity
        if interactive_mode:
            max_workers = 1
        else:
            max_workers = min(mp.cpu_count(), 2)  # Limit workers for single day
    
    print(f"Using multiprocessing with {max_workers} worker processes")
    if enable_plotting:
        if interactive_mode:
            print(f"{colours.YELLOW}Note: Interactive plotting enabled - using single process{colours.ENDC}")
            print(f"{colours.YELLOW}Interactive mode: You'll see plots and need to press Enter to continue{colours.ENDC}")
        else:
            print(f"{colours.YELLOW}Note: Plotting enabled - figures will be saved automatically{colours.ENDC}")
    
    try:
        # Process the day using multiprocessing
        results = process_day_multiprocessing(
            julian_day=julian_day,
            yearpath=yearpath,
            pickspath=pickspath,
            ktsispath=ktsispath,
            year=year,
            basedir=basedir,
            prev_julian=prev_julian,
            next_julian=next_julian,
            max_workers=max_workers,
            enable_plotting=enable_plotting,
            interactive_mode=interactive_mode
        )
        
        # Report results
        if results['successful_stations'] > 0:
            print(f"{colours.GREEN}SUCCESS: {results['successful_stations']}/{results['total_stations']} stations processed{colours.ENDC}")
            print(f"Retained picks: {results['total_retained']}, Kurtosis picks: {results['total_kurtosis']}")
            
            # Show output files
            year_dir = join(ktsispath, year)
            jday_padded = julian_day.zfill(3)
            kurtosis_file = join(year_dir, f"picks_max_{year}_{jday_padded}_kurtosis.csv")
            retained_file = join(year_dir, f"picks_max_{year}_{jday_padded}_retained.csv")
            
            if os.path.exists(kurtosis_file):
                print(f"Kurtosis picks saved to: {kurtosis_file}")
            if os.path.exists(retained_file):
                print(f"Retained picks saved to: {retained_file}")
            
            return {
                'success': True,
                'total_stations': results['total_stations'],
                'successful_stations': results['successful_stations'],
                'total_retained': results['total_retained'],  
                'total_kurtosis': results['total_kurtosis'],
                'errors': results['errors']
            }
        else:
            print(f"{colours.YELLOW}WARNING: No stations processed successfully{colours.ENDC}")
            if results['errors']:
                for error in results['errors']:
                    print(f"  {colours.RED}Error: {error}{colours.ENDC}")
            
            return {
                'success': False,
                'error': 'No stations processed successfully',
                'errors': results['errors']
            }
            
    except Exception as e:
        print(f"{colours.RED}ERROR in processing day {julian_day}: {e}{colours.ENDC}")
        return {'success': False, 'error': str(e)}

# -----------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------

def main():
    """
    Main function to handle command line arguments and process single day.
    """
    parser = argparse.ArgumentParser(
        description="Kurtosis Picker - Single Day Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process without plotting:
  python kurtosis_single_day.py /path/to/data DATA 2023 001
  
  # Save plots automatically (non-interactive):
  python kurtosis_single_day.py /path/to/data DATA 2023 001 --plot
  
  # Interactive mode (shows plots, waits for user input):
  python kurtosis_single_day.py /path/to/data DATA 2023 001 --interactive
  
  # With custom worker count:
  python kurtosis_single_day.py /path/to/data DATA 2023 001 --plot --max-workers 4
        """
    )
    
    parser.add_argument('basedir', help='Base directory path')
    parser.add_argument('datadir', help='Data directory name')
    parser.add_argument('year', help='Year to process')
    parser.add_argument('julian_day', help='Julian day to process')
    parser.add_argument('--plot', action='store_true', 
                       help='Enable plotting and save figures (non-interactive)')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive plotting mode (shows plots and waits for user input)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of worker processes (default: auto, 1 if interactive)')
    
    args = parser.parse_args()
    
    # Extract arguments
    basedir = args.basedir
    datadir = args.datadir  
    year = args.year
    julian_day = args.julian_day
    enable_plotting = args.plot or args.interactive
    interactive_mode = args.interactive
    max_workers = args.max_workers
    
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
    
    print(f"{colours.GREEN}Starting kurtosis processing for single day{colours.ENDC}")
    print(f"Parameters: basedir={basedir}, datadir={datadir}, year={year}, julian_day={julian_day}")
    if enable_plotting:
        if interactive_mode:
            print(f"{colours.YELLOW}Interactive plotting mode enabled (Cedric's style){colours.ENDC}")
        else:
            print(f"{colours.YELLOW}Plotting enabled - figures will be saved{colours.ENDC}")
    if max_workers:
        print(f"Maximum workers: {max_workers}")
    
    start_time = datetime.now()
    
    # Process the single day
    results = process_single_day(basedir, datadir, year, julian_day, enable_plotting, interactive_mode, max_workers)
    
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    print(f"\n{colours.CYAN}=== PROCESSING SUMMARY ==={colours.ENDC}")
    print(f"Processing time: {processing_time}")
    
    if results['success']:
        print(f"{colours.GREEN}Processing completed successfully!{colours.ENDC}")
        print(f"Stations processed: {results['successful_stations']}/{results['total_stations']}")
        print(f"Total retained picks: {results['total_retained']}")
        print(f"Total kurtosis picks: {results['total_kurtosis']}")
        sys.exit(0)
    else:
        print(f"{colours.RED}Processing failed: {results['error']}{colours.ENDC}")
        sys.exit(1)

if __name__ == '__main__':
    main()
