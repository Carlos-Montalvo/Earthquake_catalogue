#!/usr/bin/env python3
"""
Simplified version of 6_PRE_CROSS_CORRELATIONS.py for processing a single day
Designed to be called from shell scripts for better memory management
"""

import sys
import pickle
import datetime
import os
import gc
import argparse
from obspy import *
from os.path import join, exists


# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

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


def process_single_day(jday, year, yearpath, nll_dir, inv, region,
                       station_groups, streams_output_dir):
    """
    Procesa un día específico
    """
    print(f'Processing day: {jday} (PID: {os.getpid()})')
    
    MORIA = station_groups['MORIA']
    DPRI = station_groups['DPRI'] 
    GEONET = station_groups['GEONET']
    
    cat = Catalog()
    event_count = 0
    a = 0
    
    # Statistics for picks processing
    total_picks_processed = 0
    total_picks_valid = 0
    
    # Statistics for gap filling and data issues
    total_gaps_filled = 0
    events_with_gaps = 0
    picks_lost_no_channel = 0
    picks_lost_no_data = 0
    picks_lost_no_time_window = 0
    s_waves_optimized = 0  # Count of S-waves that used horizontal component optimization

    # Output file
    streams_file = join(streams_output_dir, f'{year}_{jday}.p')
    stream_dict = {}
    
    # Read events
    try:
        cat = read_events(join(nll_dir, f'{year}/{year}_{jday}_nll.xml'))
    except Exception as e:
        print(f'No events found for day {jday}: {e}')
        return {'day': jday, 'events': 0, 'picks_processed': 0, 'picks_valid': 0, 'success': False}

    if len(cat) > 1:
        num_events = len(cat)
        print(f'Found {num_events} events for day {jday}')
        
        # Read waveforms
        st = Stream()
        
        # MORIA - Structure: DATA/YEAR/MORIA/STATION/
        cnt = 0
        for station in MORIA:
            try:
                st += read(join(yearpath, f'MORIA/{station}/*{year}.{jday}'))
                cnt += 1
            except Exception:
                cnt += 0
        print(f'{cnt}/{len(MORIA)} MORIA stations read successfully for day {jday}')
        
        # DPRI - Structure: DATA/YEAR/DPRI/STATION/
        cnt = 0
        for station in DPRI:
            try:
                st += read(join(yearpath, f'DPRI/{station}/*{year}.{jday}'))
                cnt += 1
            except Exception:
                cnt += 0
        print(f'{cnt}/{len(DPRI)} DPRI stations read successfully for day {jday}')
        
        # GEONET - Structure: DATA/YEAR/GEONET/STATION/
        cnt = 0
        for station in GEONET:
            try:
                st += read(join(yearpath, f'GEONET/{station}/*{year}.{jday}'))
                cnt += 1
            except Exception:
                cnt += 0
        print(f'{cnt}/{len(GEONET)} GEONET stations read successfully for day {jday}')
        
        for event in cat:
            if event.origins[0].latitude > region[2] and event.origins[0].latitude < region[3]:
                if event.origins[0].longitude > region[0] and event.origins[0].longitude < region[1]:
                    # Check for depth uncertainty - handle None values
                    depth_uncertainty = event.origins[0].depth_errors.uncertainty if event.origins[0].depth_errors else None
                    if depth_uncertainty is None or depth_uncertainty < 1000000:  # Uncertainty in meters
                        event_count += 1
                        eventid = str(event.resource_id)
                        print(f"Day {jday}: Reading data for event {event_count}/{num_events}")
                        st_copy = st.copy()

                        # Find event time
                        try:
                            origin_time = event.preferred_origin().time or event.origins[0].time
                        except AttributeError:
                            # If there isn't an origin time, use the first pick time
                            origin_time = min([pick.time for pick in event.picks])
                        
                        noskip = False
                        if eventid not in stream_dict.keys():
                            noskip = True
                        elif len(stream_dict[eventid]) == 0:
                            noskip = True
                            del(stream_dict[eventid])

                        year_temp = origin_time.year
                        julday = origin_time.julday

                        st2 = Stream()
                        valid_picks = 0
                        initial_picks = len(event.picks)
                        total_picks_processed += initial_picks

                        for pick in event.picks:
                            net = pick.waveform_id.network_code
                            sta = pick.waveform_id.station_code
                            # New waveform length
                            length = 3  # seconds
                            starttime = pick.time - length
                            endtime = pick.time + length
                            
                            # Select appropriate channels based on station type and phase
                            channel = None
                            
                            # Determine station type
                            if sta in DPRI:
                                # DPRI stations: EHZ, EH1, EH2
                                if pick.phase_hint == 'P':
                                    channel_codes = ['EHZ']
                                else:  # S wave
                                    channel_codes = ['EH1', 'EH2']
                            elif sta in MORIA:
                                # MORIA stations: keep original logic
                                if pick.phase_hint == 'P':
                                    channel_codes = ['*HZ']
                                else:  # S wave
                                    channel_codes = ['*HE', '*HN', '*H1', '*H2']
                            elif sta in GEONET:
                                # GeoNet stations (NZ network)
                                if pick.phase_hint == 'P':
                                    # Try broadband first, then short period
                                    channel_codes = ['HHZ', 'EHZ']
                                else:  # S wave
                                    # Try broadband first, then short period
                                    channel_codes = ['HHE', 'HHN', 'EHE', 'EHN']
                            else:
                                # Unknown station type - try common channels
                                if pick.phase_hint == 'P':
                                    channel_codes = ['HHZ', 'EHZ', '*HZ']
                                else:  # S wave
                                    channel_codes = ['HHE', 'HHN', 'EHE', 'EHN', '*HE', '*HN']
                            
                            # Try to find the channel in inventory
                            for ch_code in channel_codes:
                                try:
                                    if '*' in ch_code:
                                        # Use wildcard selection
                                        channel = inv.select(station=sta, channel=ch_code)[0][0].channels[0]
                                    else:
                                        # Use exact channel name
                                        channel = inv.select(station=sta, channel=ch_code)[0][0].channels[0]
                                    break
                                except:
                                    continue
                            
                            if channel is None:
                                # Fallback: try original pick channel
                                try:
                                    original_channel = pick.waveform_id.channel_code
                                    channel = inv.select(station=sta, channel=original_channel)[0][0].channels[0]
                                except:
                                    picks_lost_no_channel += 1
                                    continue
                            
                            # Update pick with found channel
                            loc = channel.location_code
                            pick.waveform_id.channel_code = channel.code
                            pick.waveform_id.location_code = channel.location_code
                            
                            if noskip:
                                # For S waves, try both horizontal components and pick the best one
                                # if pick.phase_hint == 'S':
                                #     best_trace = None
                                #     best_channel_code = None
                                #     best_score = 0
                                    
                                #     # Get possible horizontal channels based on station type
                                #     if sta in DPRI:
                                #         horizontal_channels = ['EH1', 'EH2', 'EHE', 'EHN']
                                #     elif sta in GEONET:
                                #         horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN']
                                #     elif sta in MORIA:
                                #         horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN', 'EH1', 'EH2']
                                #     else:
                                #         horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN', 'EH1', 'EH2']
                                    
                                #     # Try each horizontal channel and evaluate quality
                                #     for ch_code in horizontal_channels:
                                #         try:
                                #             test_trace = st_copy.select(station=sta, channel=ch_code)
                                #             if len(test_trace) > 0:
                                #                 s1 = pick.time - length
                                #                 e1 = pick.time + length
                                #                 windowed_trace = test_trace.trim(starttime=s1, endtime=e1)
                                #                 if len(windowed_trace) > 0:
                                #                     # Score based on data availability and continuity
                                #                     trace = windowed_trace[0]
                                #                     expected_samples = int((e1 - s1) * trace.stats.sampling_rate)
                                #                     actual_samples = len(trace.data)
                                #                     completeness = actual_samples / expected_samples if expected_samples > 0 else 0
                                                    
                                #                     # Additional score based on data quality (check for gaps)
                                #                     gaps = windowed_trace.get_gaps()
                                #                     gap_penalty = len(gaps) * 0.1
                                                    
                                #                     score = completeness - gap_penalty
                                                    
                                #                     if score > best_score:
                                #                         best_score = score
                                #                         best_trace = windowed_trace.copy()
                                #                         best_channel_code = ch_code
                                                    
                                #                     # Clear temporary traces
                                #                     windowed_trace.clear()
                                #                     del windowed_trace
                                #             test_trace.clear()
                                #             del test_trace
                                #         except Exception:
                                #             continue
                                    
                                #     # Use the best horizontal component if found
                                #     if best_trace is not None and best_score > 0.5:  # Minimum quality threshold
                                #         st2 += best_trace
                                #         pick.waveform_id.channel_code = best_channel_code
                                #         valid_picks += 1
                                #         s_waves_optimized += 1
                                #         print(f"    S-wave: Selected {best_channel_code} (score: {best_score:.2f}) for {sta}")
                                #         best_trace.clear()
                                #         del best_trace
                                #     else:
                                #         picks_lost_no_time_window += 1
                                #         print(f"    S-wave: No suitable horizontal component found for {sta}")
                                #         continue
                                
                                # For S waves, try both horizontal components and pick the best one
                                if pick.phase_hint == 'S':
                                    best_trace = None
                                    best_channel_code = None
                                    best_score = 0
                                    best_critical_completeness = 0

                                    # Get possible horizontal channels based on station type
                                    if sta in DPRI:
                                        horizontal_channels = ['EH1', 'EH2', 'EHE', 'EHN']
                                    elif sta in GEONET:
                                        horizontal_channels = ['HHE', 'HHN', 'EHE', 'EHN']
                                    elif sta in MORIA:
                                        horizontal_channels = ['HHE', 'HHN','HH1', 'HH2',]
                                    else:
                                        horizontal_channels = ['HHE', 'HHN','HH1', 'HH2', 'EHE', 'EHN', 'EH1', 'EH2']

                                    for ch_code in horizontal_channels:
                                        try:
                                            # Verificar que el canal esté activo en el momento del pick
                                            ch_inv = inv.select(station=sta, channel=ch_code,
                                                               starttime=pick.time - length,
                                                               endtime=pick.time + length)
                                            if len(ch_inv) == 0:
                                                continue
                                            
                                            test_trace = st_copy.select(station=sta, channel=ch_code)
                                            if len(test_trace) == 0:
                                                continue
                                            
                                            s1 = pick.time - length
                                            e1 = pick.time + length
                                            windowed_trace = test_trace.trim(starttime=s1, endtime=e1)
                                            if len(windowed_trace) == 0:
                                                continue
                                            
                                            trace = windowed_trace[0]
                                            sr = trace.stats.sampling_rate
                                            expected_samples = int((e1 - s1) * sr)
                                            actual_samples = len(trace.data)

                                            if expected_samples == 0:
                                                continue
                                            
                                            # --- Criterio primario: ventana crítica alrededor del pick ---
                                            # El pick está en el centro de la ventana (en t = length)
                                            critical_half = 0.5  # segundos antes y después del pick
                                            critical_start_idx = int((length - critical_half) * sr)
                                            critical_end_idx = int((length + critical_half) * sr)
                                            critical_start_idx = max(0, critical_start_idx)
                                            critical_end_idx = min(actual_samples, critical_end_idx)

                                            critical_samples = critical_end_idx - critical_start_idx
                                            expected_critical = int(2 * critical_half * sr)
                                            critical_completeness = critical_samples / expected_critical if expected_critical > 0 else 0

                                            # Verificar gaps en la ventana crítica
                                            critical_gaps = [g for g in windowed_trace.get_gaps()
                                                            if g[4] < pick.time + critical_half  # gap empieza antes del fin de ventana crítica
                                                            and g[5] > pick.time - critical_half]  # gap termina después del inicio
                                            critical_has_gaps = len(critical_gaps) > 0

                                            # Descartar si la ventana crítica está incompleta o tiene gaps
                                            if critical_completeness < 0.9 or critical_has_gaps:
                                                windowed_trace.clear()
                                                del windowed_trace
                                                continue
                                            
                                            # --- Criterio secundario: calidad de toda la traza ---
                                            completeness = actual_samples / expected_samples
                                            all_gaps = windowed_trace.get_gaps()
                                            gap_penalty = len(all_gaps) * 0.1
                                            secondary_score = completeness - gap_penalty

                                            # Seleccionar el mejor canal:
                                            # Primero por critical_completeness, luego por secondary_score como desempate
                                            is_better = False
                                            if critical_completeness > best_critical_completeness:
                                                is_better = True
                                            elif critical_completeness == best_critical_completeness and secondary_score > best_score:
                                                is_better = True

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
                                        
                                    # Usar la mejor horizontal si pasó el criterio crítico
                                    if best_trace is not None:
                                        st2 += best_trace
                                        pick.waveform_id.channel_code = best_channel_code
                                        valid_picks += 1
                                        s_waves_optimized += 1
                                        print(f"    S-wave: Selected {best_channel_code} "
                                              f"(critical={best_critical_completeness:.2f}, score={best_score:.2f}) for {sta}")
                                        best_trace.clear()
                                        del best_trace
                                    else:
                                        picks_lost_no_time_window += 1
                                        print(f"    S-wave: No suitable horizontal component found for {sta}")
                                        continue        
                                else:
                                    # P waves - use original logic
                                    st5 = st_copy.select(station=sta, channel=pick.waveform_id.channel_code)
                                    if len(st5) == 0:
                                        picks_lost_no_data += 1
                                        continue
                                    s1 = pick.time - length
                                    e1 = pick.time + length
                                    st6 = st5.trim(starttime=s1, endtime=e1)
                                    if len(st6) == 0:
                                        picks_lost_no_time_window += 1
                                        continue
                                    st2 += st6
                                    valid_picks += 1
                                    
                                    # Clear temporary streams to save memory
                                    st5.clear()
                                    st6.clear()
                                
                        st2.merge()
                        
                        # Check for gaps and fill with zeros using ObsPy functionality
                        # This helps ensure continuous waveforms for cross-correlation
                        if len(st2) > 0:
                            try:
                                # Get gap information before filling
                                gaps = st2.get_gaps()
                                num_gaps = len(gaps)
                                
                                if num_gaps > 0:
                                    # Fill gaps with zeros - ObsPy method
                                    st2.merge(method=1, fill_value=0)
                                    total_gaps_filled += num_gaps
                                    events_with_gaps += 1
                                    print(f"Event {eventid[-8:]}: Filled {num_gaps} gaps with zeros for {len(st2)} traces")
                                else:
                                    print(f"Event {eventid[-8:]}: No gaps detected in {len(st2)} traces")
                            except Exception as e:
                                print(f"Event {eventid[-8:]}: Warning - gap detection/filling failed: {e}")
                                # Continue with original merge if gap filling fails
                                pass
                        
                        total_picks_valid += valid_picks
                        
                        print(f"Event {eventid[-8:]}: {initial_picks} picks → {valid_picks} valid → {len(st2)} traces")

                        if noskip:
                            stream_dict[eventid] = st2

                        # Clear temporary variables
                        st_copy.clear()
                        del st_copy

        # Clear large objects from memory
        cat.clear()
        st.clear()
        
        # Print statistics
        pick_success_rate = (total_picks_valid / total_picks_processed * 100) if total_picks_processed > 0 else 0
        picks_lost_total = picks_lost_no_channel + picks_lost_no_data + picks_lost_no_time_window
        
        # print(f'Day {jday} PROCESSING STATISTICS:')
        # print(f'  Total picks processed: {total_picks_processed}')
        # print(f'  Valid picks: {total_picks_valid}')
        # print(f'  Pick success rate: {pick_success_rate:.1f}%')
        # print(f'  Events with data: {len(stream_dict)}')
        # print()
        # print(f'GAP FILLING STATISTICS:')
        # print(f'  Total gaps filled: {total_gaps_filled}')
        # print(f'  Events with gaps: {events_with_gaps}')
        # print(f'  Gap filling: Applied to all traces (zeros for missing data)')
        # print()
        # print(f'PICKS LOST BREAKDOWN:')
        # print(f'  Lost due to missing channel: {picks_lost_no_channel}')
        # print(f'  Lost due to no station data: {picks_lost_no_data}')
        # print(f'  Lost due to no time window: {picks_lost_no_time_window}')
        # print(f'  Total picks lost: {picks_lost_total}')
        # print(f'  Dictionary size: {sys.getsizeof(stream_dict)} bytes')

        if len(stream_dict) > 1:
            with open(streams_file, 'wb') as fp:
                pickle.dump(stream_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Day {jday}: Written {len(stream_dict)} events to pickle file')
        
        # Store events count before clearing
        events_processed = len(stream_dict)
        
        # Clear dictionary and force garbage collection
        stream_dict.clear()
        del stream_dict
        gc.collect()
            
        return {'day': jday, 'events': events_processed, 
                'picks_processed': total_picks_processed, 'picks_valid': total_picks_valid, 
                'gaps_filled': total_gaps_filled, 'events_with_gaps': events_with_gaps,
                'picks_lost_no_channel': picks_lost_no_channel, 'picks_lost_no_data': picks_lost_no_data,
                'picks_lost_no_time_window': picks_lost_no_time_window, 's_waves_optimized': s_waves_optimized,
                'success': True}
    
    else:
        print(f'No data for day {jday}')
        return {'day': jday, 'events': 0, 'picks_processed': 0, 'picks_valid': 0, 
                'gaps_filled': 0, 'events_with_gaps': 0, 'picks_lost_no_channel': 0, 
                'picks_lost_no_data': 0, 'picks_lost_no_time_window': 0, 's_waves_optimized': 0,
                'success': False}


def main():
    """
    Main function to process a single day
    """
    parser = argparse.ArgumentParser(description='Process seismic data for a single day')
    parser.add_argument('--year', type=str, required=True, help='Year (e.g., 2022)')
    parser.add_argument('--jday', type=str, required=True, help='Julian day (e.g., 001)')
    parser.add_argument('--basedir', type=str, required=True, help='Base directory path')
    parser.add_argument('--datadir', type=str, required=True, help='Data directory path')
    parser.add_argument('--region', type=float, nargs=4, 
                        default=[171.2,176.1,-43.8,-39.3],
                        help='Region bounds: lon_min lon_max lat_min lat_max')
    
    args = parser.parse_args()
    
    codestart = datetime.datetime.now()
    
    ### DIRECTORIES AND FILES ###
    basedir = args.basedir
    # Catalogues
    ctlgdir = join(basedir, 'CATALOGS')
    nll_dir = join(ctlgdir, 'NLL')
    growclust_dir = join(ctlgdir, 'GROWCLUST')
    # Stations
    sta_dir = join(basedir, 'STATIONS')
    inv = read_inventory(join(sta_dir, 'nll_region_all_stations.xml'))
    
    # Data directory
    yearpath = join(args.datadir, args.year)
    
    # Check if directories exist
    if not exists(yearpath):
        print(f'Error: Data directory does not exist: {yearpath}')
        sys.exit(1)
    
    if not exists(join(nll_dir, args.year)):
        print(f'Error: NLL catalog directory does not exist: {join(nll_dir, args.year)}')
        sys.exit(1)
    
    # Region
    region = args.region
    print(f'Processing day {args.jday} of year {args.year}')
    print(f'Data path: {yearpath}')
    print(f'Region: {region}')
    
    # Organize stations by network
    station_groups = organize_stations_by_network(inv)
    print(f'Station groups: MORIA={len(station_groups["MORIA"])}, DPRI={len(station_groups["DPRI"])}, GEONET={len(station_groups["GEONET"])}')
    
    # Create output directory
    streams_output_dir = join(growclust_dir, 'STREAMS')
    if not exists(streams_output_dir):
        os.makedirs(streams_output_dir)
        print(f'Created output directory: {streams_output_dir}')
    
    # Process the single day
    result = process_single_day(args.jday, args.year, yearpath, nll_dir, inv, region, station_groups, streams_output_dir)
    
    # Print final statistics
    codestop = datetime.datetime.now()
    print(f'\n--- DAY {args.jday} SUMMARY ---')
    print(f'Success: {result["success"]}')
    print(f'Events processed: {result["events"]}')
    print(f'Picks processed: {result["picks_processed"]}')
    print(f'Valid picks: {result["picks_valid"]}')
    if result["picks_processed"] > 0:
        success_rate = result["picks_valid"] / result["picks_processed"] * 100
        print(f'Pick success rate: {success_rate:.1f}%')
    
    # Gap filling summary
    print(f'\nGAP FILLING SUMMARY:')
    print(f'Total gaps filled: {result["gaps_filled"]}')
    print(f'Events with gaps: {result["events_with_gaps"]}')
    
    # S-wave optimization summary
    print(f'\nS-WAVE HORIZONTAL COMPONENT OPTIMIZATION:')
    print(f'S-waves with optimized components: {result["s_waves_optimized"]}')
    
    # Detailed breakdown of lost picks
    total_lost = result["picks_lost_no_channel"] + result["picks_lost_no_data"] + result["picks_lost_no_time_window"]
    print(f'\nPICKS LOST BREAKDOWN:')
    print(f'Missing channel in inventory: {result["picks_lost_no_channel"]}')
    print(f'No data for station/channel: {result["picks_lost_no_data"]}')
    print(f'No data in time window: {result["picks_lost_no_time_window"]}')
    print(f'Total picks lost: {total_lost}')
    
    print(f'\nProcessing time: {codestop - codestart}')
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == '__main__':
    main()
