import sys
import pickle
import datetime
import os
import gc
from multiprocessing import Pool, cpu_count
from obspy import *
from pandas import date_range
from os.path import join,exists


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

def process_single_day(jday, year, yearpath, nll_dir, inv, region, station_groups, streams_output_dir):
    """
    Procesa un día específico - función independiente para multiprocessing
    """
    print(f'Processing day: {jday} (PID: {os.getpid()})')
    
    MORIA = station_groups['MORIA']
    DPRI = station_groups['DPRI'] 
    GEONET = station_groups['GEONET']
    
    cat = Catalog()
    event_count = 0
    event_count1 = 0
    a = 0
    
    # Statistics for picks processing
    total_picks_processed = 0
    total_picks_valid = 0

    # Output file
    streams_file = join(streams_output_dir, f'{year}_{jday}.p')
    stream_dict = {}
    
    # Read events
    try:
        cat = read_events(join(nll_dir,f'{year}/{year}_{jday}_nll.xml'))
    except:
        print(f'No events found for day {jday}')
        return {'day': jday, 'events': 0, 'picks_processed': 0, 'picks_valid': 0, 'success': False}

    if len(cat) > 1:
        num_events = len(cat)
        # Read waveforms
        st = Stream()
        
        # MORIA
        cnt = 0
        for station in MORIA:
            try:
                st += read(join(yearpath,f'{station}/*{year}.{jday}'))
                cnt += 1
            except Exception as ex:
                cnt += 0
        # print(f'{str(cnt)}/{str(len(MORIA))} of MORIA stations read successfully for day {jday}')
        
        # DPRI
        cnt = 0
        for station in DPRI:
            try:
                st += read(join(yearpath,f'{station}/*{year}.{jday}'))
                cnt += 1
            except Exception as ex:
                cnt += 0
        # print(f'{str(cnt)}/{str(len(DPRI))} of DPRI stations read successfully for day {jday}')
        
        # GEONET
        cnt = 0
        for station in GEONET:
            try:
                st += read(join(yearpath,f'{station}/*{year}.{jday}'))
                cnt += 1
            except Exception as ex:
                cnt += 0
        # print(f'{str(cnt)}/{str(len(GEONET))} of GEONET stations read successfully for day {jday}')
        
        for event in cat:
            if event.origins[0].latitude > region[2] and event.origins[0].latitude < region[3]:
                if event.origins[0].longitude > region[0] and event.origins[0].longitude < region[1]:
                    # Check for depth uncertainty - handle None values
                    depth_uncertainty = event.origins[0].depth_errors.uncertainty if event.origins[0].depth_errors else None
                    if depth_uncertainty is None or depth_uncertainty < 1000000: # Uncertainty in meters
                        event_count += 1
                        event_count1 += 1
                        eventid = str(event.resource_id)
                        print(f"Day {jday}: Reading data for event {event_count}/{num_events}")
                        st_copy = st.copy()

                        # Find event time
                        try:
                            origin_time = event.preferred_origin().time or event.origins[0].time
                        except AttributeError:
                            # If there isn't an origin time, use the first pick time start of the stream
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
                        st5 = None
                        st6 = None
                        valid_picks = 0
                        initial_picks = len(event.picks)
                        total_picks_processed += initial_picks

                        for pick in event.picks:
                            net = pick.waveform_id.network_code
                            sta = pick.waveform_id.station_code
                            # New waveform length
                            length = 3 # seconds
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
                                    a += 1
                                    continue
                            
                            # Update pick with found channel
                            loc = channel.location_code
                            pick.waveform_id.channel_code = channel.code
                            pick.waveform_id.location_code = channel.location_code
                            
                            if noskip:
                                station = sta
                                station = sta
                                year_str = str(year_temp)
                                julian_day = str(julday)
                                st5=st_copy.select(station = sta,channel = pick.waveform_id.channel_code)
                                if len(st5) == 0:
                                    print(f"Warning: No data for station {sta}, channel {pick.waveform_id.channel_code}")
                                    continue
                                s1 = pick.time - length
                                e1 = pick.time + length
                                st6=st5.trim(starttime=s1,endtime=e1)
                                if len(st6) == 0:
                                    print(f"Warning: No data after trim for station {sta}")
                                    continue
                                st2+=st6
                                valid_picks += 1
                        st2.merge()
                        total_picks_valid += valid_picks
                        
                        print(f"Event {eventid[-8:]}: {initial_picks} picks → {valid_picks} valid → {len(st2)} traces")

                        if noskip:
                            stream_dict[eventid] = st2
                            if st5 is not None:
                                st5.clear()
                            if st6 is not None:
                                st6.clear()
                            # Clear temporary variables
                            del st5, st6

                    else:
                        event_count += 1
                else:
                    event_count += 1
            else:
                event_count += 1
        
        # Clear large objects from memory
        cat.clear()
        st.clear()
        if 'st_copy' in locals():
            st_copy.clear()
        
        # Print statistics
        pick_success_rate = (total_picks_valid / total_picks_processed * 100) if total_picks_processed > 0 else 0
        print(f'Day {jday} PICK STATISTICS:')
        print(f'  Total picks processed: {total_picks_processed}')
        print(f'  Valid picks: {total_picks_valid}')
        print(f'  Success rate: {pick_success_rate:.1f}%')
        print(f'  Events with data: {len(stream_dict)}')
        print(f'  Dictionary size: {sys.getsizeof(stream_dict)} bytes')

        if len(stream_dict) > 1:
            with open(streams_file,'wb') as fp:
                pickle.dump(stream_dict,fp,protocol=pickle.HIGHEST_PROTOCOL)
                print(f'Day {jday}: Written {len(stream_dict)} events to pickle file')
        
        # Final cleanup
        try:
            if 'st2' in locals():
                st2.clear()
        except:
            pass
        
        # Store events count before clearing
        events_processed = len(stream_dict)
        
        # Clear dictionary and force garbage collection
        stream_dict.clear()
        del stream_dict
        gc.collect()
            
        return {'day': jday, 'events': events_processed, 
                'picks_processed': total_picks_processed, 'picks_valid': total_picks_valid, 'success': True}
    
    else:
        print(f'No data for day {jday}')
        return {'day': jday, 'events': 0, 'picks_processed': 0, 'picks_valid': 0, 'success': False}


# ------------------------------------------------------------------------------
# Main script starts here
# ------------------------------------------------------------------------------

def main():
    """
    Main function to execute the pre-cross-correlations workflow
    """
    ### DIRECTORIES AND FILES ###
    basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
    # Catalogues
    ctlgdir = join(basedir,'CATALOGS')
    nll_dir = join(ctlgdir,'NLL')
    growclust_dir = join(ctlgdir,'GROWCLUST')
    # Stations
    sta_dir = join(basedir,'STATIONS')
    sta_csv = join(sta_dir,'')
    inv = read_inventory(join(sta_dir,'nll_region_all_stations.xml'))

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

    # Multiprocessing configuration
    available_cpus = cpu_count()
    print(f'\nAvailable CPU cores: {available_cpus}')
    while True:
        try:
            num_processes = int(input(f'How many processes to use? (1-{available_cpus}): '))
            if 1 <= num_processes <= available_cpus:
                break
            else:
                print(f'Please enter a number between 1 and {available_cpus}')
        except ValueError:
            print('Please enter a valid number')

    print(f'Using {num_processes} processes for parallel processing')

    # Region
    # NLL
    # region = [171, 176.6, -44.3, -39.3]
    # GrowClust3D region
    region = [172,175.6,-43,-40.5]

    codestart = datetime.datetime.now()

    # Organize stations by network
    station_groups = organize_stations_by_network(inv)

    # Create output directory
    streams_output_dir = join(growclust_dir, 'STREAMS')
    if not exists(streams_output_dir):
        os.makedirs(streams_output_dir)
        print(f'Created output directory: {streams_output_dir}')

    print(f'\nStarting parallel processing of {len(time_period[1])} days using {num_processes} processes...')

    # Prepare arguments for each day
    day_args = []
    for jday in time_period[1][:]:
        day_args.append((jday, year, yearpath, nll_dir, inv, region, station_groups, streams_output_dir))
    
    # First round: Process all days in parallel
    print("=== FIRST ROUND ===")
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_single_day, day_args)
    
    # Analyze results and identify failed days
    successful_days = [r for r in results if r['success']]
    failed_days = [r for r in results if not r['success']]
    failed_day_numbers = [r['day'] for r in failed_days]
    
    print(f'\n--- FIRST ROUND SUMMARY ---')
    print(f'Total days processed: {len(results)}')
    print(f'Successful days: {len(successful_days)}')
    print(f'Failed days: {len(failed_days)}')
    
    if failed_days:
        print(f'Failed days: {failed_day_numbers}')
        
        # Retry failed days
        print(f'\n=== RETRY ROUND ===')
        print(f'Retrying {len(failed_day_numbers)} failed days...')
        
        # Prepare arguments for retry
        retry_args = []
        for jday in failed_day_numbers:
            retry_args.append((jday, year, yearpath, nll_dir, inv, region, station_groups, streams_output_dir))
        
        # Process failed days again (potentially with fewer processes to avoid resource conflicts)
        retry_processes = min(num_processes, len(failed_day_numbers))
        with Pool(processes=retry_processes) as pool:
            retry_results = pool.starmap(process_single_day, retry_args)
        
        # Analyze retry results
        retry_successful = [r for r in retry_results if r['success']]
        retry_failed = [r for r in retry_results if not r['success']]
        
        print(f'\n--- RETRY ROUND SUMMARY ---')
        print(f'Days retried: {len(retry_results)}')
        print(f'Successful after retry: {len(retry_successful)}')
        print(f'Still failed: {len(retry_failed)}')
        
        if retry_successful:
            print(f'Days recovered: {[r["day"] for r in retry_successful]}')
        if retry_failed:
            print(f'Permanently failed days: {[r["day"] for r in retry_failed]}')
        
        # Combine all successful results
        all_successful = successful_days + retry_successful
        all_failed = retry_failed
    else:
        print("All days processed successfully!")
        all_successful = successful_days
        all_failed = []
    
    # Final summary with picks statistics
    total_events = sum(r['events'] for r in all_successful)
    total_picks_processed = sum(r.get('picks_processed', 0) for r in all_successful)
    total_picks_valid = sum(r.get('picks_valid', 0) for r in all_successful)
    overall_pick_success_rate = (total_picks_valid / total_picks_processed * 100) if total_picks_processed > 0 else 0
    
    print(f'\n--- FINAL SUMMARY ---')
    print(f'Total days attempted: {len(time_period[1])}')
    print(f'Final successful days: {len(all_successful)}')
    print(f'Final failed days: {len(all_failed)}')
    print(f'Success rate: {len(all_successful)/len(time_period[1])*100:.1f}%')
    print(f'\n--- PROCESSING STATISTICS ---')
    print(f'Total events processed: {total_events}')
    print(f'Total picks processed: {total_picks_processed:,}')
    print(f'Total valid picks: {total_picks_valid:,}')
    print(f'Overall pick success rate: {overall_pick_success_rate:.1f}%')
    print(f'Average picks per event: {total_picks_processed/total_events:.1f}' if total_events > 0 else 'N/A')
    print(f'Average valid picks per event: {total_picks_valid/total_events:.1f}' if total_events > 0 else 'N/A')
    
    if all_failed:
        print(f'\nDays that could not be processed: {[r["day"] for r in all_failed]}')
        print("You may want to check these days manually for data availability or other issues.")
    else:
        print("\nAll days processed successfully!")

    codestop = datetime.datetime.now()
    print(f'\nCode execution time: {codestop - codestart}')
    print(f'Average time per day: {(codestop - codestart) / len(time_period[1])}')


if __name__ == '__main__':
    main()