from eqcorrscan.utils.mag_calc import _sim_WA, _max_p2t, PAZ_WA
from obspy import read_events, read_inventory, read, UTCDateTime
import os
import argparse
import pandas as pd
from pandas import date_range
from os.path import join, exists
from obspy.core.event.base import CreationInfo, Comment
from obspy.core.event.base import WaveformStreamID, TimeWindow
from obspy.core.event.resourceid import ResourceIdentifier
# from colours import colours
from datetime import datetime

import numpy as np
import logging
import eqcorrscan  # Used to get version number
import os
import glob
import matplotlib.pyplot as plt
import itertools
import copy
import random
import pickle
import math

from inspect import currentframe
from scipy.signal import iirfilter, sosfreqz
from collections import Counter
from obspy import Stream, Trace
from obspy.signal.invsim import simulate_seismometer as seis_sim
from obspy.core.event import (
    Amplitude, Pick, WaveformStreamID, Origin, ResourceIdentifier)
from obspy.geodetics import degrees2kilometers


Logger = logging.getLogger(__name__)

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
    return time_period,year

def amp_pick_event(event, st, inventory, chans=('Z',), var_wintype=True,
                   winlen=0.9, pre_pick=0.2, pre_filt=True, lowcut=1.0,
                   highcut=20.0, corners=4, min_snr=1.0, plot=False,
                   remove_old=False, ps_multiplier=0.34, velocity=False,
                   water_level=0, iaspei_standard=False):
    """
    Original from EQcorrscan, modified by Cedrid de Meyer
    Pick amplitudes for local magnitude for a single event.

    Looks for maximum peak-to-trough amplitude for a channel in a stream, and
    picks this amplitude and period.  There are a few things it does
    internally to stabilise the result:

        1. Applies a given filter to the data using obspy's bandpass filter.
        The filter applied is a time-domain digital SOS filter.
        This is often necessary for small magnitude earthquakes.  To correct
        for this filter later the gain of the filter at the period of the
        maximum amplitude is retrieved using scipy's sosfreqz, and used to
        divide the resulting picked amplitude.

        2. Picks the peak-to-trough amplitude, but records half of this to
        cope with possible DC offsets.

        3. The maximum amplitude within the given window is picked. Care must
        be taken to avoid including surface waves in the window;

        4. A variable window-length is used by default that takes into account
        P-S times if available, this is in an effort to include only the
        body waves.  When P-S times are not available the ps_multiplier
        variable is used, which defaults to 0.34 x hypocentral distance.

    :type event: obspy.core.event.event.Event
    :param event: Event to pick
    :type st: obspy.core.stream.Stream
    :param st: Stream associated with event
    :type inventory: obspy.core.inventory.Inventory
    :param inventory:
        Inventory containing response information for the stations in st.
    :type chans: tuple
    :param chans:
        Tuple of the components to pick on, e.g. (Z, 1, 2, N, E)
    :type var_wintype: bool
    :param var_wintype:
        If True, the winlen will be multiplied by the P-S time if both P and
        S picks are available, otherwise it will be multiplied by the
        hypocentral distance*ps_multiplier, defaults to True
    :type winlen: float
    :param winlen:
        Length of window, see above parameter, if var_wintype is False then
        this will be in seconds, otherwise it is the multiplier to the
        p-s time, defaults to 0.9.
    :type pre_pick: float
    :param pre_pick:
        Time before the s-pick to start the cut window, defaults to 0.2.
    :type pre_filt: bool
    :param pre_filt: To apply a pre-filter or not, defaults to True
    :type lowcut: float
    :param lowcut: Lowcut in Hz for the pre-filter, defaults to 1.0
    :type highcut: float
    :param highcut: Highcut in Hz for the pre-filter, defaults to 20.0
    :type corners: int
    :param corners: Number of corners to use in the pre-filter
    :type min_snr: float
    :param min_snr:
        Minimum signal-to-noise ratio to allow a pick - see note below on
        signal-to-noise ratio calculation.
    :type plot: bool
    :param plot: Turn plotting on or off.
    :type remove_old: bool
    :param remove_old:
        If True, will remove old amplitudes and associated picks from event
        and overwrite with new picks. Defaults to False.
    :type ps_multiplier: float
    :param ps_multiplier:
        A p-s time multiplier of hypocentral distance - defaults to 0.34,
        based on p-s ratio of 1.68 and an S-velocity 0f 1.5km/s, deliberately
        chosen to be quite slow.
    :type velocity: bool
    :param velocity:
        Whether to make the pick in velocity space or not. Original definition
        of local magnitude used displacement of Wood-Anderson, MLv in seiscomp
        and Antelope uses a velocity measurement. *velocity and iaspei_standard
        are mutually exclusive*.
    :type water_level: float
    :param water_level:
        Water-level for seismometer simulation, see
        https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.remove_response.html
    :type iaspei_standard: bool
    :param iaspei_standard:
        Whether to output amplitude in IASPEI standard IAML (wood-anderson
        static amplification of 1), or AML with wood-anderson static
        amplification of 2080. Note: Units are SI (and specified in the
        amplitude)

    :returns: Picked event
    :rtype: :class:`obspy.core.event.Event`

    .. Note::
        Signal-to-noise ratio is calculated using the filtered data by
        dividing the maximum amplitude in the signal window (pick window)
        by the normalized noise amplitude (taken from the whole window
        supplied).

    .. Note::
        With `iaspei_standard=False`, picks will be returned in SI units
        (m or m/s), with the standard Wood-Anderson sensitivity of 2080 applied
        such that the measurements reflect the amplitude measured on a Wood
        Anderson instrument, as per the original local magnitude definitions
        of Richter and others.
    """
    #print("hello")
    if iaspei_standard and velocity:
        raise NotImplementedError("Velocity is not IASPEI standard for IAML.")
    try:
        event_origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        event_origin = Origin()
    depth = event_origin.depth
    if depth is None:
        Logger.warning("No depth for the event, setting to 0 km")
        depth = 0
        


    # Remove amplitudes and picks for those amplitudes - this is not always
    # safe: picks may not be exclusively linked to amplitudes - hence the
    # default is *not* to do this.
    if remove_old and event.amplitudes:
        removal_ids = {amp.pick_id for amp in event.amplitudes}
        event.picks = [
            p for p in event.picks if p.resource_id not in removal_ids]
        event.amplitudes = []

    # We just want to look at P and S picks.
    picks = [p for p in event.picks
             if p.phase_hint and p.phase_hint[0].upper() in ("P", "S")]
    
    
    if len(picks) == 0:
        Logger.warning('No P or S picks found')
        return event

    st = st.copy().merge()  # merge the data, just in case! Work on a copy.
    # For each station cut the window
    for sta in {p.waveform_id.station_code for p in picks}:
        for chan in chans:
            Logger.info(f'Working on {sta} {chan}')
            tr = st.select(station=sta, component=chan)
            if not tr:
                Logger.warning(f'{sta} {chan} not found in the stream.')
                continue
            tr = tr.merge()[0]
            # Apply the pre-filter
            if pre_filt:
                tr = tr.split().detrend('simple').merge(fill_value=0)[0]
                tr.filter('bandpass', freqmin=lowcut, freqmax=highcut,
                          corners=corners)
            # tr = _sim_WA(tr, inventory, water_level=water_level,
            #              velocity=velocity)
            # if tr is None:  # None returned when no matching response is found
            #     continue
            try:
                tr = _sim_WA(tr, inventory, water_level=water_level,
                             velocity=velocity)
            except Exception as e:
                Logger.warning(f"No response for {sta} {chan}, skipping: {e}")
                continue
            if tr is None:
                Logger.warning(f"No matching response for {sta} {chan}, skipping")
                continue

            # Get the distance from an appropriate arrival
            sta_picks = [p for p in picks if p.waveform_id.station_code == sta]
            distances = []
            for pick in sta_picks:
                distances += [
                    a.distance for a in event_origin.arrivals
                    if a.pick_id == pick.resource_id and
                    a.distance is not None]
            if len(distances) == 0:
                Logger.error(f"Arrivals for station: {sta} do not contain "
                             "distances. Have you located this event?")
                hypo_dist = None
            else:
                # They should all be the same, but take the mean to be sure...
                distance = sum(distances) / len(distances)
                #print(distance, sta)
                hypo_dist = np.sqrt(
                    np.square(distance) +
                    np.square(depth / 1000))
            #print(hypo_dist, sta)

            # Get the earliest P and S picks on this station
            phase_picks = {"P": None, "S": None}
            for _hint in phase_picks.keys():
                _picks = sorted(
                    [p for p in sta_picks if p.phase_hint[0].upper() == _hint],
                    key=lambda p: p.time)
                if len(_picks) > 0:
                    phase_picks[_hint] = _picks[0]
            p_pick = phase_picks["P"]
            s_pick = phase_picks["S"]
            #print(s_pick)
            # Get the window size.
            if var_wintype:
                if p_pick and s_pick:
                    p_time, s_time = p_pick.time, s_pick.time
                elif s_pick and hypo_dist:
                    s_time = s_pick.time
                    p_time = s_time - (hypo_dist * ps_multiplier)
                elif p_pick and hypo_dist:
                    p_time = p_pick.time
                    s_time = p_time + (hypo_dist * ps_multiplier)
                elif (s_pick or p_pick) and hypo_dist is None:
                    Logger.error(
                        "No hypocentral distance and no matching P and S "
                        f"picks for {sta}, skipping.")
                    continue
                else:
                    raise NotImplementedError(
                        "No p or s picks - you should not have been able to "
                        "get here")
                trim_start = p_time - pre_pick
                #trim_end = s_time + (s_time - p_time) * winlen
                trim_end = s_time + winlen
                print(sta, hypo_dist, s_time - p_time, (s_time - p_time) * 0.3, winlen)
                # Work out the window length based on p-s time or distance
            else:  # Fixed window-length
                if s_pick:
                    s_time = s_pick.time
                elif p_pick and hypo_dist:
                    # In this case, there is no S-pick and the window length is
                    # fixed we need to calculate an expected S_pick based on
                    # the hypocentral distance, this will be quite hand-wavey
                    # as we are not using any kind of velocity model.
                    s_time = p_pick.time + hypo_dist * ps_multiplier
                else:
                    Logger.warning(
                        "No s-pick or hypocentral distance to predict "
                        f"s-arrival for station {sta}, skipping")
                    continue
                trim_start = s_time - pre_pick
                trim_end = s_time + winlen
            tr = tr.trim(trim_start, trim_end)
            if len(tr.data) <= 10:
                Logger.warning(f'Insufficient data for {sta}: {trim_start} - {trim_end}')
                continue
            # Get the amplitude
            try:
                amplitude, period, delay, peak, trough = _max_p2t(
                    tr.data, tr.stats.delta, return_peak_trough=True)
            except ValueError as e:
                Logger.error(e)
                Logger.error(f'No amplitude picked for tr {tr.id}')
                continue
            # Calculate the normalized noise amplitude
            snr = amplitude / np.sqrt(np.mean(np.square(tr.data)))
            if amplitude == 0.0:
                continue
            if snr < min_snr:
                Logger.info(
                    f'Signal to noise ratio of {snr} is below threshold.')
                continue
            if plot:
                plt.plot(np.arange(len(tr.data)), tr.data, 'k')
                plt.scatter(tr.stats.sampling_rate * delay, peak)
                plt.scatter(tr.stats.sampling_rate * (delay + period / 2),
                            trough)
                plt.show()
            Logger.info(f'Amplitude picked: {amplitude}')
            Logger.info(f'Signal-to-noise ratio is: {snr}')
            # Note, amplitude should be in meters at the moment!
            # Remove the pre-filter response
            if pre_filt:
                # Generate poles and zeros for the filter we used earlier.
                # We need to get the gain for the digital SOS filter used by
                # obspy.
                sos = iirfilter(
                    corners, [lowcut / (0.5 * tr.stats.sampling_rate),
                              highcut / (0.5 * tr.stats.sampling_rate)],
                    btype='band', ftype='butter', output='sos')
                _, gain = sosfreqz(sos, worN=[1 / period],
                                   fs=tr.stats.sampling_rate)
                gain = np.abs(gain[0])  # Convert from complex to real.
                if gain < 1e-2:
                    Logger.warning(
                        f"Pick made outside stable pass-band of filter "
                        f"on {tr.id}, rejecting")
                    continue
                amplitude /= gain
                Logger.debug(f"Removed filter gain: {gain}")
            # Write out the half amplitude, approximately the peak amplitude as
            # used directly in magnitude calculations
            amplitude *= 0.5
            # Documentation standards
            module = _sim_WA.__module__
            fname = currentframe().f_code.co_name
            # This is here to ensure that if the function name changes this
            # is still correct
            method_id = ResourceIdentifier(
                id=f"{module}.{fname}",
                prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")
            filter_id = ResourceIdentifier(
                id=f"{module}._sim_WA",
                prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")
            if iaspei_standard:
                # Remove wood-anderson amplification
                units, phase_hint, amplitude_type = (
                    "m", "IAML", "IAML")
                # amplitude *= 10 ** 9  # *THIS IS NOT SUPPORTED BY QML*
                amplitude /= PAZ_WA["sensitivity"]  # Remove WA sensitivity
                # Set the filter ID to state that sensitivity was removed
                filter_id = ResourceIdentifier(
                    id=f"{module}._sim_WA.WA_sensitivity_removed",
                    prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")
            else:  # Not IAML, use SI units.
                if velocity:
                    units, phase_hint, amplitude_type = (
                        "m/s", "AML", "AML")
                else:
                    units, phase_hint, amplitude_type = (
                        "m", "AML", "AML")
            if tr.stats.channel.endswith("Z"):
                magnitude_hint = "MLv"
                # MLv is ML picked on the vertical channel
            else:
                magnitude_hint = "ML"
            # Append an amplitude reading to the event
            _waveform_id = WaveformStreamID(
                station_code=tr.stats.station, channel_code=tr.stats.channel,
                network_code=tr.stats.network)
            pick = Pick(
                waveform_id=_waveform_id, phase_hint=phase_hint,
                polarity='undecidable', time=tr.stats.starttime + delay,
                evaluation_mode='automatic',
                method_id=method_id, filter_id=filter_id)
            event.picks.append(pick)
            event.amplitudes.append(Amplitude(
                generic_amplitude=amplitude, period=period,
                pick_id=pick.resource_id, waveform_id=pick.waveform_id,
                unit=units, magnitude_hint=magnitude_hint,
                type=amplitude_type, category='point', method_id=method_id,
                filter_id=filter_id, 
                time_window = TimeWindow(begin = (trim_end - trim_start)/2, end = (trim_end - trim_start)/2, reference = trim_start + (trim_end - trim_start)/2),
                evaluation_mode = "automatic",
                evaluation_status = "confirmed"))
    return event

def amp_pick_event_precut(event, st, inventory, chans=('Z',), pre_filt=True,
                          lowcut=1.0, highcut=20.0, corners=4, min_snr=1.0,
                          plot=False, remove_old=False, velocity=False,
                          water_level=0, iaspei_standard=False):
    """
    Pick amplitudes for local magnitude using pre-cut waveforms.
    Assumes traces in st are already trimmed around the event window.
    EQcorrscan function modified by Carlos Montalvo
    """
    if iaspei_standard and velocity:
        raise NotImplementedError("Velocity is not IASPEI standard for IAML.")

    try:
        event_origin = event.preferred_origin() or event.origins[0]
    except IndexError:
        event_origin = Origin()
    depth = event_origin.depth
    if depth is None:
        Logger.warning("No depth for the event, setting to 0 km")
        depth = 0

    if remove_old and event.amplitudes:
        removal_ids = {amp.pick_id for amp in event.amplitudes}
        event.picks = [
            p for p in event.picks if p.resource_id not in removal_ids]
        event.amplitudes = []

    picks = [p for p in event.picks
             if p.phase_hint and p.phase_hint[0].upper() in ("P", "S")]
    if len(picks) == 0:
        Logger.warning('No P or S picks found')
        return event

    st = st.copy().merge()

    for sta in {p.waveform_id.station_code for p in picks}:
        for chan in chans:
            Logger.info(f'Working on {sta} {chan}')
            tr = st.select(station=sta, component=chan)
            if not tr:
                Logger.warning(f'{sta} {chan} not found in the stream.')
                continue
            tr = tr.merge()[0]

            # Aplicar prefiltro
            if pre_filt:
                tr = tr.split().detrend('simple').merge(fill_value=0)[0]
                tr.filter('bandpass', freqmin=lowcut, freqmax=highcut,
                          corners=corners)

            # Simular Wood-Anderson
            tr = _sim_WA(tr, inventory, water_level=water_level,
                         velocity=velocity)
            if tr is None:
                continue

            # Verificar datos suficientes
            if len(tr.data) <= 10:
                Logger.warning(f'Insufficient data for {sta}')
                continue

            # Calcular amplitud máxima pico a pico
            try:
                amplitude, period, delay, peak, trough = _max_p2t(
                    tr.data, tr.stats.delta, return_peak_trough=True)
            except ValueError as e:
                Logger.error(e)
                Logger.error(f'No amplitude picked for tr {tr.id}')
                continue

            # SNR
            snr = amplitude / np.sqrt(np.mean(np.square(tr.data)))
            if amplitude == 0.0:
                continue
            if snr < min_snr:
                Logger.info(f'SNR {snr:.2f} below threshold for {sta}')
                continue

            if plot:
                plt.plot(np.arange(len(tr.data)), tr.data, 'k')
                plt.scatter(tr.stats.sampling_rate * delay, peak)
                plt.scatter(tr.stats.sampling_rate * (delay + period / 2), trough)
                plt.title(f"{tr.stats.network}.{tr.stats.station}."
                          f"{tr.stats.location}.{tr.stats.channel}")
                plt.show()

            # Remover respuesta del prefiltro
            if pre_filt:
                sos = iirfilter(
                    corners, [lowcut / (0.5 * tr.stats.sampling_rate),
                              highcut / (0.5 * tr.stats.sampling_rate)],
                    btype='band', ftype='butter', output='sos')
                _, gain = sosfreqz(sos, worN=[1 / period],
                                   fs=tr.stats.sampling_rate)
                gain = np.abs(gain[0])
                if gain < 1e-2:
                    Logger.warning(
                        f"Pick made outside stable pass-band on {tr.id}, rejecting")
                    continue
                amplitude /= gain

            amplitude *= 0.5  # Half peak-to-trough

            # Metadata
            module = _sim_WA.__module__
            fname = currentframe().f_code.co_name
            method_id = ResourceIdentifier(
                id=f"{module}.{fname}",
                prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")
            filter_id = ResourceIdentifier(
                id=f"{module}._sim_WA",
                prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")

            if iaspei_standard:
                units, phase_hint, amplitude_type = ("m", "IAML", "IAML")
                amplitude /= PAZ_WA["sensitivity"]
                filter_id = ResourceIdentifier(
                    id=f"{module}._sim_WA.WA_sensitivity_removed",
                    prefix=f"smi:eqcorrscan{eqcorrscan.__version__}")
            else:
                if velocity:
                    units, phase_hint, amplitude_type = ("m/s", "AML", "AML")
                else:
                    units, phase_hint, amplitude_type = ("m", "AML", "AML")

            if tr.stats.channel.endswith("Z"):
                magnitude_hint = "MLv"
            else:
                magnitude_hint = "ML"

            _waveform_id = WaveformStreamID(
                station_code=tr.stats.station, channel_code=tr.stats.channel,
                network_code=tr.stats.network)
            pick = Pick(
                waveform_id=_waveform_id, phase_hint=phase_hint,
                polarity='undecidable', time=tr.stats.starttime + delay,
                evaluation_mode='automatic',
                method_id=method_id, filter_id=filter_id)
            event.picks.append(pick)
            event.amplitudes.append(Amplitude(
                generic_amplitude=amplitude, period=period,
                pick_id=pick.resource_id, waveform_id=pick.waveform_id,
                unit=units, magnitude_hint=magnitude_hint,
                type=amplitude_type, category='point', method_id=method_id,
                filter_id=filter_id))

    return event

def process_single_day(year, jday, waveforms, amp_dir, nll_dir, inventory):
    """
    Gets event amplitude with EQcorrscan for a single day.
    
    Reads ALL QuakeML events from nll_dir/{year}/{year}_{jday}_nll.xml
    Processes waveforms from waveforms/ subdirectories with new format: YYYY_JDD_HHMMSS/
    Each directory corresponds to one event (HHMMSS is the event time)
    Saves individual results to amp_dir/
    """
    start_time = datetime.now()
    print(f"Starting amplitude picking for year {year}, day {jday}")

    inv = read_inventory(inventory)
    
    # Remove end_date constraints from all stations and channels
    # to allow response lookup for stations outside their recorded end dates
    for network in inv:
        for station in network:
            station.end_date = None
            for channel in station:
                channel.end_date = None

    # Convert julian day to month and day
    month, day = datetime.strptime(f"{year}-{jday}", "%Y-%j").strftime("%m-%d").split('-')
    print(f"Reading data for day {jday} ({year}-{month}-{day})")
    
    # Read ALL events from NLL catalog for this day
    event_file = join(nll_dir, str(year), f"{year}_{jday:03d}_nll.xml")
    if not exists(event_file):
        Logger.error(f"Event file not found: {event_file}")
        return False
    
    try:
        catalog = read_events(event_file)
        print(f"✓ Read {len(catalog)} events from {event_file}")
    except Exception as e:
        Logger.error(f"Error reading event file {event_file}: {e}")
        return False
    
    # Count total picks in catalog
    total_picks = sum(len([p for p in evt.picks 
                          if p.phase_hint and p.phase_hint[0].upper() in ("P", "S")]) 
                     for evt in catalog)
    print(f"  Total P/S picks available: {total_picks}\n")
    
    # Get all waveform directories for this day (new format: YYYY_JDD_HHMMSS)
    date_prefix = f"{year}_{jday:03d}_"
    try:
        all_waveform_dirs = [d for d in os.listdir(waveforms) 
                            if d.startswith(date_prefix) and os.path.isdir(join(waveforms, d))]
    except Exception as e:
        Logger.error(f"Error reading waveform directories: {e}")
        return False
    
    if not all_waveform_dirs:
        Logger.warning(f"No waveform directories found for {date_prefix}")
        return False
    
    all_waveform_dirs.sort()
    print(f"Found {len(all_waveform_dirs)} waveform directories for this day\n")
    
    processed_count = 0
    amplitudes_count = 0
    
    # Process each waveform directory
    for waveform_dir_name in all_waveform_dirs:
        waveform_dir_path = join(waveforms, waveform_dir_name)
        
        # Extract event time from directory name (YYYY_JDD_HHMMSS)
        # Format: YYYY_JDD_HHMMSS, so we split by '_' and take the last part (HHMMSS)
        parts = waveform_dir_name.split('_')
        if len(parts) < 3:
            Logger.warning(f"Invalid waveform directory name format: {waveform_dir_name}")
            continue
        
        event_hms = parts[2]  # HHMMSS (third part after year and jday)
        
        # Find matching event in catalog
        matching_event = None
        try:
            event_hour = int(event_hms[0:2])
            event_minute = int(event_hms[2:4])
            event_second = int(event_hms[4:6])
            
            for evt in catalog:
                evt_origin = evt.preferred_origin() or evt.origins[0]
                evt_time = evt_origin.time
                
                # Match if hour, minute, second match (allow 1 second tolerance)
                if (evt_time.hour == event_hour and 
                    evt_time.minute == event_minute and 
                    abs(evt_time.second - event_second) <= 1):
                    matching_event = evt
                    break
        except (ValueError, IndexError) as e:
            Logger.warning(f"Error extracting time from {waveform_dir_name}: {e}")
            continue
        
        if matching_event is None:
            Logger.warning(f"No matching event found for {waveform_dir_name} ({event_hms})")
            continue
        
        print(f"Processing waveforms from {waveform_dir_name}")
        evt_origin = matching_event.preferred_origin() or matching_event.origins[0]
        print(f"  Event time: {evt_origin.time}")
        
        try:
            # Read all mseed files in the directory (one per station)
            mseed_files = glob.glob(join(waveform_dir_path, '*.mseed'))
            if not mseed_files:
                Logger.warning(f"No mseed files found in {waveform_dir_path}")
                continue
            
            # Read each mseed file and accumulate into a single stream
            st = Stream()
            for mseed_file in sorted(mseed_files):
                try:
                    st_temp = read(mseed_file)
                    st += st_temp
                    Logger.debug(f"Read {mseed_file}: {len(st_temp)} traces")
                except Exception as e:
                    Logger.warning(f"Error reading {mseed_file}: {e}")
                    continue
            
            if len(st) == 0:
                Logger.warning(f"No traces loaded from {waveform_dir_name}")
                continue
            
            print(f"  Stream read: {len(st)} traces from {len(mseed_files)} files")
            
            # DEBUG: Show what stations are available in the stream
            available_stations = set(tr.stats.station for tr in st)
            print(f"  Available stations: {sorted(available_stations)}")
            
            # Get P/S picks from event
            picks = [p for p in matching_event.picks
                     if p.phase_hint and p.phase_hint[0].upper() in ("P", "S")]
            picks_stations = set(p.waveform_id.station_code for p in picks)
            print(f"  Stations with P/S picks: {sorted(picks_stations)}")
            
            # Find intersection
            matching_stations = available_stations & picks_stations
            if matching_stations:
                print(f"  ✓ Can process: {sorted(matching_stations)}")
            else:
                Logger.warning(f"No matching stations between picks and data for {waveform_dir_name}")
                continue
            
            # Apply amplitude picking on a COPY of the event
            print("  Picking amplitudes...")
            event_copy = copy.deepcopy(matching_event)
            picked_event = amp_pick_event_precut(
                event_copy, st, inv, 
                chans=('Z',), 
                pre_filt=True, 
                lowcut=1.0, 
                highcut=20.0, 
                corners=4, 
                min_snr=1.0, 
                plot=False, 
                remove_old=False, 
                velocity=False, 
                water_level=0, 
                iaspei_standard=False
            )
            
            amps_in_event = len(picked_event.amplitudes)
            if amps_in_event > 0:
                amplitudes_count += amps_in_event
                print(f"  ✓ Amplitudes picked: {amps_in_event}")
                
                # Save individual event with amplitudes
                output_filename = f"{year}_{jday}_{event_hms}_amplitudes.xml"
                output_file = join(amp_dir, output_filename)
                try:
                    picked_event.write(output_file, format='QUAKEML')
                    print(f"  Saved to {output_filename}")
                    processed_count += 1
                except Exception as e:
                    Logger.error(f"Error saving event to {output_file}: {e}")
            else:
                Logger.warning(f"No amplitudes picked for {waveform_dir_name}")
            
        except Exception as e:
            Logger.error(f"Error processing waveforms from {waveform_dir_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing completed in {datetime.now() - start_time}")
    print(f"Events:     {processed_count}/{len(catalog)}")
    print(f"Amplitudes: {amplitudes_count}")
    print(f"{'='*60}")
    
    return processed_count > 0
        

    
def main():
    """ Main function """
    parser = argparse.ArgumentParser(description="Pick amplitudes for local magnitude")
    parser.add_argument('--year', required=True, type=int, help='Year to process (YYYY)')
    parser.add_argument('--jday', required=True, type=int, help='Julian day to process (001-366)')
    parser.add_argument('--inventory', required=False, help='Inventory file path (optional)')

    args = parser.parse_args()

    ### DIRECTORIES AND FILES ###
    montalca = r'/Volumes/GeoPhysics_49/users-data/montalca'
    # Catalogues
    ctlgdir = join(montalca, 'CATALOGS')
    nll_dir = join(ctlgdir, 'NLL')
    rpnet_dir = join(ctlgdir, 'RPNET')
    waveforms = join(rpnet_dir, 'WAVEFORMS')
    amp_dir = join(ctlgdir, 'MAGNITUDES', 'AMPLITUDES')

    # Stations
    sta_dir = join(montalca, 'STATIONS')
    if args.inventory:
        inventory_file = args.inventory
    else:
        inventory_file = join(sta_dir, 'MORIA_GEONET.xml')
        # inventory_file = join(sta_dir, 'ALL_STATIONS.xml')
    
    # Verify inventory file exists
    if not exists(inventory_file):
        print(f"ERROR: Inventory file not found: {inventory_file}")
        return False

    # Create output directory if it doesn't exist
    if not exists(amp_dir):
        os.makedirs(amp_dir)
        print(f'Created amplitudes directory: {amp_dir}')

    success = process_single_day(
        year=args.year,
        jday=args.jday,
        waveforms=waveforms,
        amp_dir=amp_dir,
        nll_dir=nll_dir,
        inventory=inventory_file
    )
    
    if success:
        print(f"\n✓ Processing completed successfully for {args.year}-{args.jday}")
    else:
        print(f"\n✗ Processing failed for {args.year}-{args.jday}")
    
    return success

if __name__ == "__main__":
    main()
