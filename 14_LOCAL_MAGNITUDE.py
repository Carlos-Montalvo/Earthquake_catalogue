from os.path import join
from obspy import read_events, Catalog, read_inventory
from obspy.core.event import Magnitude, ResourceIdentifier
import numpy as np
import logging
import glob
import os
from math import radians, cos, sin, asin, sqrt

Logger = logging.getLogger(__name__)

# Load station coordinates from MORIA_GEONET.xml
STATION_COORDS = {}

def _load_station_inventory():
    """Load station inventory from XML file and extract coordinates"""
    global STATION_COORDS
    
    inventory_file = "/Volumes/GeoPhysics_49/users-data/montalca/STATIONS/MORIA_GEONET.xml"
    
    try:
        Logger.info(f"Loading station inventory from {inventory_file}")
        inventory = read_inventory(inventory_file)
        
        # Extract coordinates for all stations
        for network in inventory:
            for station in network:
                sta_code = station.code
                lat = station.latitude
                lon = station.longitude
                STATION_COORDS[sta_code] = (lat, lon)
                Logger.debug(f"Loaded {network.code}.{sta_code}: ({lat}, {lon})")
        
        Logger.info(f"Loaded {len(STATION_COORDS)} station coordinates")
        
    except Exception as e:
        Logger.error(f"Failed to load station inventory: {e}")
        Logger.warning("Will continue with empty station coordinates")

def distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance using Haversine formula.
    
    Parameters
    ----------
    lat1, lon1 : float
        Origin latitude/longitude in decimal degrees
    lat2, lon2 : float
        Target latitude/longitude in decimal degrees
    
    Returns
    -------
    float
        Distance in kilometers
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in km
    return c * r

def get_station_coords(station_code, network_code='NZ'):
    """
    Get station coordinates from loaded inventory.
    
    Parameters
    ----------
    station_code : str
        Station code (e.g., 'BFZ')
    network_code : str
        Network code (e.g., 'NZ') - not used, just for consistency
    
    Returns
    -------
    tuple or None
        (latitude, longitude) or None if not found
    """
    station_code_upper = station_code.upper()
    if station_code_upper in STATION_COORDS:
        return STATION_COORDS[station_code_upper]
    return None

def calculate_local_magnitudes(quakeml_file_or_dir):
    """
    Calculate local magnitudes (ML) for each event from a QuakeML catalog.
    
    For each event, calculates ML for each station with an amplitude pick,
    then averages across all stations to get the final event magnitude.
    
    Parameters
    ----------
    quakeml_file_or_dir : str
        Path to a QuakeML file OR a directory containing QuakeML files.
        If directory: processes all *.xml files and saves with suffix '_local_magnitude.xml'
        If file: processes single file and saves with suffix '_local_magnitude.xml'
    
    Returns
    -------
    catalog : obspy.Catalog
        Updated catalog with local magnitudes added
        
    Notes
    -----
    The local magnitude formula is:
        ML = log10(A) + 2.48 + 3.0*log10(Δ)
    where:
        A = amplitude in mm (half amplitude used by default)
        Δ = epicentral distance in km
    
    This assumes the amplitudes in the QuakeML file are already 
    corrected for instrumental response (as done by amplitude_picker_single_day.py).
    """
    
    # Determine if input is file or directory
    if os.path.isdir(quakeml_file_or_dir):
        # Process all XML files in directory
        quakeml_files = glob.glob(join(quakeml_file_or_dir, '*.xml'))
        if not quakeml_files:
            Logger.warning(f"No XML files found in {quakeml_file_or_dir}")
            return None
    elif os.path.isfile(quakeml_file_or_dir):
        # Single file
        quakeml_files = [quakeml_file_or_dir]
    else:
        raise FileNotFoundError(f"Path not found: {quakeml_file_or_dir}")
    
    Logger.info(f"Processing {len(quakeml_files)} file(s)")
    
    all_catalogs = []
    
    for quakeml_file in quakeml_files:
        Logger.info(f"\n{'='*70}")
        Logger.info(f"Processing: {os.path.basename(quakeml_file)}")
        Logger.info(f"{'='*70}")
        
        # Generate output filename
        base_name = os.path.basename(quakeml_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(
            os.path.dirname(quakeml_file),
            f"{base_name_no_ext}_local_magnitude.xml"
        )
        
        # Read the catalog
        catalog = read_events(quakeml_file)
        Logger.info(f"Loaded {len(catalog)} events from {os.path.basename(quakeml_file)}")
        
        events_with_magnitudes = 0
        events_without_amplitudes = 0
        
        for event in catalog:
            # Get origin information
            origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
            if origin is None:
                Logger.warning(f"Event {event.resource_id} has no origin, skipping")
                continue
            
            # DEBUG: Print event info on first event
            if events_with_magnitudes == 0 and events_without_amplitudes == 0:
                Logger.info(f"DEBUG - Event {event.resource_id}:")
                Logger.info(f"  - {len(event.amplitudes)} amplitudes")
                Logger.info(f"  - {len(event.picks)} picks")
                Logger.info(f"  - {len(origin.arrivals)} arrivals")
                if len(event.amplitudes) > 0:
                    amp = event.amplitudes[0]
                    Logger.info(f"  - First amplitude attributes: {vars(amp)}")
            
            # Group amplitudes by station
            station_magnitudes = {}  # {station_code: [magnitude1, magnitude2, ...]}
            
            for amplitude in event.amplitudes:
                # Get waveform info - it's stored directly in amplitude
                if amplitude.waveform_id is None:
                    Logger.debug(f"No waveform_id in amplitude")
                    continue
                
                station_code = amplitude.waveform_id.station_code
                
                # Get amplitude value - try multiple attributes
                amp_value = None
                if hasattr(amplitude, 'generic_amplitude') and amplitude.generic_amplitude is not None:
                    amp_value = amplitude.generic_amplitude
                elif hasattr(amplitude, 'mag') and amplitude.mag is not None:
                    amp_value = amplitude.mag
                
                if amp_value is None or amp_value == 0:
                    Logger.debug(f"No valid amplitude value for {station_code}")
                    continue
                
                # Convert from m to mm if needed (amplitudes are typically in m)
                # amplitude_picker_single_day multiplies by 0.5 (half-amplitude) and stores in meters
                # Must multiply by 2 to get full pico-a-pico amplitude for Richter formula
                if amp_value > 0:
                    # If it's in meters (typical from amplitude_picker), convert to mm
                    if amp_value < 1.0:  # Likely in meters (0.001 - 0.1 m is typical)
                        amp_mm = amp_value * 2 * 1000  # ×2 for half→full amplitude, ×1000 for m→mm
                    else:
                        amp_mm = amp_value * 2  # Assume already in mm, just apply half→full correction
                else:
                    continue
                
                # Get epicentral distance for this station from coordinates
                # Since origin.arrivals is often empty, we calculate from station coordinates
                dist_km = None
                
                # Get origin lat/lon (may be float directly or wrapped in a Quantity object)
                orig_lat = origin.latitude
                orig_lon = origin.longitude
                if hasattr(orig_lat, 'value'):
                    orig_lat = orig_lat.value
                if hasattr(orig_lon, 'value'):
                    orig_lon = orig_lon.value
                
                if orig_lat is not None and orig_lon is not None:
                    # Get station coordinates
                    station_network = amplitude.waveform_id.network_code
                    sta_coords = get_station_coords(station_code, station_network)
                    
                    if sta_coords is not None:
                        sta_lat, sta_lon = sta_coords
                        dist_km = distance_km(
                            orig_lat,
                            orig_lon,
                            sta_lat,
                            sta_lon
                        )
                    else:
                        Logger.debug(f"No coordinates for {station_network}.{station_code}")
                        continue
                elif origin.arrivals:
                    # Fallback: try to find distance from arrivals
                    if amplitude.pick_id:
                        for arrival in origin.arrivals:
                            if arrival.pick_id == amplitude.pick_id and arrival.distance is not None:
                                dist_km = arrival.distance * 111.195  # Convert degrees to km
                                break
                
                if dist_km is None or dist_km <= 0:
                    Logger.debug(f"Could not calculate distance for {station_code}")
                    continue
                
                # Calculate local magnitude using Richter's formula
                # ML = log10(A) + 2.48 + 3.0*log10(Δ)
                # where A is amplitude in mm and Δ is epicentral distance in km
                try:
                    if amp_mm > 0:
                        ml = np.log10(amp_mm) + 2.48 + 3.0 * np.log10(dist_km)
                        
                        # Store magnitude for this station
                        if station_code not in station_magnitudes:
                            station_magnitudes[station_code] = []
                        station_magnitudes[station_code].append(ml)
                        
                        Logger.debug(
                            f"Event {event.resource_id}: Station {station_code} - "
                            f"Amplitude={amp_mm:.4f}mm, Distance={dist_km:.2f}km, ML={ml:.2f}")
                except (ValueError, TypeError) as e:
                    Logger.warning(
                        f"Could not calculate magnitude for {station_code}: {e}")
                    continue
            
            # Calculate average magnitude from all stations
            if len(station_magnitudes) == 0:
                Logger.debug(f"Event {event.resource_id} has no valid amplitude picks")
                events_without_amplitudes += 1
                continue
            
            # Flatten all magnitudes and calculate mean
            all_magnitudes = []
            for sta_mags in station_magnitudes.values():
                all_magnitudes.extend(sta_mags)
            
            if len(all_magnitudes) == 0:
                continue
            
            mean_magnitude = np.mean(all_magnitudes)
            std_magnitude = np.std(all_magnitudes) if len(all_magnitudes) > 1 else 0.0
            
            # Create Magnitude object
            magnitude = Magnitude(
                mag=mean_magnitude,
                magnitude_type='ML',
                origin_id=origin.resource_id,
                station_count=len(station_magnitudes),
                azimuthal_gap=None,
                evaluation_status='reviewed',
                resource_id=ResourceIdentifier()
            )
            
            # Add magnitude to event
            event.magnitudes.append(magnitude)
            
            # Set as preferred magnitude
            event.preferred_magnitude_id = magnitude.resource_id
            
            Logger.info(
                f"Event {event.resource_id}: "
                f"ML={mean_magnitude:.2f}±{std_magnitude:.2f} "
                f"({len(station_magnitudes)} stations, "
                f"{len(all_magnitudes)} amplitudes)")
            
            events_with_magnitudes += 1
        
        Logger.info(f"Processed {events_with_magnitudes} events with magnitudes")
        Logger.info(f"Skipped {events_without_amplitudes} events without valid amplitudes")
        
        # Write updated catalog
        catalog.write(output_file, format='QUAKEML')
        Logger.info(f"Wrote catalog with magnitudes to {os.path.basename(output_file)}")
        
        all_catalogs.append(catalog)
    
    # Return last catalog if single file, or all catalogs if directory
    return all_catalogs[0] if len(all_catalogs) == 1 else all_catalogs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Calculate local magnitudes from QuakeML catalog(s)')
    parser.add_argument(
        'input_dir',
        help='Directory containing QuakeML file(s) or path to single QuakeML file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load station inventory
    _load_station_inventory()
    
    calculate_local_magnitudes(args.input_dir)


