import glob
from datetime import datetime
from os.path import join,exists
import pandas as pd
from obspy import read_events, Catalog, UTCDateTime
from obspy.core.event import (
    Event, Origin, Magnitude, FocalMechanism, NodalPlane, NodalPlanes,
    MomentTensor, Tensor, ResourceIdentifier)

### FUNCTIONS ###
def _to_float(value):
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_geonet_datetime(value):
    """Parse several possible GeoNet date encodings."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value).strip()
    if not text:
        return None

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        # Convert to naive datetime (remove timezone info) for consistent comparisons
        return parsed.to_pydatetime().replace(tzinfo=None)

    try:
        digits = str(abs(int(float(text))))
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())

    if len(digits) >= 14:
        try:
            return datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            return None

    if len(digits) == 8:
        try:
            return datetime.strptime(digits, "%Y%m%d")
        except ValueError:
            return None

    return None

def convert_geonet_to_quakeml(geonet_catalog_file, output_file, 
                               lon_min=170, lon_max=177.6, 
                               lat_min=-45.3, lat_max=-38.3):
    geonet_catalog = pd.read_csv(geonet_catalog_file)
    catalog = Catalog()
    start_date = datetime(2024, 1, 1)

    for _, row in geonet_catalog.iterrows():
        origin_time = _parse_geonet_datetime(row.get("Date"))
        if origin_time is None or origin_time < start_date:
            continue

        public_id = row.get("PublicID")
        if pd.isna(public_id):
            continue
        
        # Filter by region
        latitude = _to_float(row.get("Latitude"))
        longitude = _to_float(row.get("Longitude"))
        if latitude is None or longitude is None:
            continue
        if not (lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max):
            continue

        event = Event(
            resource_id=ResourceIdentifier(
                id=f"smi:local/geonet/event/{str(public_id)}"))

        depth_km = _to_float(row.get("Depth"))
        depth_m = None if depth_km is None else depth_km * 1000.0

        origin = Origin(
            time=UTCDateTime(origin_time),
            latitude=latitude,
            longitude=longitude,
            depth=depth_m,
            resource_id=ResourceIdentifier(
                id=f"smi:local/geonet/origin/{str(public_id)}"))
        event.origins = [origin]
        event.preferred_origin_id = origin.resource_id

        ml = _to_float(row.get("ML"))
        mw = _to_float(row.get("Mw"))

        if ml is not None:
            ml_mag = Magnitude(
                mag=ml,
                magnitude_type="ML",
                origin_id=origin.resource_id,
                resource_id=ResourceIdentifier(
                    id=f"smi:local/geonet/magnitude/{str(public_id)}/ML"))
            event.magnitudes.append(ml_mag)

        if mw is not None:
            mw_mag = Magnitude(
                mag=mw,
                magnitude_type="Mw",
                origin_id=origin.resource_id,
                resource_id=ResourceIdentifier(
                    id=f"smi:local/geonet/magnitude/{str(public_id)}/Mw"))
            event.magnitudes.append(mw_mag)
            event.preferred_magnitude_id = mw_mag.resource_id
        elif ml is not None:
            event.preferred_magnitude_id = event.magnitudes[-1].resource_id

        strike1 = _to_float(row.get("strike1"))
        dip1 = _to_float(row.get("dip1"))
        rake1 = _to_float(row.get("rake1"))
        strike2 = _to_float(row.get("strike2"))
        dip2 = _to_float(row.get("dip2"))
        rake2 = _to_float(row.get("rake2"))

        mxx = _to_float(row.get("Mxx"))
        mxy = _to_float(row.get("Mxy"))
        mxz = _to_float(row.get("Mxz"))
        myy = _to_float(row.get("Myy"))
        myz = _to_float(row.get("Myz"))
        mzz = _to_float(row.get("Mzz"))

        tensor = None
        if any(value is not None for value in (mxx, mxy, mxz, myy, myz, mzz)):
            tensor = Tensor(
                m_rr=mzz,
                m_tt=mxx,
                m_pp=myy,
                m_rt=mxz,
                m_rp=myz,
                m_tp=mxy)

        nodal_planes = None
        if all(value is not None for value in (strike1, dip1, rake1)):
            nodal_plane_1 = NodalPlane(strike=strike1, dip=dip1, rake=rake1)
            nodal_plane_2 = None
            if all(value is not None for value in (strike2, dip2, rake2)):
                nodal_plane_2 = NodalPlane(
                    strike=strike2, dip=dip2, rake=rake2)
            nodal_planes = NodalPlanes(
                nodal_plane_1=nodal_plane_1,
                nodal_plane_2=nodal_plane_2)

        if tensor is not None or nodal_planes is not None:
            moment_tensor = None
            if tensor is not None:
                moment_tensor = MomentTensor(
                    tensor=tensor,
                    scalar_moment=_to_float(row.get("Mo")),
                    resource_id=ResourceIdentifier(
                        id=f"smi:local/geonet/momenttensor/{str(public_id)}"))
            focal_mechanism = FocalMechanism(
                nodal_planes=nodal_planes,
                moment_tensor=moment_tensor,
                resource_id=ResourceIdentifier(
                    id=f"smi:local/geonet/focalmechanism/{str(public_id)}"),
                triggering_origin_id=origin.resource_id)
            event.focal_mechanisms = [focal_mechanism]
            event.preferred_focal_mechanism_id = focal_mechanism.resource_id

        catalog.events.append(event)

    catalog.write(output_file, format="QUAKEML")
    return catalog

def merge_quakeml_files(input_dir, output_file):
    # Get all QuakeML files in the input directory
    filenames = glob.glob(join(input_dir, "*.xml"))
    
    # Initialize an empty Catalog
    master_catalog = Catalog()
    
    # Loop through each file and extend the master catalog
    for file in filenames:
        temp_cat = read_events(file)
        master_catalog.extend(temp_cat)
    
    # Write the final merged catalog to a new QuakeML file
    master_catalog.write(output_file, format="QUAKEML")
    
def nll_quakeml_to_csv_split(quakeml_file, output_dir):
    """
    Convert QuakeML to two CSV files: event catalog and phase catalog.
    
    Event catalog: one record per event with magnitudes and focal mechanism
    Phase catalog: one record per pick/arrival for each event
    """
    from obspy.geodetics import gps2dist_azimuth
    import os
    
    catalog = read_events(quakeml_file)
    
    event_records = []
    phase_records = []
    
    for event in catalog:
        # Get origin, magnitudes, focal mechanism
        origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
        if origin is None:
            continue
        
        # Get first and second magnitudes
        mag1 = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
        mag2 = event.magnitudes[1] if len(event.magnitudes) > 1 else None
        
        # Add dummy magnitude if none exists
        if mag1 is None:
            class DummyMagnitude:
                def __init__(self):
                    self.mag = 3.0
                    self.magnitude_type = 'ML'
            mag1 = DummyMagnitude()
        
        # Get focal mechanism
        fm = event.preferred_focal_mechanism() or (event.focal_mechanisms[0] if event.focal_mechanisms else None)
        
        event_id = str(event.resource_id).split('/')[-1]  # Extract just the ID part
        
        # Get data_id from event time (YYYY_JDD_HHMMSS format)
        data_id = f"{origin.time.year}_{origin.time.julday:03d}_{origin.time.hour:02d}{origin.time.minute:02d}{origin.time.second:02d}"
        
        # Extract focal mechanism parameters
        strike, dip, slip = None, None, None
        if fm and fm.nodal_planes and fm.nodal_planes.nodal_plane_1:
            np1 = fm.nodal_planes.nodal_plane_1
            strike = np1.strike
            dip = np1.dip
            slip = np1.rake
        
        # Event record
        event_record = {
            'event_id': event_id,
            'data_id': data_id,
            'jst': origin.time.isoformat() if origin.time else None,
            'lat': origin.latitude,
            'lon': origin.longitude,
            'dep': origin.depth / 1000.0 if origin.depth is not None else None,
            'mag': mag1.mag if mag1 else None,
            'tmag': mag1.magnitude_type if mag1 else None,
            'mag2': mag2.mag if mag2 else None,
            'tmag2': mag2.magnitude_type if mag2 else None,
            'strike': strike,
            'dip': dip,
            'slip': slip,
            'dist': None  # Placeholder for hypocentral distance
        }
        event_records.append(event_record)
        
        # Phase records (one per pick/arrival)
        for pick in event.picks:
            if pick.waveform_id is None:
                continue
            
            sta_code = pick.waveform_id.station_code
            phase_type = pick.phase_hint if pick.phase_hint else None
            polarity = pick.polarity if pick.polarity else None
            
            # Try to get distance from arrivals
            distance = None
            azimuth = None
            takeoff = None
            for arrival in origin.arrivals:
                if arrival.pick_id == pick.resource_id:
                    distance = arrival.distance if arrival.distance else None
                    azimuth = arrival.azimuth if arrival.azimuth else None
                    takeoff = arrival.takeoff_angle if arrival.takeoff_angle else None
                    break
            
            # Create phase_record with dynamic time column based on phase type
            phase_record = {
                'event_id': event_id,
                'sta': sta_code,
                'phase': phase_type,
                'polarity': polarity,
                'distance': distance,
                'azimuth': azimuth,
                'takeoff': takeoff,
                'network': pick.waveform_id.network_code if pick.waveform_id else None,
                'channel': pick.waveform_id.channel_code if pick.waveform_id else None
            }
            
            # Add time as ptime or stime depending on phase
            pick_time = pick.time.isoformat() if pick.time else None
            if phase_type == 'P':
                phase_record['ptime'] = pick_time
                phase_record['stime'] = None
            elif phase_type == 'S':
                phase_record['ptime'] = None
                phase_record['stime'] = pick_time
            else:
                # Unknown phase type, put in ptime by default
                phase_record['ptime'] = pick_time
                phase_record['stime'] = None
            
            phase_records.append(phase_record)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save event catalog
    df_events = pd.DataFrame(event_records)
    event_csv_path = join(output_dir, 'event_catalogue.csv')
    df_events.to_csv(event_csv_path, index=False)
    print(f"✓ Event catalog saved: {event_csv_path} ({len(df_events)} events)")
    
    # Save phase catalog
    df_phases = pd.DataFrame(phase_records)
    phase_csv_path = join(output_dir, 'phase_catalogue.csv')
    df_phases.to_csv(phase_csv_path, index=False)
    print(f"✓ Phase catalog saved: {phase_csv_path} ({len(df_phases)} picks)")
    
    return df_events, df_phases

### DIRECTORIES ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
ctlg_dir = join(basedir,'CATALOGS')
mag_dir = join(ctlg_dir,'MAGNITUDES')
amp_dir = join(mag_dir,'AMPLITUDES')
rpnet_dir = join(ctlg_dir, 'RPNET')
gn_file = join(ctlg_dir,'GeoNet_CMT_solutions_corregido.csv')

# Flags: mode = 1 for merging QuakeML files, mode = 2 for converting GeoNet CSV to QuakeML, mode = 3 for splitting QuakeML to CSV
mode = 1

if mode == 1:
    merge_quakeml_files(amp_dir, join(mag_dir, 'JAN24_SEP25.xml'))
elif mode == 2:
    convert_geonet_to_quakeml(gn_file, join(mag_dir,'GeoNet_CMT_solutions.xml'))
elif mode == 3:
    # Example: Convert a QuakeML file to event and phase catalogs
    quakeml_input = join(mag_dir, 'JAN24_SEP25.xml')  # Change this to your QuakeML file
    if exists(quakeml_input):
        df_events, df_phases = nll_quakeml_to_csv_split(quakeml_input, rpnet_dir)
    else:
        print(f"QuakeML file not found: {quakeml_input}")