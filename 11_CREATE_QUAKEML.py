import glob
from os import makedirs
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
    
def nll_quakeml_to_csv_split(quakeml_file, output_dir,
                            station_xml="/Volumes/GeoPhysics_49/users-data/montalca/STATIONS/ALL_STATIONS.xml",
                            taup_model="/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.npz"):
    from obspy.geodetics import gps2dist_azimuth
    from obspy.taup import TauPyModel
    from obspy import read_inventory
    import os

    print(f">>> Leyendo archivo: {quakeml_file}")
    catalog = read_events(quakeml_file)
    print(f">>> Eventos cargados: {len(catalog)}")

    # Cargar inventario y modelo UNA sola vez fuera del loop
    print(f">>> Cargando inventario: {station_xml}")
    inventory = read_inventory(station_xml)
    # taup_model = TauPyModel(model="iasp91")

    # Construir lookup dict: station_code -> (lat, lon)
    station_coords = {}
    for net in inventory:
        for sta in net:
            station_coords[sta.code] = (sta.latitude, sta.longitude)
    print(f">>> Estaciones cargadas: {len(station_coords)}")

    event_records = []
    phase_records = []

    for event in catalog:
        origin = event.preferred_origin() or (event.origins[0] if event.origins else None)
        if origin is None:
            continue

        preferred_mag_id = str(event.preferred_magnitude_id) if event.preferred_magnitude_id else None
        mag1 = None
        if preferred_mag_id:
            for m in event.magnitudes:
                if str(m.resource_id) == preferred_mag_id:
                    mag1 = m
                    break
        if mag1 is None and event.magnitudes:
            mag1 = event.magnitudes[0]

        if mag1 is None:
            class DummyMagnitude:
                mag = 2.0
                magnitude_type = 'ML'
            mag1 = DummyMagnitude()

        event_id = str(event.resource_id).split('/')[-1]
        data_id = f"{origin.time.year}_{origin.time.julday:03d}_{origin.time.hour:02d}{origin.time.minute:02d}{origin.time.second:02d}"

        event_record = {
            'event_id': event_id,
            'data_id':  data_id,
            'jst':      origin.time.isoformat() if origin.time else None,
            'lat':      origin.latitude,
            'lon':      origin.longitude,
            'dep':      origin.depth / 1000.0 if origin.depth is not None else None,
            'mag':      mag1.mag if mag1 else None,
            'tmag':     mag1.magnitude_type if mag1 else None,
        }
        event_records.append(event_record)

        # Lookup arrivals por pick_id
        arrival_lookup = {str(a.pick_id): a for a in origin.arrivals}

        dep_km = origin.depth / 1000.0 if origin.depth is not None else 0.0

        for pick in event.picks:
            if pick.waveform_id is None:
                continue

            sta_code  = pick.waveform_id.station_code
            phase_type = pick.phase_hint if pick.phase_hint else None
            polarity   = str(pick.polarity) if pick.polarity else None

            # Distance desde arrival (en grados)
            arrival = arrival_lookup.get(str(pick.resource_id))
            distance = arrival.distance if arrival and arrival.distance is not None else None

            # Azimuth: evento -> estación
            azimuth = None
            sta_coords = station_coords.get(sta_code)
            if sta_coords and origin.latitude is not None and origin.longitude is not None:
                _, az, _ = gps2dist_azimuth(
                    origin.latitude, origin.longitude,
                    sta_coords[0],   sta_coords[1]
                )
                azimuth = round(az, 4)

            # Takeoff angle via TauPy
            takeoff = None
            if distance is not None and phase_type in ('P', 'S'):
                try:
                    ray_paths = taup_model.get_ray_paths(
                        source_depth_in_km=dep_km,
                        distance_in_degree=distance,
                        phase_list=[phase_type]
                    )
                    takeoff = round(ray_paths[0].takeoff_angle, 4) if ray_paths else None
                except Exception:
                    takeoff = None

            pick_time = pick.time.isoformat() if pick.time else None
            phase_record = {
                'event_id': event_id,
                'sta':      sta_code,
                'phase':    phase_type,
                'polarity': polarity,
                'distance': distance,
                'azimuth':  azimuth,
                'takeoff':  takeoff,
                'network':  pick.waveform_id.network_code if pick.waveform_id else None,
                'channel':  pick.waveform_id.channel_code if pick.waveform_id else None,
                'ptime':    pick_time if phase_type == 'P' else None,
                'stime':    pick_time if phase_type == 'S' else None,
            }
            phase_records.append(phase_record)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_events = pd.DataFrame(event_records)
    event_csv_path = join(output_dir, 'event_catalogue.csv')
    df_events.to_csv(event_csv_path, index=False)
    print(f"✓ Event catalog saved: {event_csv_path} ({len(df_events)} events)")

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
mode = 3

if mode == 1:
    merge_quakeml_files(amp_dir, join(mag_dir, 'JAN24_SEP25.xml'))
elif mode == 2:
    convert_geonet_to_quakeml(gn_file, join(mag_dir,'GeoNet_CMT_solutions.xml'))
elif mode == 3:
    # Example: Convert a QuakeML file to event and phase catalogs
    quakeml_input = join(mag_dir, 'JAN24_SEP25_MAGNITUDES.xml')  # Change this to your QuakeML file
    if exists(quakeml_input):
        df_events, df_phases = nll_quakeml_to_csv_split(quakeml_input, rpnet_dir)
    else:
        print(f"QuakeML file not found: {quakeml_input}")