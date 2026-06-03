#!/usr/bin/env python3
"""
Add arrivals to a QuakeML catalog by calculating distances from event location to station.

Usage:
    python add_arrivals_to_catalog.py -i input_catalog.xml -s stations.xml -o output_catalog.xml
"""

import argparse
from obspy import read_events, read_inventory, Catalog
from obspy.core.event import Arrival, ResourceIdentifier
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
import sys

def add_arrivals_to_event(event, inventory):
    """
    Add Arrival objects to an event by calculating distances from picks.
    
    Parameters
    ----------
    event : Event
        Event to add arrivals to
    inventory : Inventory
        Station inventory to look up station coordinates
    
    Returns
    -------
    Event
        Modified event with arrivals added
    """
    ori = event.preferred_origin() or (event.origins[-1] if event.origins else None)
    if ori is None:
        return event
    
    event_lat = ori.latitude
    event_lon = ori.longitude
    event_depth = ori.depth / 1000.0 if ori.depth else 0  # Convert to km
    
    arrivals_added = 0
    
    # For each pick in the event
    for pick in event.picks:
        waveform_id = pick.waveform_id
        network = waveform_id.network_code
        station = waveform_id.station_code
        
        # Try to find station in inventory
        try:
            sta_obj = inventory.select(network=network, station=station)[0][0]
            sta_lat = sta_obj.latitude
            sta_lon = sta_obj.longitude
            
            # Calculate epicentral distance in kilometers
            dist_m, _, _ = gps2dist_azimuth(
                lat1=event_lat, 
                lon1=event_lon,
                lat2=sta_lat, 
                lon2=sta_lon
            )
            dist_km = dist_m / 1000.0
            
            # Convert to degrees
            dist_deg = kilometers2degrees(dist_km)
            
            # Create Arrival object
            arrival = Arrival(
                pick_id=pick.resource_id,
                phase=pick.phase_hint or "P",  # Default to P if not specified
                distance=dist_deg,  # Distance in degrees
                azimuth=None,  # We could calculate this if needed
                takeoff_angle=None,
            )
            
            ori.arrivals.append(arrival)
            arrivals_added += 1
            
        except (IndexError, AttributeError):
            # Station not found in inventory
            pass
    
    return event, arrivals_added

def process_catalog(catalog_file, inventory_file, output_file):
    """
    Read catalog, add arrivals using inventory, write to output.
    """
    print(f"Reading catalog: {catalog_file}")
    catalog = read_events(catalog_file)
    print(f"  Total events: {len(catalog)}")
    print(f"  Total picks: {sum(len(e.picks) for e in catalog)}")
    print(f"  Arrivals before: {sum(len(e.preferred_origin().arrivals) if e.preferred_origin() else 0 for e in catalog)}")
    
    print(f"\nReading inventory: {inventory_file}")
    inventory = read_inventory(inventory_file)
    print(f"  Networks: {len(inventory)}")
    total_stations = sum(len(net) for net in inventory)
    print(f"  Total stations: {total_stations}")
    
    print("\nProcessing events...")
    total_arrivals_added = 0
    events_with_arrivals = 0
    
    for i, event in enumerate(catalog):
        ori = event.preferred_origin() or (event.origins[-1] if event.origins else None)
        if ori is None:
            continue
        
        event, arrivals_added = add_arrivals_to_event(event, inventory)
        if arrivals_added > 0:
            total_arrivals_added += arrivals_added
            events_with_arrivals += 1
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(catalog)} events")
    
    print(f"\nResults:")
    print(f"  Events with arrivals added: {events_with_arrivals}")
    print(f"  Total arrivals added: {total_arrivals_added}")
    print(f"  Arrivals now: {sum(len(e.preferred_origin().arrivals) if e.preferred_origin() else 0 for e in catalog)}")
    
    print(f"\nWriting output: {output_file}")
    catalog.write(output_file, format="QUAKEML")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add Arrival objects to a QuakeML catalog by calculating distances from picks"
    )
    
    parser.add_argument(
        "-i", "--input", 
        type=str, 
        required=True,
        help="Input QuakeML catalog file"
    )
    parser.add_argument(
        "-s", "--stations", 
        type=str, 
        required=True,
        help="Station inventory XML file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output QuakeML catalog file with arrivals"
    )
    
    args = parser.parse_args()
    
    try:
        process_catalog(args.input, args.stations, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
