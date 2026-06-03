#!/usr/bin/env python3
"""
Scan MSEED files for data integrity issues (incomplete/corrupted files).
"""

import sys
from glob import glob
from pathlib import Path
from obspy import read
from datetime import datetime, timedelta
import json

def check_mseed_integrity(file_path, min_duration_hours=18):
    """Check if MSEED file has sufficient temporal coverage."""
    try:
        st = read(file_path)
        if len(st) == 0:
            return False, 0, "Empty"
        
        max_duration = 0
        for trace in st:
            duration = (trace.stats.endtime - trace.stats.starttime) / 3600
            max_duration = max(max_duration, duration)
        
        is_valid = max_duration >= min_duration_hours
        return is_valid, max_duration, "OK" if is_valid else "CORRUPT"
    except Exception as e:
        return False, 0, f"Error: {str(e)[:30]}"

# Configuration
DATADIR = "/Volumes/GeoPhysics_49/users-data/montalca/DATA"
YEAR = 2025
JDAY = 100

# Scan all stations
yearpath = Path(DATADIR) / str(YEAR)
jday_str = f'{int(JDAY):03d}'

corrupted_files = []
total_files = 0
valid_files = 0

for label in ['MORIA', 'DPRI', 'GEONET']:
    label_path = yearpath / label
    if label_path.exists():
        # Find all MSEED files for this day
        pattern = str(label_path / f'*/*{YEAR}.{jday_str}')
        files = sorted(glob(pattern))
        
        for file_path in files:
            total_files += 1
            is_valid, duration, status = check_mseed_integrity(file_path)
            
            if not is_valid:
                corrupted_files.append({
                    'file': Path(file_path).name,
                    'path': file_path,
                    'duration_hours': duration,
                    'status': status
                })
            else:
                valid_files += 1

# Print report
print(f"\n{'='*70}")
print(f"MSEED Data Integrity Report - Day {JDAY}, Year {YEAR}")
print(f"{'='*70}\n")

print(f"Files scanned: {total_files}")
print(f"Valid files:   {valid_files}")
print(f"Corrupted:     {len(corrupted_files)}\n")

if corrupted_files:
    print(f"{'CORRUPTED/INCOMPLETE FILES':^70}\n")
    print(f"{'Filename':<40} {'Duration':<15} {'Status'}")
    print(f"{'-'*70}")
    for item in corrupted_files:
        duration_str = f"{item['duration_hours']:.2f}h"
        print(f"{item['file']:<40} {duration_str:<15} {item['status']}")
        
        # Extract station name
        parts = item['file'].split('.')
        if len(parts) >= 2:
            station = parts[0]
            print(f"  → Station {station} has incomplete data for this day")
else:
    print("✓ All files have good temporal coverage!")

print(f"\n{'='*70}\n")
