import datetime
from typing import Union
from os.path import join
from pathlib import Path
from pandas import DataFrame,date_range,read_csv,read_excel,concat
import glob
from os.path import basename


### FUNCTIONS ###
# Time period to process
def dates(s_date,e_date):
    year,sm,sd = s_date.split('-')
    yy,em,ed = e_date.split('-')
    date_format = '%Y-%m-%d'
    tperiod = date_range(start=s_date,end=e_date).strftime(date_format)
    jperiod = date_range(start=s_date,end=e_date).strftime('%j')
    time_period = ((tperiod,jperiod))
    return time_period,year

# Read assigments and event information from CSV files
def merge_events_and_assignments(events_path,assignments_path,time_period,year):
    # Directorios de iteraciones
    event_iterations = join(events_path, year, 'iterations')
    assignment_iterations = join(assignments_path, year, 'iterations')
    for jday in time_period[1]:
        # Buscar todos los archivos de eventos y asociaciones para el día
        event_files = sorted(glob.glob(join(event_iterations, f"{year}_{jday}_*.csv")))
        assignment_files = sorted(glob.glob(join(assignment_iterations, f"{year}_{jday}_*.csv")))
        if not event_files or not assignment_files:
            continue

        # Leer y concatenar todos los eventos
        events_list = [read_csv(f) for f in event_files]
        events_df = concat(events_list, ignore_index=True)

        # Asignar id global único (sin ordenar)
        events_df["global_idx"] = range(len(events_df))

        # Crear mapeo de idx local a global
        idx_map = {}
        local_counter = 0
        for f in event_files:
            df = read_csv(f)
            for i in range(len(df)):
                idx_map[(basename(f), df.iloc[i]["idx"])] = local_counter
                local_counter += 1

        # Leer y concatenar todas las asociaciones
        assignments_list = []
        for f in assignment_files:
            df = read_csv(f)
            base = basename(f)
            if len(df) > 0:
                df["event_idx"] = df["event_idx"].apply(lambda x: idx_map.get((base.replace('assignments','events').replace('ASSIGNMENTS','EVENTS'), x), -1))
            assignments_list.append(df)
        assignments_df = concat(assignments_list, ignore_index=True)

        # Guardar archivos finales
        out_event_file = join(event_yearpath, f"DATA_{year}_EVENTS_{jday}_iter.csv")
        out_assign_file = join(assignment_yearpath, f"DATA_{year}_ASSIGNMENTS_{jday}_iter.csv")

        # Reindexar eventos: idx = global_idx (sin ordenar)
        events_df["idx"] = events_df["global_idx"]
        cols_event = [c for c in ["idx","time","x","y","z","picks","latitude","longitude","depth"] if c in events_df.columns]
        events_df = events_df[cols_event]

        events_df.to_csv(out_event_file, index=False)
        assignments_df.to_csv(out_assign_file, index=False)

# Order events by time and reassign idx

def order_events_by_time(events_path, assignments_path, time_period, year):
    for jday in time_period[1]:
        events_file = join(events_path, year, f"DATA_{year}_EVENTS_{jday}_iter.csv")
        assignments_file = join(assignments_path, year, f"DATA_{year}_ASSIGNMENTS_{jday}_iter.csv")
        try:
            events_df = read_csv(events_file)
            assignments_df = read_csv(assignments_file)
        except Exception as e:
            print(f"No se pudo leer {events_file} o {assignments_file}: {e}")
            continue

        # Ordenar eventos por tiempo y reasignar idx
        events_df = events_df.sort_values("time").reset_index(drop=True)
        events_df["idx"] = events_df.index

        # Crear mapeo de id viejo a nuevo
        if "global_idx" in events_df.columns:
            old_to_new = dict(zip(events_df["global_idx"], events_df["idx"]))
        elif "event_idx" in events_df.columns:
            old_to_new = dict(zip(events_df["event_idx"], events_df["idx"]))
        else:
            old_to_new = dict(zip(events_df["idx"], events_df["idx"]))

        # Actualizar event_idx en assignments
        assignments_df["event_idx"] = assignments_df["event_idx"].map(old_to_new)
        assignments_df = assignments_df.sort_values(["event_idx", "time"]).reset_index(drop=True)

        events_df.to_csv(events_file, index=False)
        assignments_df.to_csv(assignments_file, index=False)
    

# Save assignments to NonLinLoc format (modified version of the pyocto.associator.to_nonlinloc 17/06/2025)
def assignments_to_nonlinloc(assignments: DataFrame, path: Union[str, Path]):
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

# ------------------------------------------------------------------------------
# MAIN CODE
# ------------------------------------------------------------------------------
codestart = datetime.datetime.now()

### DIRECTORIES & FILES ###
basedir = r'/Volumes/GeoPhysics_49/users-data/montalca'
catalogs = join(basedir,'CATALOGS')
assignments_path = join(catalogs,'PYOCTO/ASSIGNMENTS')
events_path = join(catalogs,'PYOCTO/EVENTS')
nll_path = join(basedir,'NLL/PYOCTO_ASSIGNMENTS')

# GeoNet station information for channel type
stafile = read_excel(join(basedir, 'STATIONS/STATIONS.xlsx'), sheet_name='GEONET', header=0,
                    usecols=[2, 6], names=['code', 'type'])

print('')
# Time period
print('Time period has to be from the same year')
s_date = input('Start date (format yyyy-mm-dd): ')
e_date = input('End date (format yyyy-mm-dd): ')
time_period,year = dates(s_date,e_date)

event_yearpath = join(events_path, year)
assignment_yearpath = join(assignments_path, year)

# Merge events and assignments, assign global ids, and save final CSVs
merge_events_and_assignments(events_path,assignments_path,time_period,year)

# # Order events by time and reassign idx
# order_events_by_time(events_path,assignments_path,time_period,year) # Doesn't work

# Convert assignments to NonLinLoc format
for jday in time_period[1]:
    assignments_df = read_csv(join(assignment_yearpath, f"DATA_{year}_ASSIGNMENTS_{jday}_iter.csv"))
    
    if len(assignments_df) > 0:
        nll_file = join(nll_path, f'DATA_{year}_{jday}.obs')
        assignments_to_nonlinloc(assignments_df, nll_file)

codeend = datetime.datetime.now()
print(f'Code duration: {codeend - codestart}')