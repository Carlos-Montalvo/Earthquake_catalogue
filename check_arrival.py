
from obspy import read_events
import glob

xml_files = glob.glob("/Volumes/GeoPhysics_49/users-data/montalca/CATALOGS/NLL/**/*.xml", recursive=True)

n_total = 0
n_missing = 0
missing_by_file = {}

for xf in xml_files:
    cat = read_events(xf)
    for event in cat:
        origin = event.preferred_origin()
        arrival_lookup = {str(a.pick_id): a for a in origin.arrivals}
        for pick in event.picks:
            n_total += 1
            arrival = arrival_lookup.get(str(pick.resource_id))
            if arrival is None or arrival.azimuth is None or arrival.takeoff_angle is None:
                n_missing += 1
                missing_by_file[xf] = missing_by_file.get(xf, 0) + 1

print(f"Total: {n_total}  Missing: {n_missing}  ({100*n_missing/n_total:.1f}%)")
# Archivos con más picks faltantes, para inspeccionar patrones
for xf, n in sorted(missing_by_file.items(), key=lambda x: -x[1])[:10]:
    print(f"  {xf}: {n} picks faltantes")

def check_rqual_distribution(hyp_pattern, label):
    rqual_counts = {}
    total = 0
    for hf in glob.glob(hyp_pattern):
        in_phase = False
        with open(hf) as fh:
            for line in fh:
                if line.startswith('PHASE ID'):
                    in_phase = True
                    continue
                if line.startswith('END_PHASE'):
                    in_phase = False
                    continue
                if in_phase and line.strip():
                    parts = line.split()
                    if len(parts) >= 26:
                        rq = parts[25]  # RQual
                        rqual_counts[rq] = rqual_counts.get(rq, 0) + 1
                        total += 1
    print(f"\n{label} (total picks: {total})")
    for rq, n in sorted(rqual_counts.items(), key=lambda x: -x[1]):
        print(f"  RQual={rq}: {n} ({100*n/total:.1f}%)")

check_rqual_distribution("/Volumes/GeoPhysics_49/users-data/montalca/NLL/OUT_*/DATA_2024_272.*.grid0.loc.hyp",
        "2024_272 (mal)")

check_rqual_distribution("/Volumes/GeoPhysics_49/users-data/montalca/NLL/OUT_JAN25_SEP25/DATA_2025_257.*.grid0.loc.hyp",
        "2025_257 (bien)")

check_rqual_distribution("/Volumes/GeoPhysics_49/users-data/montalca/NLL/OUT_JAN25_SEP25/DATA_2025_129.*.grid0.loc.hyp",
    "2025_129")

n_arrival_none = 0
n_azimuth_none = 0
n_takeoff_none = 0
n_total = 0

for hf in glob.glob("/Volumes/GeoPhysics_49/users-data/montalca/NLL/OUT_*/DATA_2024_272.*.grid0.loc.hyp"):
    try:
        cat = read_events(hf)
    except Exception:
        continue
    for event in cat:
        origin = event.preferred_origin()
        arrival_lookup = {str(a.pick_id): a for a in origin.arrivals}
        for pick in event.picks:
            n_total += 1
            arrival = arrival_lookup.get(str(pick.resource_id))
            if arrival is None:
                n_arrival_none += 1
                continue
            if arrival.azimuth is None:
                n_azimuth_none += 1
            if arrival.takeoff_angle is None:
                n_takeoff_none += 1
print(f"Total: {n_total}")
print(f"Arrival is None (nunca se creó): {n_arrival_none}")
print(f"Arrival existe, azimuth None: {n_azimuth_none}")
print(f"Arrival existe, takeoff_angle None: {n_takeoff_none}")
