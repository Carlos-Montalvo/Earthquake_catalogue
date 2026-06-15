import pandas as pd
import numpy as np
from obspy.taup.taup_create import build_taup_model
from obspy.taup import TauPyModel

# ── Rutas ────────────────────────────────────────────────────────────────────
XLSX_FILE  = '/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/VELOCITY_MODEL.xlsx'
OUTPUT_DIR = '/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL'

# ── Parámetros ────────────────────────────────────────────────────────────────
USE_OFFSET   = True   # True: desplaza profundidades negativas | False: las ignora
MAX_INTERVAL = 50.0   # km — intervalo máximo permitido entre capas; se interpola si se excede

# Filtro espacial — bounding box de la región de estudio (Marlborough)
LAT_MIN = -42.5
LAT_MAX = -40.5
LON_MIN = 173.0
LON_MAX = 175.0

# ── 1. Leer xlsx ──────────────────────────────────────────────────────────────
print("Leyendo archivo xlsx...")
df = pd.read_excel(XLSX_FILE, sheet_name=0)
df.columns = [c.strip() for c in df.columns]
print(f"  Total de puntos: {len(df):,}")

# ── 2. Filtrar por región ─────────────────────────────────────────────────────
mask = (
    (df['Latitude']  >= LAT_MIN) & (df['Latitude']  <= LAT_MAX) &
    (df['Longitude'] >= LON_MIN) & (df['Longitude'] <= LON_MAX)
)
df_region = df[mask].copy()
print(f"\nFiltro espacial: lat [{LAT_MIN}, {LAT_MAX}], lon [{LON_MIN}, {LON_MAX}]")
print(f"  Puntos dentro de la región: {len(df_region):,}")

if len(df_region) == 0:
    raise ValueError("No hay puntos en la región especificada. Revisa LAT_MIN/MAX y LON_MIN/MAX.")

# ── 3. Promediar por profundidad ──────────────────────────────────────────────
df_avg = (df_region
          .groupby('Depth(km_BSL)')[['Vp', 'Vs', 'Density']]
          .mean()
          .reset_index()
          .rename(columns={'Depth(km_BSL)': 'depth', 'Vp': 'vp', 'Vs': 'vs', 'Density': 'density'}))
df_avg = df_avg.sort_values('depth').reset_index(drop=True)

print(f"\nPromedio por capa (región filtrada):")
print(f"  {'Depth(BSL)':>10}  {'Vp':>7}  {'Vs':>7}  {'Density':>8}")
for _, row in df_avg.iterrows():
    print(f"  {row['depth']:>10.0f}  {row['vp']:>7.3f}  {row['vs']:>7.3f}  {row['density']:>8.3f}")

# ── 4. Interpolar capas donde el intervalo excede MAX_INTERVAL ───────────────
def interpolate_layers(df, max_interval):
    rows = [df.iloc[0].to_dict()]
    for i in range(1, len(df)):
        prev, curr = df.iloc[i-1], df.iloc[i]
        gap = curr['depth'] - prev['depth']
        if gap > max_interval:
            n_steps = int(np.ceil(gap / max_interval))
            for step in range(1, n_steps):
                t = step / n_steps
                rows.append({
                    'depth':   prev['depth']   + t * gap,
                    'vp':      prev['vp']      + t * (curr['vp']      - prev['vp']),
                    'vs':      prev['vs']      + t * (curr['vs']       - prev['vs']),
                    'density': prev['density'] + t * (curr['density']  - prev['density']),
                })
        rows.append(curr.to_dict())
    return pd.DataFrame(rows).reset_index(drop=True)

df_avg = interpolate_layers(df_avg, MAX_INTERVAL)

n_interp = len(df_avg) - len(df_avg[df_avg['depth'].isin(
    df_region.groupby('Depth(km_BSL)').mean().reset_index()['Depth(km_BSL)']
)])
print(f"\nCapas después de interpolar intervalos >{MAX_INTERVAL:.0f} km: {len(df_avg)} ({n_interp} capas añadidas)")

# ── 5. Aplicar estrategia de profundidades ────────────────────────────────────
if USE_OFFSET:
    SURFACE_OFFSET = abs(df_avg['depth'].min())
    df_avg['depth_nd'] = df_avg['depth'] + SURFACE_OFFSET
    print(f"Modo: con offset ({SURFACE_OFFSET:.1f} km)")
else:
    SURFACE_OFFSET = 0.0
    df_avg = df_avg[df_avg['depth'] >= 0].copy()
    df_avg['depth_nd'] = df_avg['depth']
    print(f"Modo: sin offset — profundidades negativas ignoradas")

# Añadir capa en 0.0 km con los valores de la primera capa positiva
if 0.0 not in df_avg['depth_nd'].values:
    first_pos = df_avg[df_avg['depth_nd'] > 0].iloc[0].copy()
    first_pos['depth_nd'] = 0.0
    df_avg = pd.concat([pd.DataFrame([first_pos]), df_avg], ignore_index=True)
    df_avg = df_avg.sort_values('depth_nd').reset_index(drop=True)

print(f"Rango en .nd: {df_avg['depth_nd'].min():.1f} – {df_avg['depth_nd'].max():.1f} km")

# ── 6. Construir bloque cortical ──────────────────────────────────────────────
AK135_START = 750.0 + SURFACE_OFFSET

nd_lines = []
for _, row in df_avg.iterrows():
    nd_lines.append(
        f"{row['depth_nd']:.1f}  {row['vp']:.3f}  {row['vs']:.3f}  {row['density']:.3f}"
    )

if df_avg['depth_nd'].max() < AK135_START:
    last = df_avg.iloc[-1]
    nd_lines.append(
        f"{AK135_START:.1f}  {last['vp']:.3f}  {last['vs']:.3f}  {last['density']:.3f}"
    )

# ── 7. Bloque AK135 con offset opcional ──────────────────────────────────────
def build_ak135_block(offset):
    ak135_raw = {
        'mantle': [
            (800.0,10.609,6.134,3.606),(850.0,10.622,6.141,3.609),
            (900.0,10.751,6.213,3.648),(950.0,10.869,6.279,3.683),
            (1000.0,10.982,6.341,3.717),(1050.0,11.089,6.400,3.750),
            (1100.0,11.190,6.456,3.781),(1150.0,11.287,6.509,3.811),
            (1200.0,11.379,6.560,3.840),(1250.0,11.467,6.607,3.867),
            (1300.0,11.551,6.653,3.894),(1350.0,11.630,6.696,3.919),
            (1400.0,11.706,6.738,3.944),(1450.0,11.777,6.777,3.967),
            (1500.0,11.845,6.814,3.990),(1600.0,11.970,6.882,4.032),
            (1700.0,12.079,6.943,4.071),(1800.0,12.174,6.997,4.107),
            (1900.0,12.257,7.044,4.141),(2000.0,12.328,7.084,4.172),
            (2100.0,12.387,7.117,4.200),(2200.0,12.435,7.143,4.226),
            (2300.0,12.471,7.162,4.249),(2400.0,12.497,7.173,4.270),
            (2500.0,12.512,7.177,4.288),(2600.0,12.516,7.173,4.304),
            (2700.0,12.509,7.160,4.318),(2800.0,12.490,7.138,4.330),
            (2889.0,13.648,7.263,5.566),
        ],
        'outer-core': [
            (2889.0,8.065,0.000,9.914),(3000.0,8.175,0.000,10.018),
            (3100.0,8.270,0.000,10.111),(3200.0,8.360,0.000,10.200),
            (3300.0,8.444,0.000,10.285),(3400.0,8.523,0.000,10.365),
            (3500.0,8.596,0.000,10.441),(3600.0,8.664,0.000,10.513),
            (3700.0,8.726,0.000,10.580),(3800.0,8.783,0.000,10.643),
            (3900.0,8.835,0.000,10.702),(4000.0,8.882,0.000,10.756),
            (4100.0,8.923,0.000,10.806),(4200.0,8.960,0.000,10.852),
            (4300.0,8.991,0.000,10.893),(4400.0,9.017,0.000,10.930),
            (4500.0,9.038,0.000,10.963),(4600.0,9.053,0.000,10.992),
            (4700.0,9.063,0.000,11.016),(4800.0,9.067,0.000,11.036),
            (4900.0,9.066,0.000,11.051),(5000.0,9.059,0.000,11.062),
            (5100.0,9.047,0.000,11.068),(5153.5,9.040,0.000,11.071),
        ],
        'inner-core': [
            (5153.5,11.028,3.505,12.764),(5200.0,11.054,3.519,12.783),
            (5300.0,11.104,3.546,12.820),(5400.0,11.152,3.573,12.856),
            (5500.0,11.197,3.598,12.890),(5600.0,11.239,3.622,12.922),
            (5700.0,11.278,3.644,12.953),(5800.0,11.314,3.664,12.982),
            (5900.0,11.346,3.682,13.009),(6000.0,11.375,3.698,13.034),
            (6100.0,11.400,3.712,13.057),(6200.0,11.421,3.723,13.077),
            (6300.0,11.438,3.731,13.094),(6371.0,11.262,3.668,13.088),
        ]
    }
    lines = []
    for section, rows in ak135_raw.items():
        lines.append(section)
        for r in rows:
            depth_shifted = r[0] + offset
            if r[0] == 6371.0:
                depth_shifted = 6371.0
            lines.append(f"{depth_shifted:.1f}  {r[1]:.3f}  {r[2]:.3f}  {r[3]:.3f}")
    return '\n'.join(lines)

# ── 8. Escribir .nd ───────────────────────────────────────────────────────────
nd_content = "\n".join(nd_lines) + "\n" + build_ak135_block(SURFACE_OFFSET)

if USE_OFFSET == True:
    ND_FILE = '/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel_offset.nd'
else:
    ND_FILE = '/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.nd'

with open(ND_FILE, 'w') as f:
    f.write(nd_content)

print(f"\nArchivo .nd escrito: {ND_FILE}")

# ── 9. Compilar ───────────────────────────────────────────────────────────────
build_taup_model(ND_FILE, output_folder=OUTPUT_DIR)
print("Modelo compilado exitosamente.")

# ── 10. Verificar ─────────────────────────────────────────────────────────────
if USE_OFFSET == True:
    model = TauPyModel(f'{OUTPUT_DIR}/transition_zone_vmodel_offset.npz')
else:
    model = TauPyModel(f'{OUTPUT_DIR}/transition_zone_vmodel.npz')

event_depth_bsl = 10.0
taup_depth = event_depth_bsl + SURFACE_OFFSET

arrivals = model.get_travel_times(source_depth_in_km=taup_depth,
                                   distance_in_degree=0.5,
                                   phase_list=['P', 'S', 'Pg', 'Sg'])
print(f"\nTiempos de arribo de prueba (prof={event_depth_bsl:.0f} km BSL → TauPy={taup_depth:.0f} km):")
for arr in arrivals:
    print(f"  {arr.name}: {arr.time:.2f} s")