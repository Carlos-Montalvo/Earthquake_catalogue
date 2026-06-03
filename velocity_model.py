from obspy.taup.taup_create import build_taup_model

# Escribir modelo en formato .nd
nd_content = """0.0  4.400  2.549  2.178
1.0  4.400  2.549  2.178
1.0  4.786  2.789  2.301
3.0  4.786  2.789  2.301
3.0  5.119  2.993  2.408
5.0  5.119  2.993  2.408
5.0  5.567  3.263  2.551
8.0  5.567  3.263  2.551
8.0  6.055  3.521  2.708
15.0  6.055  3.521  2.708
15.0  6.592  3.795  2.879
23.0  6.592  3.795  2.879
23.0  7.232  4.138  3.084
30.0  7.232  4.138  3.084
30.0  7.549  4.327  3.186
34.0  7.549  4.327  3.186
34.0  7.871  4.522  3.289
38.0  7.871  4.522  3.289
38.0  8.035  4.616  3.341
42.0  8.035  4.616  3.341
42.0  8.245  4.753  3.408
48.0  8.245  4.753  3.408
48.0  8.340  4.810  3.439
55.0  8.340  4.810  3.439
55.0  8.440  4.877  3.471
65.0  8.440  4.877  3.471
65.0  8.405  4.861  3.459
85.0  8.405  4.861  3.459
85.0  8.384  4.847  3.453
105.0  8.384  4.847  3.453
105.0  8.384  4.848  3.453
130.0  8.384  4.848  3.453
130.0  8.370  4.835  3.448
155.0  8.370  4.835  3.448
155.0  8.434  4.869  3.469
185.0  8.434  4.869  3.469
185.0  8.562  4.939  3.510
225.0  8.562  4.939  3.510
225.0  8.670  4.999  3.544
275.0  8.670  4.999  3.544
275.0  8.890  5.137  3.615
370.0  8.890  5.137  3.615
mantle
370.0  10.200  5.900  4.034
620.0  10.200  5.900  4.034
620.0  10.600  6.130  4.162
750.0  10.600  6.130  4.162"""

# Guardar archivo .nd
nd_file = r'/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.nd'
with open(nd_file, 'w') as f:
    f.write(nd_content)

# Compilar modelo para TauPy
build_taup_model(nd_file, output_folder=r'/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL')
print('Modelo compilado: transition_zone_vmodel.npz')

# Verificar
from obspy.taup import TauPyModel
model = TauPyModel(r'/Volumes/GeoPhysics_49/users-data/montalca/VEL_MODEL/transition_zone_vmodel.npz')
arrivals = model.get_travel_times(source_depth_in_km=10,
                                   distance_in_degree=0.5,
                                   phase_list=['P', 'S', 'Pg', 'Sg'])
for arr in arrivals:
    print(f'{arr.name}: {arr.time:.2f}s')