# Explicación Detallada: 12_MAGNITUDE_INVERSION.py

## ¿QUÉ HACE EL SCRIPT?

Este script **calibra las magnitudes locales (ML)** de tu catálogo local comparándolo con magnitudes conocidas de referencia (como las de GeoNet). Usa un método de **inversión de mínimos cuadrados** para encontrar dos cosas:

1. **γ (gamma)**: Parámetro de atenuación anelástica (cómo disminuye la amplitud con la distancia)
2. **Correcciones de estación (S)**: Ajustes específicos para cada estación

## LA FÓRMULA MATEMÁTICA

El script resuelve esta ecuación de magnitud:

$$M_L = \log_{10}(a) + \alpha \log_{10}(\Delta) + 0.4343 \cdot \gamma \cdot \Delta + S$$

Donde:
- **a** = amplitud en mm en un sismómetro Wood-Anderson
- **α** = parámetro geométrico (~1 para atenuación esférica)
- **Δ** = distancia hipocentral (km)
- **γ** = parámetro de atenuación anelástica (lo que el script busca)
- **S** = corrección de estación (específica para cada estación)

## FLUJO DEL CÓDIGO PASO A PASO

### PASO 1: Funciones auxiliares (líneas 33-117)

```
_get_origin_attrib()       → Obtiene datos del origen (tiempo, profundidad, lat/lon)
_get_magnitude_attrib()    → Obtiene magnitud específica
_get_arrival_for_amplitude() → CRÍTICO: Encuentra la "llegada de onda" para cada amplitud
_get_amplitude_value()     → Convierte amplitud a metros
```

**PROBLEMA: "No arrival found for KRHZ, skipping"**
→ Esta línea 80 significa que la estación KRHZ tiene mediciones de amplitud PERO 
   NO tiene una "arrival" registrada en los datos. Una "arrival" es un registro de 
   cuándo llegó la onda a esa estación, y más importante, la DISTANCIA calculada.

Sin la distancia, no se puede aplicar la corrección de atenuación, así que esa 
medición se descarta.

---

### PASO 2: Resumen de catálogos (líneas 119-160)

```python
summarize_catalog()       → Extrae tiempo, lat/lon, profundidad, magnitud
summarize_amplitudes()    → Extrae todas las mediciones de amplitud
```

**Que pasa aquí:**
- Para CADA evento en tu catálogo
- Para CADA amplitud en ese evento
- Llama a `_get_arrival_for_amplitude()` 
  - Si NO encuentra arrival → SALTA (prints "No arrival found for...")
  - Si encuentra arrival → Extrae: distancia epicentral, estación, amplitud, período

**La función calcula magnitud de referencia (Seisan):**
$$M_{seisan} = \log_{10}(a \times 1000) + \log_{10}(\Delta) + 0.0067 \times 0.4343 \times a \times 1000$$

---

### PASO 3: Matching de eventos (líneas 233-313)

```python
find_matching_events(catalog_1, catalog_2)
```

**Objetivo:** Encontrar qué eventos en tu catálogo corresponden a qué eventos en GeoNet.

**Criterios de match:**
- Diferencia de tiempo < 5 segundos (default, configurable)
- Diferencia epicentral < 20 km (configurable)
- Diferencia de profundidad < 40 km (configurable)

**RESULTADO:** Diccionario: {evento_GeoNet → evento_tuyo}

---

### PASO 4: Construcción de matrices Y y X (líneas 639-690)

**MATRIZ Y (vector respuesta):**
- Contiene lo que queremos predecir
- Primeras n_observations filas: logaritmo de amplitudes + término geométrico
- Últimas n_constraining_events filas: magnitudes de referencia (GeoNet)

$$Y_{top} = \log_{10}(a \times 1000) + \alpha \log_{10}(\Delta)$$
$$Y_{bottom} = M_{GeoNet}$$

**MATRIZ X (matriz de diseño):**
- Columnas 0 a n_events-1: Variables binarias (¿pertenece a este evento?)
- Columna n_events: **γ × distancia** (lo que queremos resolver)
- Columnas n_events+1 en adelante: Correcciones de estación (S)

---

### PASO 5: SOLUCIÓN POR MÍNIMOS CUADRADOS (líneas 692-722)

```python
conditionX = X.T @ X
conditionY = X.T @ Y.T
params = linalg.lstsq(u, linalg.lstsq(l, conditionY)[0])[0]
```

Resuelve el sistema:
$$(X^T X) \mathbf{p} = X^T Y$$

Donde **p** = [magnitudes nuevas, γ, correcciones de estación]

**AQUÍ OCURRE TU ERROR:**
```
assert(np.allclose(condition_inv @ conditionX, _id, atol=1e-3))
AssertionError
```

La matriz X^T × X no es invertible (o muy mal condicionada).

---

## ¿POR QUÉ OBTIENES EL ERROR?

### CAUSA 1: No hay suficientes "arrivals" 

"No arrival found for KRHZ, skipping" significa:
- Tienes amplitudes de KRHZ
- PERO no hay llegadas de onda con distancia calculada
- Por lo tanto se descartan TODAS las mediciones de esa estación

**RESULTADO:** Muy pocas observaciones totales en tu matriz.

### CAUSA 2: Muy pocos eventos coinciden

Si solo 2-3 eventos coinciden entre tu catálogo y GeoNet (de los requeridos ~20):
- n_constraining_events es muy pequeño
- Las ecuaciones del sistema son insuficientes
- No hay datos para resolver γ únicamente

### CAUSA 3: Matriz X singular

Si todas las estaciones tienen el mismo problema → matriz X no tiene rango completo

**Ejemplo problema:**
- Solo tienes 5 eventos con datos
- Pero tienes 10 estaciones
- Matriz X: 5 filas × 15 columnas
- 5 < 15 → Sistema subdeterminado

---

## SOLUCIONES

### 1. VERIFICAR LOS DATOS

```bash
# ¿Cuántos eventos tienes realmente?
python3 << 'EOF'
from obspy import read_events

# Tu catálogo
cat1 = read_events('JAN24_SEP25.xml')
print(f"Tu catálogo: {len(cat1)} eventos")
print(f"Con amplitudes: {sum(len(e.amplitudes) for e in cat1)}")

# Catálogo GeoNet
cat2 = read_events('GeoNet_CMT_solutions.xml')
print(f"\nGeoNet: {len(cat2)} eventos")
print(f"Con magnitudes Mw: {sum(1 for e in cat2 if any(m.magnitude_type=='Mw' for m in e.magnitudes))}")

# Arrivals
print(f"\nEventos con arrivals: {sum(1 for e in cat1 if e.preferred_origin().arrivals)}")
print(f"Arrivals totales: {sum(len(e.preferred_origin().arrivals) for e in cat1 if e.preferred_origin().arrivals)}")
EOF
```

### 2. VERIFICAR ARRIVALS EN TU CATÁLOGO

```bash
python3 << 'EOF'
from obspy import read_events

cat = read_events('JAN24_SEP25.xml')
ev = cat[0]
ori = ev.preferred_origin() or ev.origins[-1]

print(f"Evento: {ev.resource_id}")
print(f"Amplitudes: {len(ev.amplitudes)}")
print(f"Arrivals con distancia:")
for arr in ori.arrivals:
    if arr.distance:
        print(f"  {arr.pick_id.get_referred_object().waveform_id.station_code}: {arr.distance:.2f}°")
    else:
        print(f"  {arr.pick_id.get_referred_object().waveform_id.station_code}: SIN DISTANCIA")
EOF
```

### 3. USAR PARÁMETROS MÁS RELAJADOS

Si tu catálogo es pequeño/cercano a GeoNet:

```bash
python 12_MAGNITUDE_INVERSION.py \
  -i JAN24_SEP25.xml \
  -c GeoNet_CMT_solutions.xml \
  -o JAN24_SEP25_MAGNITUDES.xml
```

El script debería aceptar parámetros de línea de comandos para las tolerancias. 
Si no, edita la función magnitude_inversion() con:
- time_difference=15 (en lugar de 5)
- epicentral_difference=50 (en lugar de 20)
- depth_difference=100 (en lugar de 40)

### 4. ASEGURAR QUE TIENES SUFICIENTES OBSERVACIONES

Necesitas:
- **Mínimo ~20 eventos que coincidan** entre catálogos
- **Mínimo ~200-300 observaciones de amplitud** totales
- **Mínimo 5-10 estaciones** diferentes con datos

Si tu catálogo tiene <100 amplitudes totales, esta inversión **no funcionará**.

---

## RESUMEN DEL PROBLEMA EN TU CASO

1. **Estaciones sin arrivals:** KRHZ, MSWZ, DVHZ, MRZ, BHHZ, HOWZ no tienen arrivals con distancia calculada
2. **Datos insuficientes:** Esto reduce drásticamente las observaciones disponibles
3. **Matriz singular:** Con pocas observaciones y muchas incógnitas, la matriz X^T X no es invertible
4. **AssertionError:** La verificación de calidad falla porque la solución no es confiable

---

## ACCIONES RECOMENDADAS

1. **Verifica tus datos de entrada** (especialmente los arrivals)
2. **Usa un catálogo GeoNet más grande** (más eventos para matching)
3. **Considera usar -C GEONET** para descargar automáticamente eventos de GeoNet cercanos
4. **Aumenta las tolerancias** de matching (tiempo, distancia, profundidad)
5. **Si todo falla:** Tu catálogo podría ser demasiado pequeño/diferente para esta inversión
