import math, numpy as np
from pyulog import ULog

def safe_get(ulog: ULog, name: str):
    try:
        return ulog.get_dataset(name).data
    except (KeyError, IndexError):
        return None

# geometry for 20-m square (unchanged)
SIDE = 20.0
SEGMENTS = [((0, 0), ( SIDE, 0)),
            ((SIDE, 0), ( SIDE, SIDE)),
            ((SIDE, SIDE), (0,  SIDE)),
            ((0,  SIDE), (0, 0))]

def _cte(pt, a, b):
    ax, ay = a; bx, by = b; px, py = pt
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
    cx, cy = ax + t*dx, ay + t*dy
    return math.hypot(px - cx, py - cy)

def kpis_from_log(path: str) -> dict:
    ulog = ULog(path)
    pos  = safe_get(ulog, 'vehicle_local_position')
    if pos is None:
        raise ValueError("vehicle_local_position topic missing")

    # local positions (NED metres)
    t_sec = pos['timestamp'] / 1e6
    x, y, z = pos['x'], pos['y'], pos['z']

   
    init_x, init_y = float(x[0]), float(y[0])
    final_x, final_y = float(x[-1]), float(y[-1])
    flight_time = float(t_sec[-1] - t_sec[0])
   
    # landing offset
    landing_offset = math.hypot(final_x - init_x, final_y - init_y)
    cte_vals = [min(_cte((xi, yi), *seg) for seg in SEGMENTS)
                for xi, yi in zip(x, y)]
    cte_rms  = math.sqrt(np.mean(np.square(cte_vals)))
    cte_max  = float(np.max(cte_vals))

    # altitude RMS plateaus
    alt = -z
    def _rms(t1, t2, target):
        m = (t_sec > t1) & (t_sec < t2)
        return float('nan') if not np.any(m) else \
               math.sqrt(np.mean((alt[m] - target) ** 2))
    alt_rms10 = _rms(5,  8, 10)
    alt_rms20 = _rms(13,18, 20)
    alt_rms15 = _rms(23,28, 15)

    wind = safe_get(ulog, 'vehicle_wind_estimate')
    wind_mean = wind_std = float('nan')
    if wind is not None:
        wspd = np.sqrt(wind['windspeed_north']**2 + wind['windspeed_east']**2)
        wind_mean, wind_std = float(np.mean(wspd)), float(np.std(wspd))

    return {
        "file":              path,
        "init_x_m":          round(init_x, 3),
        "init_y_m":          round(init_y, 3),
        "final_x_m":         round(final_x,3),
        "final_y_m":         round(final_y,3),
        "flight_time_s":     int(flight_time),
        "landing_offset_m":  round(landing_offset,2),
        "cte_rms_m":         round(cte_rms,2),
        "cte_max_m":         round(cte_max,2),
        "alt_rms10_m":       round(alt_rms10,2),
        "alt_rms20_m":       round(alt_rms20,2),
        "alt_rms15_m":       round(alt_rms15,2),
        "wind_mean_mps":     round(wind_mean,2),
        "wind_std_mps":      round(wind_std,2),
    }

