import math, numpy as np, pandas as pd
from pyulog import ULog

def safe_get(ulog: ULog, name: str):
    """Return .data for the first dataset with this name, or None."""
    try:
        return ulog.get_dataset(name).data
    except (KeyError, IndexError):
        return None

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
    t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy)/(dx*dx + dy*dy)))
    cx, cy = ax + t*dx, ay + t*dy
    return math.hypot(px - cx, py - cy)

def kpis_from_log(path: str) -> dict:
    ulog = ULog(path)

    pos = safe_get(ulog, 'vehicle_local_position')
    if pos is None:
        raise ValueError("Log has no vehicle_local_position topic")

    t = pos['timestamp'] / 1e6
    x, y, z = pos['x'], pos['y'], pos['z']
    vx, vy, vz = pos['vx'], pos['vy'], pos['vz']

    landing_offset = math.hypot(x[-1]-x[0], y[-1]-y[0])
    cte_vals = [min(_cte((xi, yi), *seg) for seg in SEGMENTS)
                for xi, yi in zip(x, y)]
    cte_rms, cte_max = (math.sqrt(np.mean(np.square(cte_vals))),
                        float(np.max(cte_vals)))

    alt = -z
    def _rms(t1, t2, tgt):
        m = (t > t1) & (t < t2)
        return float('nan') if not np.any(m) else math.sqrt(np.mean((alt[m]-tgt)**2))
    alt_rms10 = _rms(5, 8, 10)
    alt_rms20 = _rms(13, 18, 20)
    alt_rms15 = _rms(23, 28, 15)

    vsp = safe_get(ulog, 'vehicle_local_position_setpoint')
    if vsp is not None:
        n = min(len(vx), len(vsp['vx']))
        vel_xy_rms = math.sqrt(np.mean((np.hypot(vx[:n], vy[:n])
                                        - np.hypot(vsp['vx'][:n], vsp['vy'][:n]))**2))
        vel_z_rms  = math.sqrt(np.mean((vz[:n] - vsp['vz'][:n])**2))
    else:
        vel_xy_rms = vel_z_rms = float('nan')

    att = safe_get(ulog, 'vehicle_attitude')
    if att is not None:
        q0,q1,q2,q3 = att['q[0]'], att['q[1]'], att['q[2]'], att['q[3]']
        roll  = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
        pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
        max_tilt_deg = math.degrees(np.max(np.abs([roll, pitch])))
    else:
        max_tilt_deg = float('nan')

    ctl = safe_get(ulog, 'actuator_controls_0')
    if ctl is not None:
        ctrl = np.vstack([ctl[f'control[{i}]'] for i in range(4)]).T
        ctrl_sat_pct = np.mean(np.any(np.abs(ctrl) > 0.98, axis=1))*100
    else:
        ctrl_sat_pct = float('nan')

    wind = safe_get(ulog, 'vehicle_wind_estimate')
    if wind is not None:
        wspd = np.sqrt(wind['windspeed_north']**2 + wind['windspeed_east']**2)
        wind_mean, wind_std = float(np.mean(wspd)), float(np.std(wspd))
    else:
        wind_mean = wind_std = float('nan')

    innov = safe_get(ulog, 'ekf2_innovations')
    if innov is not None:
        pos_innov = np.sqrt(innov['vel_pos_innov[0]']**2 + innov['vel_pos_innov[1]']**2)
        innov_rms = math.sqrt(np.mean(pos_innov**2))
    else:
        innov_rms = float('nan')

    flight_time = t[-1] - t[0]
    batt = safe_get(ulog, 'battery_status')
    if batt is not None and len(batt['remaining']) > 1:
        batt_drop = abs(batt['remaining'][-1] - batt['remaining'][0]) / 10
    else:
        batt_drop = float('nan')

    return {
        "file":              path,
        "landing_offset_m":  round(landing_offset,2),
        "cte_rms_m":         round(cte_rms,2),
        "cte_max_m":         round(cte_max,2),
        "alt_rms10_m":       round(alt_rms10,2),
        "alt_rms20_m":       round(alt_rms20,2),
        "alt_rms15_m":       round(alt_rms15,2),
        "vel_xy_rms_mps":    round(vel_xy_rms,2),
        "vel_z_rms_mps":     round(vel_z_rms,2),
        "max_tilt_deg":      round(max_tilt_deg,1),
        "ctrl_sat_pct":      round(ctrl_sat_pct,1),
        "wind_mean_mps":     round(wind_mean,2),
        "wind_std_mps":      round(wind_std,2),
        "innov_rms":         round(innov_rms,3),
        "flight_time_s":     int(flight_time),
        "batt_drop_pct":     round(batt_drop,1),
    }
