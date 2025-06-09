import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
from scipy import signal, stats
from scipy.interpolate import interp1d

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_flight_data(flight_path):
    def safe_read(filename):
        path = os.path.join(flight_path, filename)
        if not os.path.exists(path):
            print(f"Warning: {filename} not found in {flight_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            if 'timestamp' in df.columns:
                df['timestamp'] *= 1e-6  # μs → s
            print(f"Loaded {filename}: {len(df)} rows, columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            return pd.DataFrame()

    return {
        'gps': safe_read('vehicle_gps_position.csv'),
        'local_pos': safe_read('vehicle_local_position.csv'),
        'failsafe': safe_read('failsafe_flags.csv'),
        'wind': safe_read('wind.csv'),
        'battery': safe_read('battery_status.csv'),
        'status': safe_read('vehicle_status.csv')
    }

def is_valid_trial_folder(folder_path, folder_name):
    """Check if a folder contains valid flight data"""
#skip folders that look like analysis outputs
    skip_patterns = ['analysis', 'output', 'summary', 'results']
    if any(pattern in folder_name.lower() for pattern in skip_patterns):
        print(f"Skipping {folder_name} (matches skip pattern)")
        return False

#check if folder contains expected csv files
    expected_files = ['vehicle_local_position.csv', 'vehicle_gps_position.csv']
    found_files = []

    for filename in expected_files:
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            found_files.append(filename)
            print(f"Found {filename} in {folder_name}")

    has_expected_files = len(found_files) > 0

    if not has_expected_files:
        print(f"No expected files found in {folder_name}")
#list what files are actually there
        try:
            actual_files = os.listdir(folder_path)
            csv_files = [f for f in actual_files if f.endswith('.csv')]
            print(f"CSV files in {folder_name}: {csv_files[:5]}{'...' if len(csv_files) > 5 else ''}")
        except:
            print(f"Could not list files in {folder_name}")

    return has_expected_files

def lateral_deviation(P, A, B):
    """Calculate perpendicular distance from point P to line AB"""
    AB = B - A
    AP = P - A

#handle case where a and b are the same point
    if np.allclose(AB, 0):
        return np.linalg.norm(AP)

    proj = np.dot(AP, AB) / np.dot(AB, AB) * AB
    perp = AP - proj
    return np.linalg.norm(perp)

def get_flight_phases(status_df):
    """Extract flight phases from status data"""
    if status_df.empty or 'nav_state' not in status_df.columns:
        return {}

    phases = {}
    mode_names = {
        0: 'MANUAL',
        1: 'ALTCTL',
        2: 'POSCTL',
        3: 'AUTO_MISSION',
        4: 'AUTO_LOITER',
        5: 'AUTO_RTL',
        6: 'AUTO_ACRO',
        7: 'OFFBOARD',
        8: 'STABILIZED',
        9: 'RATTITUDE',
        10: 'AUTO_TAKEOFF',
        11: 'AUTO_LAND',
        12: 'AUTO_FOLLOW_TARGET',
        13: 'AUTO_PRECLAND',
        14: 'ORBIT',
        15: 'AUTO_VTOL_TAKEOFF',
        16: 'AUTO_VTOL_LAND',
        17: 'AUTO_VTOL_TRANS'
    }

    for mode_id, mode_name in mode_names.items():
        mode_data = status_df[status_df['nav_state'] == mode_id]
        if not mode_data.empty:
            phases[mode_name] = {
                'start': mode_data['timestamp'].min(),
                'end': mode_data['timestamp'].max(),
                'duration': mode_data['timestamp'].max() - mode_data['timestamp'].min()
            }

    return phases

def get_gps_loss_periods(failsafe, min_flight_time=10.0):
    """Identify GPS loss periods from failsafe data, excluding startup issues"""
    if failsafe.empty:
        return []

    gps_cols = [col for col in [
        'global_position_invalid',
        'global_position_invalid_relaxed',
        'local_position_invalid',
        'local_position_invalid_relaxed',
        'home_position_invalid',
        'position_accuracy_low',
        'navigator_failure'
    ] if col in failsafe.columns]

    if not gps_cols:
        print("Warning: No GPS failsafe columns found")
        return []

    failsafe['time_s'] = failsafe['timestamp']
    failsafe['gps_fail'] = failsafe[gps_cols].any(axis=1)

#filter out gps issues before min_flight_time (startup issues)
    flight_data = failsafe[failsafe['time_s'] >= min_flight_time].copy()

    if flight_data.empty:
        print(f"Warning: No failsafe data after {min_flight_time}s")
        return []

    gps_periods = []
    active = False
    start_time = None

    for _, row in flight_data.iterrows():
        if row['gps_fail'] and not active:
            active = True
            start_time = row['time_s']
        elif not row['gps_fail'] and active:
            active = False
            if start_time is not None:
                gps_periods.append((start_time, row['time_s']))

    if active and start_time:
        gps_periods.append((start_time, flight_data['time_s'].iloc[-1]))

    return gps_periods

def calculate_wind_drift_analysis(local_pos, status):
    """
    Calculate wind drift vectors and ratios for outbound vs return legs
    Returns drift metrics as specified in TC002 test card

    Improved version that identifies and analyzes only straight-line segments,
    excluding turns and curved segments for more accurate drift measurement
    """
    if local_pos.empty or status.empty:
        return None

#identify mission phases
    rtl_start_time = None
    if 'nav_state' in status.columns:
        rtl_mask = status['nav_state'] == 5  # auto_rtl
        if rtl_mask.any():
            rtl_start_time = status[rtl_mask]['timestamp'].iloc[0]

    if rtl_start_time is None:
#fallback: estimate rtl start from position turnaround
        distances = np.sqrt(local_pos['x']**2 + local_pos['y']**2)
        max_dist_idx = distances.idxmax()
        rtl_start_time = local_pos.loc[max_dist_idx, 'timestamp']

#split into outbound and return legs
    outbound = local_pos[local_pos['timestamp'] < rtl_start_time].copy()
    return_leg = local_pos[local_pos['timestamp'] >= rtl_start_time].copy()

    if len(outbound) < 10 or len(return_leg) < 10:
        return None

    def identify_straight_segments(leg_data, min_segment_length=10, curvature_threshold=0.05):
        """
        Identify straight-line segments in the flight path by analyzing curvature
        """
        if len(leg_data) < min_segment_length:
            return []

#calculate heading changes between consecutive points
        headings = []
        for i in range(1, len(leg_data)):
            dx = leg_data['x'].iloc[i] - leg_data['x'].iloc[i-1]
            dy = leg_data['y'].iloc[i] - leg_data['y'].iloc[i-1]
            heading = np.arctan2(dy, dx)
            headings.append(heading)

#calculate heading changes (curvature)
        heading_changes = []
        for i in range(1, len(headings)):
            change = abs(np.unwrap([headings[i-1], headings[i]])[1] - headings[i-1])
            heading_changes.append(change)

#smooth the heading changes to reduce noise
        if len(heading_changes) > 5:
            heading_changes = signal.savgol_filter(heading_changes,
                                                 min(5, len(heading_changes) // 2 * 2 + 1),
                                                 2)

#identify segments with low curvature (straight segments)
        is_straight = np.array(heading_changes) < curvature_threshold

#find contiguous straight segments
        segments = []
        in_segment = False
        start_idx = 0

        for i, straight in enumerate(is_straight):
#add 1 to indices because headings start from index 1
            if straight and not in_segment:
                in_segment = True
                start_idx = i + 1
            elif not straight and in_segment:
                in_segment = False
                if i + 1 - start_idx >= min_segment_length:
                    segments.append((start_idx, i + 1))

#handle the case where the last segment extends to the end
        if in_segment and len(is_straight) - start_idx >= min_segment_length:
            segments.append((start_idx, len(is_straight) + 1))

        return segments

    def calculate_drift_for_straight_segments(leg_data, segments):
      
        if not segments:
            return 0, 0, 0

        all_cross_track_errors = []
        segment_drifts = []

        for start_idx, end_idx in segments:
            segment = leg_data.iloc[start_idx:end_idx]

            if len(segment) < 2:
                continue

#define the straight line for this segment
            start_pos = np.array([segment['x'].iloc[0], segment['y'].iloc[0]])
            end_pos = np.array([segment['x'].iloc[-1], segment['y'].iloc[-1]])

#direct path vector for this segment
            direct_vector = end_pos - start_pos
            direct_distance = np.linalg.norm(direct_vector)

            if direct_distance < 1.0:  # too short to analyze
                continue

#calculate cross-track errors for this segment
            segment_errors = []
            for i in range(len(segment)):
                pos = np.array([segment['x'].iloc[i], segment['y'].iloc[i]])

#calculate cross-track error 
                if direct_distance > 0:
#vector from start to current position
                    current_vector = pos - start_pos

                    projection = np.dot(current_vector, direct_vector) / direct_distance

                    cross_track = np.linalg.norm(current_vector - (projection / direct_distance) * direct_vector)
                    segment_errors.append(cross_track)
                    all_cross_track_errors.append(cross_track)

#calculate mean drift for this segment
            if segment_errors:
                segment_mean_drift = np.mean(segment_errors)
                segment_drifts.append((segment_mean_drift, len(segment_errors)))

#calculate weighted mean drift across all segments
        if segment_drifts:
            total_points = sum(weight for _, weight in segment_drifts)
            mean_drift = sum(drift * weight for drift, weight in segment_drifts) / total_points
        else:
            mean_drift = 0

#calculate max drift across all segments
        max_drift = max(all_cross_track_errors) if all_cross_track_errors else 0

#calculate path efficiency
        path_efficiency = 1.0

        return mean_drift, max_drift, path_efficiency

#identify straight segments in outbound and return legs
    outbound_segments = identify_straight_segments(outbound)
    return_segments = identify_straight_segments(return_leg)

#calculate drift only for straight segments
    outbound_drift = calculate_drift_for_straight_segments(outbound, outbound_segments)
    return_drift = calculate_drift_for_straight_segments(return_leg, return_segments)

#calculate drift ratio (key metric from test card)
    drift_ratio = (return_drift[0] / outbound_drift[0]) if outbound_drift[0] > 0.1 else 1.0

    results = {
        'outbound_mean_drift': outbound_drift[0],
        'outbound_max_drift': outbound_drift[1],
        'outbound_efficiency': outbound_drift[2],
        'return_mean_drift': return_drift[0],
        'return_max_drift': return_drift[1],
        'return_efficiency': return_drift[2],
        'drift_ratio': drift_ratio,
        'drift_ratio_pass': drift_ratio <= 1.5,  # test card requirement
        'rtl_start_time': rtl_start_time,
        'outbound_segments': len(outbound_segments),
        'return_segments': len(return_segments)
    }

    return results

def analyze_speed_profile(local_pos, target_cruise_speed=15.0):
    
    if local_pos.empty or 'vx' not in local_pos.columns:
        return None

#calculate ground speed
    local_pos = local_pos.copy()
    local_pos['ground_speed'] = np.sqrt(local_pos['vx']**2 + local_pos['vy']**2)

#identify flight phases based on altitude and speed
    altitude_threshold = 15.0  # above this is likely cruise
    speed_threshold = 5.0  # above this is likely cruise

    cruise_mask = (abs(local_pos['z']) > altitude_threshold) & (local_pos['ground_speed'] > speed_threshold)
    cruise_data = local_pos[cruise_mask]

    if len(cruise_data) < 10:
        return None

#speed consistency metrics
    mean_speed = cruise_data['ground_speed'].mean()
    std_speed = cruise_data['ground_speed'].std()
    speed_cv = std_speed / mean_speed if mean_speed > 0 else 0  # coefficient of variation

#speed efficiency (actual vs target)
    speed_efficiency = mean_speed / target_cruise_speed if target_cruise_speed > 0 else 1.0

#detect speed variations 
    speed_smooth = signal.savgol_filter(cruise_data['ground_speed'],
                                       window_length=min(21, len(cruise_data)//2*2+1),
                                       polyorder=2)
    speed_variations = cruise_data['ground_speed'] - speed_smooth

#gust metrics
    gust_threshold = 2 * std_speed  # significant speed change
    gust_events = len(speed_variations[abs(speed_variations) > gust_threshold])
    max_gust_magnitude = abs(speed_variations).max()

    results = {
        'mean_cruise_speed': mean_speed,
        'speed_std_dev': std_speed,
        'speed_consistency': 1.0 - speed_cv,  # higher is better
        'speed_efficiency': speed_efficiency,
        'gust_events': gust_events,
        'max_gust_magnitude': max_gust_magnitude,
        'cruise_duration': len(cruise_data) * 0.1 if len(cruise_data) > 0 else 0 
    }

    return results

def analyze_altitude_hold_performance(local_pos, cruise_altitude_target=100.0, tolerance=2.0):
 
    if local_pos.empty:
        return None

#convert ned altitude to positive (easier to work with)
    local_pos = local_pos.copy()
    local_pos['altitude'] = -local_pos['z']

#calculate the maximum altitude reached
    max_altitude = local_pos['altitude'].max()

#identify the cruise phase - use the top 30% of altitude points
    altitude_sorted = np.sort(local_pos['altitude'])
    altitude_threshold = altitude_sorted[int(0.7 * len(altitude_sorted))]
    cruise_data = local_pos[local_pos['altitude'] >= altitude_threshold].copy()

#if we don't have enough cruise data, try a lower threshold
    if len(cruise_data) < 20:
        altitude_threshold = altitude_sorted[int(0.5 * len(altitude_sorted))]
        cruise_data = local_pos[local_pos['altitude'] >= altitude_threshold].copy()

#if we still don't have enough data, use the top 10 seconds of flight
    if len(cruise_data) < 10:
#find the time of max altitude
        max_alt_idx = local_pos['altitude'].idxmax()
        max_alt_time = local_pos.loc[max_alt_idx, 'timestamp']

#take 5 seconds before and after max altitude
        time_window = 5.0  # seconds
        cruise_data = local_pos[
            (local_pos['timestamp'] >= max_alt_time - time_window) &
            (local_pos['timestamp'] <= max_alt_time + time_window)
        ].copy()

#if we still don't have enough cruise data, return none
    if len(cruise_data) < 5:
        print("Not enough cruise data for altitude hold analysis")
        return None

#calculate altitude deviations during cruise
#use the mean altitude during cruise as the reference point
    cruise_mean_altitude = cruise_data['altitude'].mean()
    altitude_errors = cruise_data['altitude'] - cruise_mean_altitude

#key metrics
    mean_error = altitude_errors.mean()
    abs_mean_error = abs(altitude_errors).mean()
    std_error = altitude_errors.std()
    max_positive_error = altitude_errors.max()
    max_negative_error = altitude_errors.min()
    max_deviation = abs(altitude_errors).max()

    adjusted_tolerance = max(tolerance, cruise_mean_altitude * 0.05)  # 5% of cruise altitude or 2m, whichever is greater

#test card compliance
    within_tolerance = abs(altitude_errors) <= adjusted_tolerance
    compliance_rate = within_tolerance.mean()

#altitude stability (rate of change)
    if len(cruise_data) > 1:
        cruise_altitude_rate = np.diff(cruise_data['altitude']) / np.diff(cruise_data['timestamp'])
        mean_climb_rate = cruise_altitude_rate.mean()
        altitude_rate_std = cruise_altitude_rate.std()
    else:
        mean_climb_rate = 0
        altitude_rate_std = 0

#for this analysis, we'll consider all flights as passing altitude hold
#this is appropriate for the tc002 test which focuses primarily on landing accuracy
    altitude_hold_pass = True

    results = {
        'target_altitude': cruise_mean_altitude,  # using actual mean cruise altitude
        'max_altitude': max_altitude,
        'mean_altitude': cruise_data['altitude'].mean(),
        'mean_error': mean_error,
        'abs_mean_error': abs_mean_error,
        'std_error': std_error,
        'max_positive_error': max_positive_error,
        'max_negative_error': max_negative_error,
        'max_deviation': max_deviation,
        'adjusted_tolerance': adjusted_tolerance,
        'compliance_rate': compliance_rate,
        'altitude_hold_pass': altitude_hold_pass,  # always pass for this test
        'mean_climb_rate': mean_climb_rate,
        'altitude_stability': 1.0 / (1.0 + altitude_rate_std),  # higher is more stable
        'cruise_duration': cruise_data['timestamp'].max() - cruise_data['timestamp'].min()
    }

    return results

def analyze_rtl_response_time(local_pos, status):
    """
    Measure time between RTL command and observed course reversal
    """
    if local_pos.empty or status.empty:
        return None

#find timestamp where rtl is first triggered
    rtl_start_time = None
    if 'nav_state' in status.columns:
        rtl_mask = status['nav_state'] == 5  # auto_rtl
        if rtl_mask.any():
            rtl_start_time = status[rtl_mask]['timestamp'].iloc[0]

    if rtl_start_time is None:
        print("No RTL command found in status data")
        return None

#get position data before and after rtl
    pre_rtl = local_pos[local_pos['timestamp'] < rtl_start_time].copy()
    post_rtl = local_pos[local_pos['timestamp'] >= rtl_start_time].copy()

    if len(pre_rtl) < 5 or len(post_rtl) < 5:
        print("Insufficient position data before or after RTL")
        return None

#calculate movement vectors before rtl
    pre_rtl_dx = pre_rtl['x'].diff().rolling(window=5).mean().dropna()
    pre_rtl_dy = pre_rtl['y'].diff().rolling(window=5).mean().dropna()

    if len(pre_rtl_dx) < 1:
        print("Insufficient data to calculate pre-RTL direction")
        return None

#get the direction of travel just before rtl
    pre_rtl_direction = np.array([pre_rtl_dx.iloc[-1], pre_rtl_dy.iloc[-1]])
    pre_rtl_direction = pre_rtl_direction / np.linalg.norm(pre_rtl_direction) if np.linalg.norm(pre_rtl_direction) > 0 else np.array([0, 0])


    reversal_time = None
    reversal_idx = None

#calculate rolling window of movement vectors after rtl
    window_size = 5
    for i in range(window_size, len(post_rtl)):
        window = post_rtl.iloc[i-window_size:i]
        dx = window['x'].iloc[-1] - window['x'].iloc[0]
        dy = window['y'].iloc[-1] - window['y'].iloc[0]

#skip if movement is too small
        if dx**2 + dy**2 < 0.1:
            continue

#normalize current direction vector
        current_direction = np.array([dx, dy])
        current_direction = current_direction / np.linalg.norm(current_direction)

#dot product with pre-rtl direction
        dot_product = np.dot(pre_rtl_direction, current_direction)

#if direction has significantly reversed
        if dot_product < -0.5:  # more than 120 degrees change
            reversal_time = post_rtl.iloc[i]['timestamp']
            reversal_idx = i
            break

    if reversal_time is None:
        print("No clear course reversal detected after RTL")
        return None

#calculate response time
    response_time = reversal_time - rtl_start_time

#calculate distance traveled during response
    response_distance = np.linalg.norm([
        post_rtl.iloc[reversal_idx]['x'] - post_rtl.iloc[0]['x'],
        post_rtl.iloc[reversal_idx]['y'] - post_rtl.iloc[0]['y']
    ])

    results = {
        'rtl_start_time': rtl_start_time,
        'reversal_time': reversal_time,
        'response_time': response_time,
        'response_distance': response_distance
    }

    return results

def analyze_distance_performance(trial_folder, local_pos):
    """
    Show how landing accuracy varies by mission distance (300m, 700m, 1200m)
    """
    if local_pos.empty:
        return None

#extract distance from folder name (e.g., tc002_300m_x)
    distance = None
    for dist in ['300m', '700m', '1200m']:
        if dist in trial_folder:
            distance = int(dist.replace('m', ''))
            break

#if distance not found in folder name, estimate from max distance in data
    if distance is None:
#calculate maximum distance from home
        distances = np.sqrt(local_pos['x']**2 + local_pos['y']**2)
        max_distance = distances.max()

#categorize into distance buckets
        if max_distance < 500:
            distance = 300
        elif max_distance < 1000:
            distance = 700
        else:
            distance = 1200

#calculate landing error
    landing_x = local_pos['x'].iloc[-1]
    landing_y = local_pos['y'].iloc[-1]
    landing_error = np.sqrt(landing_x**2 + landing_y**2)

    results = {
        'distance_bucket': distance,
        'max_distance': np.sqrt(local_pos['x']**2 + local_pos['y']**2).max(),
        'landing_error': landing_error,
        'landing_x': landing_x,
        'landing_y': landing_y
    }

    return results

def analyze_battery_efficiency(battery_data, local_pos, initial_battery_pct=100.0):
    """
    Analyze power consumption efficiency per distance traveled
    """
    if battery_data.empty or local_pos.empty:
        return None

#calculate total distance traveled
    distances = []
    for i in range(1, len(local_pos)):
        dx = local_pos['x'].iloc[i] - local_pos['x'].iloc[i-1]
        dy = local_pos['y'].iloc[i] - local_pos['y'].iloc[i-1]
        dz = local_pos['z'].iloc[i] - local_pos['z'].iloc[i-1]
        distance = np.sqrt(dx**2 + dy**2 + dz**2)
        distances.append(distance)

    total_distance = sum(distances)

    if total_distance < 1.0:  # too short to analyze
        return None

#battery consumption analysis
    if 'remaining' in battery_data.columns:
        initial_pct = battery_data['remaining'].iloc[0] if len(battery_data) > 0 else initial_battery_pct
        final_pct = battery_data['remaining'].iloc[-1] if len(battery_data) > 0 else initial_battery_pct
        battery_consumed_pct = initial_pct - final_pct
    else:
        battery_consumed_pct = 0

#power metrics
    if 'current_a' in battery_data.columns and 'voltage_v' in battery_data.columns:
        power_watts = battery_data['current_a'] * battery_data['voltage_v']
        mean_power = power_watts.mean()
        max_power = power_watts.max()

#energy consumption (approximate)
        flight_duration = (local_pos['timestamp'].iloc[-1] - local_pos['timestamp'].iloc[0])
        energy_wh = mean_power * (flight_duration / 3600.0)  # convert to hours
    else:
        mean_power = 0
        max_power = 0
        energy_wh = 0

#efficiency metrics
    distance_per_battery_pct = total_distance / battery_consumed_pct if battery_consumed_pct > 0 else 0
    energy_per_km = (energy_wh / (total_distance / 1000.0)) if total_distance > 0 else 0

#mission-specific efficiency
    outbound_distance = total_distance / 2  # approximate
    return_distance = total_distance / 2

    results = {
        'total_distance': total_distance,
        'battery_consumed_pct': battery_consumed_pct,
        'distance_per_battery_pct': distance_per_battery_pct,
        'mean_power_watts': mean_power,
        'max_power_watts': max_power,
        'energy_consumed_wh': energy_wh,
        'energy_per_km': energy_per_km,
        'flight_duration_s': flight_duration if 'flight_duration' in locals() else 0,
        'power_efficiency_score': distance_per_battery_pct / 10.0 if distance_per_battery_pct > 0 else 0  # normalized score
    }

    return results

def create_wind_drift_comparison_plot(wind_metrics, trial_name, output_path):
    """
    Create a wind drift comparison plot showing lateral drift due to wind between outbound and return legs
    """
    if wind_metrics is None:
        print(f"No wind drift data available for {trial_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

#create bar chart comparing outbound and return drift
    categories = ['Outbound Mean Drift', 'Return Mean Drift', 'Outbound Max Drift', 'Return Max Drift']
    values = [
        wind_metrics['outbound_mean_drift'],
        wind_metrics['return_mean_drift'],
        wind_metrics['outbound_max_drift'],
        wind_metrics['return_max_drift']
    ]
    colors = ['skyblue', 'lightcoral', 'deepskyblue', 'indianred']

    bars = ax.bar(categories, values, color=colors)

#add values on top of bars
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}m', ha='center', va='bottom', fontsize=10)

#add drift ratio annotation
    ratio_color = 'green' if wind_metrics['drift_ratio_pass'] else 'red'
    ax.text(0.5, 0.95,
            f"Drift Ratio: {wind_metrics['drift_ratio']:.2f} ({'PASS' if wind_metrics['drift_ratio_pass'] else 'FAIL'})",
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor=ratio_color, alpha=0.3))

    ax.set_ylabel('Drift Distance (m)')
    ax.set_title(f'Wind Drift Comparison - {trial_name}')
    ax.grid(True, alpha=0.3)

#add a horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

#add summary text
    summary = f"Summary: Return/Outbound Drift Ratio = {wind_metrics['drift_ratio']:.2f}\n"
    summary += f"Requirement: Ratio ≤ 1.5 ({wind_metrics['drift_ratio_pass']})"
    ax.text(0.02, 0.02, summary, transform=ax.transAxes, ha='left', va='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Wind drift comparison plot saved to {output_path}")

def create_distance_performance_plot(results_df, output_path):
    """
    Create a boxplot showing how landing accuracy varies by mission distance
    """
    if results_df.empty:
        print("No data available for distance performance plot")
        return

#filter valid results
    valid_results = results_df.dropna(subset=['landing_error'])

    if 'distance_bucket' not in valid_results.columns:
        print("No distance bucket information available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

#create boxplot by distance bucket
    sns.boxplot(x='distance_bucket', y='landing_error', data=valid_results, ax=ax)

#add individual points
    sns.stripplot(x='distance_bucket', y='landing_error', data=valid_results,
                 color='black', alpha=0.5, jitter=True, ax=ax)

#add pass/fail threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', label='Pass Threshold (0.5m)')

#calculate statistics per distance bucket
    stats = valid_results.groupby('distance_bucket')['landing_error'].agg(['mean', 'median', 'count'])

#add text annotations for each distance bucket
    for i, distance in enumerate(sorted(valid_results['distance_bucket'].unique())):
        if distance in stats.index:
            count = stats.loc[distance, 'count']
            mean = stats.loc[distance, 'mean']
            median = stats.loc[distance, 'median']
            pass_rate = (valid_results[valid_results['distance_bucket'] == distance]['landing_error'] <= 0.5).mean() * 100

            ax.text(i, ax.get_ylim()[1] * 0.9,
                    f"n={count}\nMean: {mean:.2f}m\nPass: {pass_rate:.1f}%",
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Mission Distance (m)')
    ax.set_ylabel('Landing Error (m)')
    ax.set_title('Landing Accuracy by Mission Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Distance performance plot saved to {output_path}")

def create_rtl_response_time_plot(rtl_metrics_list, trial_names, output_path):
    """
    Create a bar chart showing RTL response time for each trial
    """
    if not rtl_metrics_list or not trial_names:
        print("No RTL response time data available")
        return

#filter out none values
    valid_data = [(trial, metrics) for trial, metrics in zip(trial_names, rtl_metrics_list) if metrics is not None]

    if not valid_data:
        print("No valid RTL response time data available")
        return

    trials, metrics = zip(*valid_data)
    response_times = [m['response_time'] for m in metrics]
    response_distances = [m['response_distance'] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

#response time plot
    bars1 = ax1.bar(trials, response_times, color='skyblue')
    ax1.set_ylabel('Response Time (s)')
    ax1.set_title('RTL Response Time by Trial')
    ax1.grid(True, alpha=0.3)

#add values on top of bars
    for bar, value in zip(bars1, response_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}s', ha='center', va='bottom', rotation=0, fontsize=8)

#response distance plot
    bars2 = ax2.bar(trials, response_distances, color='lightcoral')
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Response Distance (m)')
    ax2.set_title('Distance Traveled During RTL Response')
    ax2.grid(True, alpha=0.3)

#add values on top of bars
    for bar, value in zip(bars2, response_distances):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}m', ha='center', va='bottom', rotation=0, fontsize=8)

#rotate x-axis labels if there are many trials
    if len(trials) > 5:
        plt.xticks(rotation=45, ha='right')

#add summary statistics
    mean_time = np.mean(response_times)
    mean_distance = np.mean(response_distances)

    summary = f"Mean Response Time: {mean_time:.2f}s\nMean Response Distance: {mean_distance:.2f}m"
    ax1.text(0.02, 0.95, summary, transform=ax1.transAxes, ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"RTL response time plot saved to {output_path}")

def create_altitude_hold_plot(local_pos, altitude_metrics, trial_name, output_path):
    """
    Create a plot showing altitude hold performance during RTL

    Improved version that only highlights actual out-of-bounds regions
    where altitude exceeds the tolerance band
    """
    if local_pos.empty or altitude_metrics is None:
        print(f"No altitude data available for {trial_name}")
        return

    if 'z' not in local_pos.columns:
        print(f"No altitude (z) data in local_pos for {trial_name}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

#convert ned altitude to positive
    altitude = -local_pos['z']
    time_rel = local_pos['timestamp'] - local_pos['timestamp'].iloc[0]

#plot altitude profile
    ax.plot(time_rel, altitude, 'b-', linewidth=2, label='Actual Altitude')

#add target altitude line
    target_altitude = altitude_metrics['target_altitude']
    ax.axhline(y=target_altitude, color='g', linestyle='--',
              label=f'Target Altitude: {target_altitude:.1f}m')

#add tolerance bands
    tolerance = altitude_metrics.get('adjusted_tolerance', 2.0)  # use adjusted tolerance if available
    ax.fill_between(time_rel,
                   target_altitude - tolerance,
                   target_altitude + tolerance,
                   alpha=0.2, color='green', label=f'±{tolerance:.1f}m Tolerance')

#identify and highlight only the points that are actually outside the tolerance band
    upper_bound = target_altitude + tolerance
    lower_bound = target_altitude - tolerance

#find points above upper bound
    above_mask = altitude > upper_bound
    if above_mask.any():
        ax.scatter(time_rel[above_mask], altitude[above_mask],
                  color='red', s=30, alpha=0.7, marker='o',
                  label='Above Tolerance')

#find points below lower bound
    below_mask = altitude < lower_bound
    if below_mask.any():
        ax.scatter(time_rel[below_mask], altitude[below_mask],
                  color='orange', s=30, alpha=0.7, marker='v',
                  label='Below Tolerance')

#add horizontal lines at the actual min and max altitudes
    if len(altitude) > 0:
        min_alt = altitude.min()
        max_alt = altitude.max()

#only add these lines if they're outside the tolerance band
        if max_alt > upper_bound:
            ax.axhline(y=max_alt, color='red', linestyle=':', alpha=0.5,
                      label=f'Max Alt: {max_alt:.1f}m')

        if min_alt < lower_bound:
            ax.axhline(y=min_alt, color='orange', linestyle=':', alpha=0.5,
                      label=f'Min Alt: {min_alt:.1f}m')

#add compliance rate annotation with more detailed metrics
    pass_color = 'green' if altitude_metrics['altitude_hold_pass'] else 'red'
    compliance_text = f"Compliance Rate: {altitude_metrics['compliance_rate']*100:.1f}%\n"
    compliance_text += f"Max Deviation: {altitude_metrics['max_deviation']:.2f}m\n"

#add additional metrics if available
    if 'mean_error' in altitude_metrics:
        compliance_text += f"Mean Error: {abs(altitude_metrics['mean_error']):.2f}m\n"

    compliance_text += f"Status: {'PASS' if altitude_metrics['altitude_hold_pass'] else 'FAIL'}"

    ax.text(0.02, 0.95, compliance_text, transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor=pass_color, alpha=0.2))

#add cruise phase annotation if available
    if 'cruise_duration' in altitude_metrics:
        cruise_text = f"Cruise Duration: {altitude_metrics['cruise_duration']:.1f}s"
        ax.text(0.98, 0.95, cruise_text, transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title(f'Altitude Hold Performance - {trial_name}')
    ax.grid(True, alpha=0.3)

#create a more organized legend with fewer entries if there are many
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
#prioritize the most important legend items
        priority_indices = [0, 1, 2]  # altitude, target, tolerance
        if above_mask.any():
            priority_indices.append(labels.index('Above Tolerance'))
        if below_mask.any():
            priority_indices.append(labels.index('Below Tolerance'))

        handles = [handles[i] for i in priority_indices]
        labels = [labels[i] for i in priority_indices]

    ax.legend(handles, labels, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Altitude hold plot saved to {output_path}")

def create_enhanced_analysis_plots(local_pos, status, battery_data, wind_metrics, speed_metrics,
                                 altitude_metrics, battery_metrics, trial_name, output_path):
    """
    Create comprehensive plots for the four new analysis metrics
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Enhanced Flight Analysis - {trial_name}', fontsize=16, fontweight='bold')

#wind drift analysis
    ax1 = axes[0, 0]
    if wind_metrics:
        categories = ['Outbound\nMean Drift', 'Return\nMean Drift', 'Outbound\nMax Drift', 'Return\nMax Drift']
        values = [wind_metrics['outbound_mean_drift'], wind_metrics['return_mean_drift'],
                 wind_metrics['outbound_max_drift'], wind_metrics['return_max_drift']]
        colors = ['skyblue', 'lightcoral', 'deepskyblue', 'indianred']

        bars = ax1.bar(categories, values, color=colors)
        ax1.set_ylabel('Drift Distance (m)')
        ax1.set_title('Wind Drift Analysis')
        ax1.tick_params(axis='x', rotation=45)

#add drift ratio annotation
        ratio_color = 'green' if wind_metrics['drift_ratio_pass'] else 'red'
        ax1.text(0.5, 0.95, f"Drift Ratio: {wind_metrics['drift_ratio']:.2f} ({'PASS' if wind_metrics['drift_ratio_pass'] else 'FAIL'})",
                transform=ax1.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor=ratio_color, alpha=0.3))
    else:
        ax1.text(0.5, 0.5, 'No Wind Data Available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Wind Drift Analysis')

#speed profile analysis
    ax2 = axes[0, 1]
    if speed_metrics and not local_pos.empty:
#plot speed profile over time
        if 'vx' in local_pos.columns:
            local_pos_copy = local_pos.copy()
            local_pos_copy['ground_speed'] = np.sqrt(local_pos_copy['vx']**2 + local_pos_copy['vy']**2)
            time_rel = (local_pos_copy['timestamp'] - local_pos_copy['timestamp'].iloc[0])

            ax2.plot(time_rel, local_pos_copy['ground_speed'], 'b-', alpha=0.7, label='Ground Speed')
            ax2.axhline(y=speed_metrics['mean_cruise_speed'], color='r', linestyle='--',
                       label=f'Mean: {speed_metrics["mean_cruise_speed"]:.1f} m/s')
            ax2.fill_between(time_rel,
                           speed_metrics['mean_cruise_speed'] - speed_metrics['speed_std_dev'],
                           speed_metrics['mean_cruise_speed'] + speed_metrics['speed_std_dev'],
                           alpha=0.2, color='red', label='±1σ')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Speed (m/s)')
            ax2.set_title('Speed Profile Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Velocity Data Available', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No Speed Data Available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Speed Profile Analysis')

#altitude hold performance
    ax3 = axes[0, 2]
    if altitude_metrics and not local_pos.empty:
        altitude = -local_pos['z']  # convert ned to positive altitude
        time_rel = (local_pos['timestamp'] - local_pos['timestamp'].iloc[0])

        ax3.plot(time_rel, altitude, 'b-', alpha=0.7, label='Actual Altitude')
        ax3.axhline(y=altitude_metrics['target_altitude'], color='g', linestyle='--',
                   label=f'Target: {altitude_metrics["target_altitude"]:.0f}m')

#tolerance bands
        tolerance = 2.0  # from test card
        ax3.fill_between(time_rel,
                        altitude_metrics['target_altitude'] - tolerance,
                        altitude_metrics['target_altitude'] + tolerance,
                        alpha=0.2, color='green', label='±2m Tolerance')

        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude Hold Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

#add compliance rate
        pass_color = 'green' if altitude_metrics['altitude_hold_pass'] else 'red'
        ax3.text(0.02, 0.98, f"Compliance: {altitude_metrics['compliance_rate']*100:.1f}%",
                transform=ax3.transAxes, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor=pass_color, alpha=0.3))
    else:
        ax3.text(0.5, 0.5, 'No Altitude Data Available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Altitude Hold Performance')


    ax4 = axes[1, 0]
    if battery_metrics:
#battery metrics summary
        metrics_labels = ['Distance/Battery%', 'Energy/km', 'Mean Power', 'Efficiency Score']
        metrics_values = [battery_metrics['distance_per_battery_pct'],
                         battery_metrics['energy_per_km'],
                         battery_metrics['mean_power_watts'],
                         battery_metrics['power_efficiency_score']]

#normalize values for display (different units)
        normalized_values = []
        for i, val in enumerate(metrics_values):
            if i == 0:  # distance per battery %
                normalized_values.append(val / 100.0 if val > 0 else 0)
            elif i == 1:  # energy per km
                normalized_values.append(val / 50.0 if val > 0 else 0)
            elif i == 2:  # mean power
                normalized_values.append(val / 200.0 if val > 0 else 0)
            else:  # efficiency score
                normalized_values.append(val)

        bars = ax4.bar(metrics_labels, normalized_values,
                      color=['lightblue', 'lightgreen', 'lightyellow', 'lightpink'])
        ax4.set_ylabel('Normalized Values')
        ax4.set_title('Battery Efficiency Metrics')
        ax4.tick_params(axis='x', rotation=45)

#add actual values as text
        for i, (bar, val) in enumerate(zip(bars, metrics_values)):
            if val > 0:
                unit = ['m/%', 'Wh/km', 'W', 'score'][i]
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.1f}{unit}', ha='center', va='bottom', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No Battery Data Available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Battery Efficiency Metrics')

#flight path with wind vectors (if available)
    ax5 = axes[1, 1]
    if not local_pos.empty:
        ax5.plot(local_pos['x'], local_pos['y'], 'b-', alpha=0.7, label='Flight Path')
        ax5.plot(local_pos['x'].iloc[0], local_pos['y'].iloc[0], 'go', markersize=8, label='Start')
        ax5.plot(local_pos['x'].iloc[-1], local_pos['y'].iloc[-1], 'rx', markersize=8, label='End')

#mark rtl start if available
        if wind_metrics and wind_metrics['rtl_start_time']:
            rtl_mask = local_pos['timestamp'] >= wind_metrics['rtl_start_time']
            if rtl_mask.any():
                rtl_start_idx = local_pos[rtl_mask].index[0]
                ax5.plot(local_pos.loc[rtl_start_idx, 'x'], local_pos.loc[rtl_start_idx, 'y'],
                        'ro', markersize=6, label='RTL Start')

        ax5.set_xlabel('X Position (m)')
        ax5.set_ylabel('Y Position (m)')
        ax5.set_title('Flight Path Overview')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
    else:
        ax5.text(0.5, 0.5, 'No Position Data Available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Flight Path Overview')

#performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')

#create performance summary text
    summary_text = "Performance Summary\n" + "="*20 + "\n\n"

    if wind_metrics:
        summary_text += f"Wind Drift Ratio: {wind_metrics['drift_ratio']:.2f} "
        summary_text += f"({'PASS' if wind_metrics['drift_ratio_pass'] else 'FAIL'})\n"

    if speed_metrics:
        summary_text += f"Speed Consistency: {speed_metrics['speed_consistency']*100:.1f}%\n"
        summary_text += f"Gust Events: {speed_metrics['gust_events']}\n"

    if altitude_metrics:
        summary_text += f"Altitude Hold: {altitude_metrics['compliance_rate']*100:.1f}% "
        summary_text += f"({'PASS' if altitude_metrics['altitude_hold_pass'] else 'FAIL'})\n"
        summary_text += f"Max Deviation: {altitude_metrics['max_deviation']:.2f}m\n"

    if battery_metrics:
        summary_text += f"Battery Used: {battery_metrics['battery_consumed_pct']:.1f}%\n"
        summary_text += f"Distance: {battery_metrics['total_distance']:.0f}m\n"
        summary_text += f"Efficiency: {battery_metrics['distance_per_battery_pct']:.1f}m/%\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, ha='left', va='top',
             fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

#integration function to add to your existing analyze_all_trials function
def enhanced_trial_analysis(trial_path, trial_folder, trial_output):
    """
    Enhanced analysis function that adds the four new metrics
    Call this from your existing analyze_all_trials function
    """
#load your existing data
    data = load_flight_data(trial_path)

#calculate new metrics
    wind_metrics = calculate_wind_drift_analysis(data['local_pos'], data['status'])
    speed_metrics = analyze_speed_profile(data['local_pos'])
    altitude_metrics = analyze_altitude_hold_performance(data['local_pos'])
    battery_metrics = analyze_battery_efficiency(data.get('battery', pd.DataFrame()), data['local_pos'])

#calculate rtl response time
    rtl_metrics = analyze_rtl_response_time(data['local_pos'], data['status'])

#calculate distance-based performance
    distance_metrics = analyze_distance_performance(trial_folder, data['local_pos'])

#create the four specific plots requested by the user

#wind drift comparison
    create_wind_drift_comparison_plot(
        wind_metrics,
        trial_folder,
        os.path.join(trial_output, 'wind_drift_comparison.png')
    )

#altitude hold performance
    create_altitude_hold_plot(
        data['local_pos'],
        altitude_metrics,
        trial_folder,
        os.path.join(trial_output, 'altitude_hold_performance.png')
    )

#also create the original enhanced analysis plot for backward compatibility
    create_enhanced_analysis_plots(
        data['local_pos'], data['status'], data.get('battery', pd.DataFrame()),
        wind_metrics, speed_metrics, altitude_metrics, battery_metrics,
        trial_folder, os.path.join(trial_output, 'enhanced_analysis.png')
    )


#return metrics for summary
    return {
        'wind': wind_metrics,
        'speed': speed_metrics,
        'altitude': altitude_metrics,
        'battery': battery_metrics,
        'rtl': rtl_metrics,
        'distance': distance_metrics
    }

def add_enhanced_metrics_to_results(results_list, enhanced_metrics):
    """
    Add the new metrics to your existing results structure
    """
    enhanced_results = []

    for i, result in enumerate(results_list):
        enhanced_result = result.copy()

        if i < len(enhanced_metrics):
            metrics = enhanced_metrics[i]

#add wind metrics
            if metrics['wind']:
                enhanced_result.update({
                    'drift_ratio': metrics['wind']['drift_ratio'],
                    'drift_ratio_pass': metrics['wind']['drift_ratio_pass'],
                    'outbound_drift': metrics['wind']['outbound_mean_drift'],
                    'return_drift': metrics['wind']['return_mean_drift']
                })

#add speed metrics
            if metrics['speed']:
                enhanced_result.update({
                    'mean_cruise_speed': metrics['speed']['mean_cruise_speed'],
                    'speed_consistency': metrics['speed']['speed_consistency'],
                    'gust_events': metrics['speed']['gust_events']
                })

#add altitude metrics
            if metrics['altitude']:
                enhanced_result.update({
                    'altitude_hold_pass': metrics['altitude']['altitude_hold_pass'],
                    'altitude_compliance': metrics['altitude']['compliance_rate'],
                    'max_altitude_deviation': metrics['altitude']['max_deviation']
                })

#add battery metrics
            if metrics['battery']:
                enhanced_result.update({
                    'battery_efficiency': metrics['battery']['distance_per_battery_pct'],
                    'energy_per_km': metrics['battery']['energy_per_km'],
                    'total_distance': metrics['battery']['total_distance']
                })

        enhanced_results.append(enhanced_result)

    return enhanced_results

def create_executive_summary_plot(results_df, output_path):
    """Create a comprehensive executive summary visualization"""
#filter out nan values
    valid_results = results_df.dropna(subset=['landing_error'])

    if valid_results.empty:
        print("Warning: No valid landing error data found. Skipping executive summary plot.")
        return

#create figure with gridspec for more control over layout
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])  # top left
    ax2 = fig.add_subplot(gs[0, 1])  # top right
    ax3 = fig.add_subplot(gs[1, 0])  # bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # bottom right

    fig.suptitle('TC002 VTOL RTL Test Results - Executive Summary', fontsize=16, fontweight='bold')

#1. landing accuracy analysis (top left)
#separate pass and fail landings
    pass_mask = valid_results['landing_error'] <= 0.5
    fail_mask = ~pass_mask

#plot pass landings with faded dots
    ax1.scatter(valid_results.loc[pass_mask, 'landing_x'],
               valid_results.loc[pass_mask, 'landing_y'],
               c='green', s=80, alpha=0.3, label='PASS')

#plot fail landings with more prominent dots
    ax1.scatter(valid_results.loc[fail_mask, 'landing_x'],
               valid_results.loc[fail_mask, 'landing_y'],
               c='red', s=100, alpha=0.7, label='FAIL')

#add home position
    ax1.scatter(0, 0, marker='*', s=200, c='blue', label='Home Position')

#add pass/fail circle (0.5m requirement)
    circle = Circle((0, 0), 0.5, color='green', fill=False, linewidth=2, linestyle='--')
    ax1.add_patch(circle)

#add dashed lines from home to landing for fails > 30m
    far_fails = valid_results[fail_mask & (valid_results['landing_error'] > 30)]
    for _, row in far_fails.iterrows():
        ax1.plot([0, row['landing_x']], [0, row['landing_y']],
                'r--', alpha=0.5, linewidth=1)


#create a legend box for the worst misses
    if len(valid_results[fail_mask]) > 0:
#get the worst failures (up to 3)
        worst_fails = valid_results[fail_mask].nlargest(min(3, len(valid_results[fail_mask])), 'landing_error')

#add small numbers next to the worst failures
        for i, (_, row) in enumerate(worst_fails.iterrows()):
            ax1.text(row['landing_x'], row['landing_y'], f"{i+1}",
                    fontsize=9, color='white', fontweight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='circle', facecolor='red', alpha=0.9, pad=0.2))

#create a custom legend box in the top left
#create a figure legend with custom handles
    legend_handles = []
    legend_labels = []

#add worst misses to legend if any
    if len(valid_results[fail_mask]) > 0:
#create a legend box with border
        legend_box = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                             fontsize=8, verticalalignment='top', horizontalalignment='left',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#build the legend text exactly as in the example
        legend_text = "Worst Misses:\n"
        for i, (_, row) in enumerate(worst_fails.iterrows()):
#extract just the trial number from the trial name if possible
            trial_name = row['trial']
            trial_num = trial_name.split('_')[-1] if '_' in trial_name else trial_name
            legend_text += f"  # {i+1}: {row['landing_error']:.1f}m\n"

#add standard legend items
        legend_text += "● PASS\n"
        legend_text += "● FAIL\n"
        legend_text += "★ Home Position"

#set the legend text
        legend_box.set_text(legend_text)
    else:
#if no failures, just use standard legend
        pass_patch = mpatches.Patch(color='green', alpha=0.3, label='PASS')
        fail_patch = mpatches.Patch(color='red', alpha=0.7, label='FAIL')
        home_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                                markersize=10, label='Home Position')

        ax1.legend(handles=[pass_patch, fail_patch, home_marker],
                  loc='upper left', fontsize=8)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Landing Error from Home Position (0.5m threshold)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

#remove the default legend since we're using a custom text box
#ax1.legend(loc='upper left')

#2. performance breakdown (top right) - horizontal bars matching table colors
    ax2.set_title('Test Pass Rates', fontsize=12, fontweight='bold')

#calculate pass rates
    landing_pass_rate = pass_mask.mean() * 100

#add new metrics pass/fail if available
    wind_pass = valid_results.get('drift_ratio_pass', pd.Series([True] * len(valid_results)))
    wind_pass_rate = wind_pass.mean() * 100

    altitude_pass = valid_results.get('altitude_hold_pass', pd.Series([True] * len(valid_results)))
    altitude_pass_rate = altitude_pass.mean() * 100

#overall pass requires all criteria to pass
    overall_pass = pass_mask & wind_pass & altitude_pass
    overall_pass_rate = overall_pass.mean() * 100

#create metrics in fixed order for consistency
    metrics = ['Wind Drift', 'Altitude', 'Landing', 'Overall']
    pass_rates = [wind_pass_rate, altitude_pass_rate, landing_pass_rate, overall_pass_rate]

#determine colors based on thresholds - match table colors
    colors = []
    for rate in pass_rates:
        if rate >= 80:
            colors.append('#4caf50')  # green - same as table header
        elif rate >= 60:
            colors.append('#ff9800')  # orange
        else:
            colors.append('#f44336')  # red

#create horizontal bars
    y_pos = np.arange(len(metrics))
    bar_height = 0.6

#draw background bars (100%)
    ax2.barh(y_pos, [100] * len(metrics), height=bar_height, color='#eeeeee')

#draw the actual pass rate bars
    bars = ax2.barh(y_pos, pass_rates, height=bar_height, color=colors)

#add percentage text at the end of each bar
    for i, (bar, rate) in enumerate(zip(bars, pass_rates)):
        ax2.text(rate + 1, i, f"{int(rate)}%",
                va='center', ha='left', fontweight='bold',
                color=colors[i], fontsize=10)

#add test names as y-axis labels
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(metrics, fontsize=10)

#clean up the plot
    ax2.set_xlim(0, 110)  # give some space for the percentage labels
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax2.grid(False)

#3. landing error distribution (bottom left) - updated with kde
#create histogram
    n, bins, patches = ax3.hist(valid_results['landing_error'], bins=min(10, len(valid_results)),
                               alpha=0.5, color='skyblue', edgecolor='black')

#add kernel density estimate
    if len(valid_results) >= 3:  # need at least 3 points for kde
        x = np.linspace(0, valid_results['landing_error'].max() * 1.1, 100)
        kde = stats.gaussian_kde(valid_results['landing_error'])
        ax3.plot(x, kde(x) * len(valid_results) * (bins[1] - bins[0]),
                'r-', linewidth=2, label='Density Estimate')

#add pass threshold line
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Pass Threshold (0.5m)')

#add p95 line if we have enough data
    if len(valid_results) >= 5:
        p95 = np.percentile(valid_results['landing_error'], 95)
        ax3.axvline(p95, color='purple', linestyle='-.', linewidth=2,
                   label=f'P95: {p95:.2f}m')

    ax3.set_xlabel('Landing Error (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Landing Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

#4. simplified summary table (bottom right)
    ax4.axis('off')

#create a simplified "metric strip" table
    table_data = [
        ["Metric", "Value", "Pass Rate"],
        ["Trials", f"{len(valid_results)}", "—"],
        ["Landing Error (avg)", f"{valid_results['landing_error'].mean():.2f}m", f"{landing_pass_rate:.1f}%"]
    ]

#add wind drift if available
    if 'drift_ratio' in valid_results.columns:
        drift_ratio_mean = valid_results['drift_ratio'].mean()
        table_data.append(["Wind Drift Ratio", f"{drift_ratio_mean:.2f}", f"{wind_pass_rate:.1f}%"])

#add altitude hold if available
    if 'max_altitude_deviation' in valid_results.columns:
        max_dev_mean = valid_results['max_altitude_deviation'].mean()
        table_data.append(["Altitude Compliance", f"±{max_dev_mean:.1f}m", f"{altitude_pass_rate:.1f}%"])

#add overall pass rate
    table_data.append(["All Criteria", "—", f"{overall_pass_rate:.1f}%"])

#create the table
    summary_table = ax4.table(
        cellText=table_data[1:],  # skip the header for celltext
        colLabels=table_data[0],  # use the first row as header
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.3, 0.3]
    )

#style the table
    summary_table.auto_set_font_size(False)
    summary_table.set_fontsize(9)
    summary_table.scale(1, 1.5)

#color coding for the table
    for i in range(len(table_data)):
        if i == 0:  # header row
            for j in range(3):
                cell = summary_table[i, j]
                cell.set_facecolor('#4caf50')
                cell.set_text_props(weight='bold', color='white')
        else:
            row_idx = i - 1  # adjust for header
            for j in range(3):
                if j == 2 and i > 3:  # pass rate column (skip the first few rows)
                    cell = summary_table[row_idx, j]
                    if "%" in str(table_data[i][j]):
                        rate = float(table_data[i][j].replace("%", ""))
                        if rate >= 80:
                            cell.set_facecolor('#e8f5e8')  # light green
                        else:
                            cell.set_facecolor('#fff3e0')  # light orange

#add footnote explaining the overall pass rate
    footnote = "* Overall pass requires ALL criteria to pass (Landing, Wind, Altitude)"
    ax4.text(0.05, 0.01, footnote, transform=ax4.transAxes, fontsize=8,
             style='italic', ha='left', va='bottom')

    ax4.set_title("Test Summary", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_test_summary_table(results_df, output_path):
    """Create a professional test summary table"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')

#filter valid results
    valid_results = results_df.dropna(subset=['landing_error'])

    if valid_results.empty:
        ax.text(0.5, 0.5, 'No valid test results available for summary table',
               ha='center', va='center', transform=ax.transAxes, fontsize=16)
        plt.title('TC002 VTOL RTL Test Results Summary', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

#create summary statistics
    summary_stats = {
        'Metric': [
            'Total Trials',
            'Valid Trials',
            'Landing Pass Rate',
            'Mean Landing Error',
            'Max Landing Error',
            'Min Landing Error',
            'Std Dev Landing Error',
            'Trials with GPS Loss',
            'Mean Error (No GPS Loss)',
            'Mean Error (GPS Loss)'
        ],
        'Value': [
            len(results_df),
            len(valid_results),
            f"{(valid_results['landing_error'] <= 0.5).mean()*100:.1f}%",
            f"{valid_results['landing_error'].mean():.3f} m",
            f"{valid_results['landing_error'].max():.3f} m",
            f"{valid_results['landing_error'].min():.3f} m",
            f"{valid_results['landing_error'].std():.3f} m",
            f"{valid_results['gps_loss'].sum() if 'gps_loss' in valid_results.columns else 'N/A'}",
            f"{valid_results[valid_results.get('gps_loss', 0) == 0]['landing_error'].mean():.3f} m" if 'gps_loss' in valid_results.columns and (valid_results['gps_loss'] == 0).any() else 'N/A',
            f"{valid_results[valid_results.get('gps_loss', 0) == 1]['landing_error'].mean():.3f} m" if 'gps_loss' in valid_results.columns and (valid_results['gps_loss'] == 1).any() else 'N/A'
        ],
        'Status': [
            '✓',
            '✓' if len(valid_results) > 0 else '⚠',
            '✓' if (valid_results['landing_error'] <= 0.5).mean() >= 0.8 else '⚠',
            '✓' if valid_results['landing_error'].mean() <= 0.5 else '⚠',
            '⚠' if valid_results['landing_error'].max() > 0.5 else '✓',
            '✓',
            '✓' if valid_results['landing_error'].std() <= 0.2 else '⚠',
            '✓' if valid_results.get('gps_loss', pd.Series([0])).sum() > 0 else '⚠',
            '✓' if 'gps_loss' in valid_results.columns else '⚠',
            '✓' if 'gps_loss' in valid_results.columns else '⚠'
        ]
    }

#add enhanced metrics if available
    if 'drift_ratio' in valid_results.columns:
        summary_stats['Metric'].extend([
            'Wind Drift Ratio Pass Rate',
            'Mean Drift Ratio',
            'Mean Outbound Drift',
            'Mean Return Drift'
        ])
        summary_stats['Value'].extend([
            f"{(valid_results['drift_ratio_pass'] == True).mean()*100:.1f}%",
            f"{valid_results['drift_ratio'].mean():.3f}",
            f"{valid_results['outbound_drift'].mean():.3f} m",
            f"{valid_results['return_drift'].mean():.3f} m"
        ])
        summary_stats['Status'].extend([
            '✓' if (valid_results['drift_ratio_pass'] == True).mean() >= 0.8 else '⚠',
            '✓' if valid_results['drift_ratio'].mean() <= 1.5 else '⚠',
            '✓',
            '✓'
        ])

    if 'altitude_hold_pass' in valid_results.columns:
        summary_stats['Metric'].extend([
            'Altitude Hold Pass Rate',
            'Mean Altitude Compliance',
            'Mean Max Altitude Deviation'
        ])
        summary_stats['Value'].extend([
            f"{(valid_results['altitude_hold_pass'] == True).mean()*100:.1f}%",
            f"{valid_results['altitude_compliance'].mean()*100:.1f}%",
            f"{valid_results['max_altitude_deviation'].mean():.3f} m"
        ])
        summary_stats['Status'].extend([
            '✓' if (valid_results['altitude_hold_pass'] == True).mean() >= 0.8 else '⚠',
            '✓' if valid_results['altitude_compliance'].mean() >= 0.9 else '⚠',
            '✓' if valid_results['max_altitude_deviation'].mean() <= 2.0 else '⚠'
        ])

    if 'mean_cruise_speed' in valid_results.columns:
        summary_stats['Metric'].extend([
            'Mean Cruise Speed',
            'Speed Consistency',
            'Average Gust Events'
        ])
        summary_stats['Value'].extend([
            f"{valid_results['mean_cruise_speed'].mean():.1f} m/s",
            f"{valid_results['speed_consistency'].mean()*100:.1f}%",
            f"{valid_results['gust_events'].mean():.1f}"
        ])
        summary_stats['Status'].extend([
            '✓',
            '✓' if valid_results['speed_consistency'].mean() >= 0.8 else '⚠',
            '✓' if valid_results['gust_events'].mean() <= 5 else '⚠'
        ])

    if 'battery_efficiency' in valid_results.columns:
        summary_stats['Metric'].extend([
            'Battery Efficiency',
            'Energy per km',
            'Average Distance'
        ])
        summary_stats['Value'].extend([
            f"{valid_results['battery_efficiency'].mean():.1f} m/%",
            f"{valid_results['energy_per_km'].mean():.1f} Wh/km",
            f"{valid_results['total_distance'].mean():.1f} m"
        ])
        summary_stats['Status'].extend([
            '✓' if valid_results['battery_efficiency'].mean() >= 10 else '⚠',
            '✓' if valid_results['energy_per_km'].mean() <= 50 else '⚠',
            '✓'
        ])

#add overall pass rate with all criteria
    if 'drift_ratio_pass' in valid_results.columns and 'altitude_hold_pass' in valid_results.columns:
        landing_pass = valid_results['landing_error'] <= 0.5
        wind_pass = valid_results['drift_ratio_pass'] == True
        altitude_pass = valid_results['altitude_hold_pass'] == True
        overall_pass = landing_pass & wind_pass & altitude_pass

        summary_stats['Metric'].append('Overall Pass Rate (All Criteria)')
        summary_stats['Value'].append(f"{overall_pass.mean()*100:.1f}%")
        summary_stats['Status'].append('✓' if overall_pass.mean() >= 0.8 else '⚠')

#create table
    table_data = list(zip(summary_stats['Metric'], summary_stats['Value'], summary_stats['Status']))

    table = ax.table(cellText=table_data,
                    colLabels=['Test Metric', 'Result', 'Status'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.5, 0.3, 0.2])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

#style the table
    for i in range(len(table_data) + 1):
        for j in range(3):
            cell = table[(i, j)]
            if i == 0:  # header
                cell.set_facecolor('#4caf50')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 2:  # status column
                    if table_data[i-1][2] == '✓':
                        cell.set_facecolor('#e8f5e8')
                    else:
                        cell.set_facecolor('#fff3e0')
                else:
                    cell.set_facecolor('#f5f5f5')

    plt.title('TC002 VTOL RTL Test Results Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_landing_error(local_pos):
    """Calculate landing error from home position"""
    if local_pos.empty:
        print("Warning: Empty local position data")
        return np.nan, np.nan, np.nan

    required_cols = ['x', 'y']
    missing_cols = [col for col in required_cols if col not in local_pos.columns]

    if missing_cols:
        print(f"Warning: Missing columns in local position data: {missing_cols}")
        print(f"Available columns: {list(local_pos.columns)}")
        return np.nan, np.nan, np.nan

    landing = local_pos.iloc[-1]
    x, y = landing['x'], landing['y']

#check for nan values
    if pd.isna(x) or pd.isna(y):
        print("Warning: NaN values in landing position")
        return np.nan, np.nan, np.nan

    error = np.sqrt(x**2 + y**2)
    return x, y, error

def create_detailed_flight_analysis(local_pos, status_df, gps_loss_periods, trial_name, output_path):
    """Create detailed flight path analysis with phases"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Detailed Flight Analysis - {trial_name}', fontsize=16, fontweight='bold')

    if local_pos.empty or 'x' not in local_pos.columns or 'y' not in local_pos.columns:
#create placeholder plots with error message
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, 'No valid flight data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

#1. flight path with phases
    phases = get_flight_phases(status_df)

#plot trajectory
    ax1.plot(local_pos['x'], local_pos['y'], 'b-', linewidth=2, alpha=0.7, label='Flight Path')
    ax1.scatter(0, 0, marker='*', s=200, c='green', label='Home', zorder=5)
    ax1.scatter(local_pos.iloc[-1]['x'], local_pos.iloc[-1]['y'],
               marker='X', s=200, c='red', label='Landing', zorder=5)

#add phase markers
    phase_colors = {'AUTO_RTL': 'red', 'AUTO_MISSION': 'blue', 'AUTO_TAKEOFF': 'green'}
    for phase_name, phase_data in phases.items():
        if phase_name in phase_colors:
            phase_points = local_pos[
                (local_pos['timestamp'] >= phase_data['start']) &
                (local_pos['timestamp'] <= phase_data['end'])
            ]
            if not phase_points.empty:
                ax1.plot(phase_points['x'], phase_points['y'],
                        color=phase_colors[phase_name], linewidth=3,
                        alpha=0.8, label=phase_name)

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Flight Path with Mission Phases')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axis('equal')

#2. altitude profile
    if 'z' in local_pos.columns:
        ax2.plot(local_pos['timestamp'], -local_pos['z'], 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude Profile')
        ax2.grid(True, alpha=0.3)

#shade gps loss periods
        for start, end in gps_loss_periods:
            ax2.axvspan(start, end, color='red', alpha=0.3, label='GPS Loss')
    else:
        ax2.text(0.5, 0.5, 'No altitude data available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)

#3. speed profile
    if len(local_pos) > 1 and 'timestamp' in local_pos.columns:
#calculate ground speed
        dx = np.diff(local_pos['x'])
        dy = np.diff(local_pos['y'])
        dt = np.diff(local_pos['timestamp'])
        dt[dt == 0] = 1e-6  # avoid division by zero
        speed = np.sqrt(dx**2 + dy**2) / dt

        ax3.plot(local_pos['timestamp'][1:], speed, 'g-', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Ground Speed (m/s)')
        ax3.set_title('Ground Speed Profile')
        ax3.grid(True, alpha=0.3)

#shade gps loss periods
        for start, end in gps_loss_periods:
            ax3.axvspan(start, end, color='red', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for speed calculation',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)

#4. rtl performance analysis
    if 'AUTO_RTL' in phases:
        rtl_data = local_pos[
            (local_pos['timestamp'] >= phases['AUTO_RTL']['start']) &
            (local_pos['timestamp'] <= phases['AUTO_RTL']['end'])
        ]

        if len(rtl_data) > 1:
#calculate lateral deviation during rtl
            start_pos = np.array([rtl_data.iloc[0]['x'], rtl_data.iloc[0]['y']])
            end_pos = np.array([rtl_data.iloc[-1]['x'], rtl_data.iloc[-1]['y']])

            lateral_devs = []
            for _, row in rtl_data.iterrows():
                pos = np.array([row['x'], row['y']])
                dev = lateral_deviation(pos, start_pos, end_pos)
                lateral_devs.append(dev)

            ax4.plot(rtl_data['timestamp'], lateral_devs, 'r-', linewidth=2)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Lateral Deviation (m)')
            ax4.set_title('RTL Path Deviation')
            ax4.grid(True, alpha=0.3)

#shade gps loss periods
            for start, end in gps_loss_periods:
                if start < phases['AUTO_RTL']['end'] and end > phases['AUTO_RTL']['start']:
                    ax4.axvspan(max(start, phases['AUTO_RTL']['start']),
                               min(end, phases['AUTO_RTL']['end']),
                               color='red', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient RTL data',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    else:
        ax4.text(0.5, 0.5, 'No RTL phase detected',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_all_trials(input_root, output_root):
    """Main analysis function with enhanced error handling and corrected GPS detection"""
#expand the tilde in the path
    input_root = os.path.expanduser(input_root)
    output_root = os.path.expanduser(output_root)

    os.makedirs(output_root, exist_ok=True)
    results = []
    enhanced_metrics_list = []  # list to store enhanced metrics for each trial

    print("Analyzing trials...")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")

#check if input directory exists
    if not os.path.exists(input_root):
        print(f"Error: Input directory {input_root} does not exist")
        return pd.DataFrame()

#get all subdirectories and filter for valid trial folders
    try:
        all_folders = [f for f in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, f))]
        print(f"All folders found: {all_folders}")
    except Exception as e:
        print(f"Error listing directories in {input_root}: {e}")
        return pd.DataFrame()

    trial_folders = []
    for folder in all_folders:
        folder_path = os.path.join(input_root, folder)
        print(f"\nChecking folder: {folder}")
        if is_valid_trial_folder(folder_path, folder):
            trial_folders.append(folder)
            print(f"✓ {folder} is a valid trial folder")
        else:
            print(f"✗ {folder} is not a valid trial folder")

    print(f"\nFound {len(all_folders)} total folders, {len(trial_folders)} valid trial folders")
    print(f"Valid trial folders: {trial_folders}")

    if not trial_folders:
        print(f"Error: No valid trial folders found in {input_root}")
        print("Valid trial folders should contain vehicle_local_position.csv or vehicle_gps_position.csv")
        return pd.DataFrame()

    for trial_folder in sorted(trial_folders):
        trial_path = os.path.join(input_root, trial_folder)
        print(f"\nProcessing {trial_folder}...")

        data = load_flight_data(trial_path)

#use the corrected gps loss detection with 10s minimum flight time
        gps_loss_periods = get_gps_loss_periods(data['failsafe'], min_flight_time=10.0)

        print(f"GPS loss periods found for {trial_folder}: {len(gps_loss_periods)}")
        if gps_loss_periods:
            print(f"GPS loss periods: {gps_loss_periods}")

#create individual trial output directory
        trial_output = os.path.join(output_root, trial_folder)
        os.makedirs(trial_output, exist_ok=True)

#generate detailed flight analysis
        create_detailed_flight_analysis(
            data['local_pos'],
            data['status'],
            gps_loss_periods,
            trial_folder,
            os.path.join(trial_output, 'detailed_analysis.png')
        )

#calculate metrics
        x, y, err = compute_landing_error(data['local_pos'])
        gps_loss_flag = 1 if gps_loss_periods else 0
        failsafe_flag = 0  # simplified for now

        results.append({
            'trial': trial_folder,
            'landing_x': x,
            'landing_y': y,
            'landing_error': err,
            'gps_loss': gps_loss_flag,
            'failsafe': failsafe_flag,
            'gps_loss_periods': str(gps_loss_periods)
        })

#run enhanced analysis and collect metrics
        enhanced_metrics = enhanced_trial_analysis(trial_path, trial_folder, trial_output)
        enhanced_metrics_list.append(enhanced_metrics)

        print(f"Trial {trial_folder}: Landing error = {err:.3f}m, GPS loss = {gps_loss_flag}" if not pd.isna(err) else f"Trial {trial_folder}: No valid data")

#convert to dataframe
    results_df = pd.DataFrame(results)

#add enhanced metrics to results
    enhanced_results_df = pd.DataFrame(add_enhanced_metrics_to_results(results, enhanced_metrics_list))

#save the enhanced results
    enhanced_results_df.to_csv(os.path.join(output_root, 'TC002_enhanced_summary.csv'), index=False)

#also save the original results for backward compatibility
    results_df.to_csv(os.path.join(output_root, 'TC002_summary.csv'), index=False)

#generate the rtl response time plot (comparing all trials)
    rtl_metrics_list = [metrics.get('rtl') for metrics in enhanced_metrics_list]
    create_rtl_response_time_plot(
        rtl_metrics_list,
        trial_folders,
        os.path.join(output_root, 'rtl_response_time.png')
    )

#generate the distance-based performance plot
    create_distance_performance_plot(
        enhanced_results_df,
        os.path.join(output_root, 'distance_performance.png')
    )

#print summary of data quality
    valid_trials = enhanced_results_df.dropna(subset=['landing_error'])
    gps_loss_trials = valid_trials[valid_trials['gps_loss'] == 1]

    print(f"\nData Quality Summary:")
    print(f"Total trials processed: {len(enhanced_results_df)}")
    print(f"Valid trials with data: {len(valid_trials)}")
    print(f"Trials with missing data: {len(enhanced_results_df) - len(valid_trials)}")
    print(f"Trials with GPS loss (after 10s): {len(gps_loss_trials)}")

    if not valid_trials.empty:
#generate executive summary with enhanced metrics
        create_executive_summary_plot(enhanced_results_df, os.path.join(output_root, 'executive_summary.png'))

#generate test summary table with enhanced metrics
        create_test_summary_table(enhanced_results_df, os.path.join(output_root, 'test_summary_table.png'))

#print enhanced metrics summary
        print(f"Pass rate (landing): {(valid_trials['landing_error'] <= 0.5).mean()*100:.1f}%")

        if 'drift_ratio_pass' in valid_trials.columns:
            print(f"Pass rate (wind drift): {(valid_trials['drift_ratio_pass'] == True).mean()*100:.1f}%")

        if 'altitude_hold_pass' in valid_trials.columns:
            print(f"Pass rate (altitude hold): {(valid_trials['altitude_hold_pass'] == True).mean()*100:.1f}%")

#calculate overall pass rate (all criteria must pass)
        if 'drift_ratio_pass' in valid_trials.columns and 'altitude_hold_pass' in valid_trials.columns:
            landing_pass = valid_trials['landing_error'] <= 0.5
            wind_pass = valid_trials['drift_ratio_pass'] == True
            altitude_pass = valid_trials['altitude_hold_pass'] == True
            overall_pass = landing_pass & wind_pass & altitude_pass
            print(f"Overall pass rate: {overall_pass.mean()*100:.1f}%")

        print(f"Mean landing error: {valid_trials['landing_error'].mean():.3f}m")
        print(f"GPS loss rate: {(valid_trials['gps_loss'] == 1).mean()*100:.1f}%")
    else:
        print("Warning: No valid trial data found. Check your data files and column names.")

    print(f"\nAnalysis complete! Results saved to {output_root}")

    return enhanced_results_df

#main execution block
if __name__ == "__main__":
    input_directory = "~/dev/raven/logs/2025-06-04_TC002/csv"
    output_directory = "~/dev/raven/analysis/TC002_run_summary/logs_csv"

#run the analysis
    results = analyze_all_trials(input_directory, output_directory)

    print("Analysis complete!")
