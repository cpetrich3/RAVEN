import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#this is a helper for lateral offset
def lateral_deviation(P, A, B):
    AB = B - A
    AP = P - A
    proj = np.dot(AP, AB) / np.dot(AB, AB) * AB
    perp = AP - proj
    return np.linalg.norm(perp)

#this is a function to analyze a single trial
def analyze_flight(flight_path):
    try:
        #load log files
        gps = pd.read_csv(os.path.join(flight_path, 'vehicle_gps_position.csv'))
        att = pd.read_csv(os.path.join(flight_path, 'vehicle_attitude.csv'))
        wind = pd.read_csv(os.path.join(flight_path, 'wind.csv'))
        batt = pd.read_csv(os.path.join(flight_path, 'battery_status.csv'))
        failsafe = pd.read_csv(os.path.join(flight_path, 'failsafe_flags.csv'))
        pos = pd.read_csv(os.path.join(flight_path, 'vehicle_local_position.csv'))
        status = pd.read_csv(os.path.join(flight_path, 'vehicle_status.csv'))

        #convert timestamps
        for df in [gps, att, wind, batt, failsafe, pos, status]:
            df['timestamp'] = df['timestamp'] * 1e-6

        #use last local position for landing
        landing = pos.iloc[-1]  #this is a comment
        x_landing = landing['x']  #this is a comment
        y_landing = landing['y']  #this is a comment
        z_landing = landing['z']  #this is a comment
        landing_error = np.sqrt(x_landing**2 + y_landing**2)  #this is a comment

        #battery and wind
        batt_v = batt['voltage_v'].min()  #this is a comment
        wind_mag = (wind['windspeed_north']**2 + wind['windspeed_east']**2)**0.5
        mean_wind = wind_mag.mean()
        max_wind = wind_mag.max()

        #vertical velocity
        climb = gps['vel_d_m_s'] * -1.0  #this is a comment
        max_climb = climb.max()
        max_desc = climb.min()

        #gps validity flags
        fs_valid = failsafe[failsafe['timestamp'] > 10]
        gps_lost = int(fs_valid[['global_position_invalid', 'home_position_invalid']].any().any())
        gps_loss_count = int(fs_valid[['global_position_invalid', 'home_position_invalid']].sum().sum())

        #rtl segment analysis
        rtl_rows = status[status['nav_state'] == 5]
        if not rtl_rows.empty:
            rtl_start = rtl_rows['timestamp'].iloc[0]
            rtl_end = pos['timestamp'].max()
            rtl_segment = pos[(pos['timestamp'] >= rtl_start) & (pos['timestamp'] <= rtl_end)].copy()
            if not rtl_segment.empty:
                A = np.array([rtl_segment.iloc[0]['x'], rtl_segment.iloc[0]['y']])
                B = np.array([0.0, 0.0])
                rtl_segment['lateral_offset'] = rtl_segment.apply(
                    lambda row: lateral_deviation(np.array([row['x'], row['y']]), A, B), axis=1)
                max_offset = rtl_segment['lateral_offset'].max()
                std_offset = rtl_segment['lateral_offset'].std()
                rtl_duration = rtl_end - rtl_start
            else:
                max_offset = std_offset = rtl_duration = -1
        else:
            max_offset = std_offset = rtl_duration = rtl_start = -1

        #plot trajectory
        plt.figure(figsize=(6,6))
        plt.plot(pos['x'], pos['y'], label='Trajectory')
        if rtl_start != -1:
            rtl_point = pos[pos['timestamp'] >= rtl_start].iloc[0]
            plt.scatter(rtl_point['x'], rtl_point['y'], color='red', label='RTL Start')
        plt.scatter(0, 0, color='green', label='Home', marker='*')
        plt.legend()
        plt.title('XY Trajectory')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(flight_path, 'trajectory_xy.png'))
        plt.close()

        #plot lateral offset
        if rtl_start != -1 and not rtl_segment.empty:
            plt.figure(figsize=(8,4))
            plt.plot(rtl_segment['timestamp'] - rtl_segment['timestamp'].iloc[0], rtl_segment['lateral_offset'])
            plt.title('Lateral Offset During RTL')
            plt.xlabel('Time [s]')
            plt.ylabel('Offset [m]')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(flight_path, 'rtl_lateral_offset.png'))
            plt.close()

        #plot altitude profile
        plt.figure(figsize=(8,4))
        plt.plot(pos['timestamp'] - pos['timestamp'].iloc[0], -pos['z'])  #this is a comment
        if rtl_start != -1:
            plt.axvline(rtl_start - pos['timestamp'].iloc[0], color='red', linestyle='--', label='RTL Start')
            plt.legend()
        plt.title('Altitude Profile')
        plt.xlabel('Time [s]')
        plt.ylabel('Altitude [m]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(flight_path, 'altitude_profile.png'))
        plt.close()

        #summary metrics
        summary = {
            'landing_x': x_landing,
            'landing_y': y_landing,
            'landing_z': z_landing,
            'landing_error_m': landing_error,
            'min_batt_voltage': batt_v,
            'mean_wind': mean_wind,
            'max_wind': max_wind,
            'max_climb_rate': max_climb,
            'max_descent_rate': max_desc,
            'gps_lost': gps_lost,
            'gps_loss_count': gps_loss_count,
            'rtl_lateral_offset_max': max_offset,
            'rtl_lateral_offset_std': std_offset,
            'rtl_duration': rtl_duration
        }

        return summary
    except Exception as e:
        print(f"Error in {flight_path}: {e}")
        return None

#this is a function to run batch analysis over all folders
def analyze_all_trials(root_dir):
    results = []
    for folder in sorted(os.listdir(root_dir)):
        trial_path = os.path.join(root_dir, folder)
        if os.path.isdir(trial_path):
            result = analyze_flight(trial_path)
            if result:
                result['trial'] = folder
                results.append(result)
    return pd.DataFrame(results)

if __name__ == "__main__":
    root_dir = "/home/colep/dev/raven/logs/2025-06-04_TC002/csv"
    output_file = "/home/colep/dev/raven/analysis/TC002_summary.csv"

    df = analyze_all_trials(root_dir)
    df.to_csv(output_file, index=False)
    print(df)
    print(f"Saved summary to: {output_file}")
