import pandas as pd

#examine failsafe file
df = pd.read_csv("~/dev/raven/logs/2025-06-04_TC002/csv/23_35_53_csv/failsafe_flags.csv")

df["time_s"] = df["timestamp"] / 1e6

#gps cols
gps_flags = [
    "global_position_invalid",
    "global_position_invalid_relaxed",
    "local_position_invalid",
    "local_position_invalid_relaxed",
    "home_position_invalid",
    "position_accuracy_low",
    "navigator_failure"
]

#looking for timestamps of gps failure
triggered = df[df[gps_flags].any(axis=1)]

if triggered.empty:
    print(" No GPS/navigation-related failsafe flags were triggered.")
else:
    print("GPS/Navigation-related failsafe triggers found:")
    print(triggered[["time_s"] + gps_flags])

