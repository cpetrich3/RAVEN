import asyncio
import argparse
import csv
import math
import os
import time
import yaml
from mavsdk import System, offboard
from mavsdk.offboard import PositionNedYaw

# return PositionNedYaw setpoint
def pos(n: float, e: float, d: float, yaw: float = 0.0):
    return offboard.PositionNedYaw(n, e, d, yaw)

#calculating distance from target location
async def wait_until_near(drone, target_n, target_e, target_d, radius=10.0):
    async for pv in drone.telemetry.position_velocity_ned():
        dx = pv.position.north_m - target_n
        dy = pv.position.east_m - target_e
        dz = pv.position.down_m - target_d
        dist = math.sqrt(dx**2 + dy**2 + dz**2)
        if dist <= radius:
            break
        await asyncio.sleep(2)
       
async def main():
    
    # define arguments 
    parser = argparse.ArgumentParser(description="TC002: VTOL RTL Test Script")
    parser.add_argument("--config", type=str, help="Path to YAML config file with test parameters")
    parser.add_argument("--test_id", type=str, required=True, help="Test case ID from matrix")
    args = parser.parse_args()
     
     # load matrix and extract test case parameters
    with open(args.config, 'r') as f:
        matrix = yaml.safe_load(f)
    test_case = next((tc for tc in matrix["test_cases"] if tc["test_id"] == args.test_id), None)

    if test_case is None:
        raise ValueError(f"Test ID '{args.test_id}' not found in config.")
        
    test_cases = matrix.get("test_cases", [])
    config = next((tc for tc in test_cases if tc["test_id"] == args.test_id), None)

    if config is None:
        raise ValueError(f"Test ID '{args.test_id}' not found in config.")


    distance = config.get("distance", None) 
    test_id = config.get("test_id", "TC002")
    gps_cut = config.get("gps_cut", False)
    gps_cut_delay = config.get("gps_cut_delay", 5)
    log_dir = config.get("log_dir", os.path.expanduser("~/dev/raven/logs/TC002"))
    # set up .csv for summary metrics
    os.makedirs(log_dir, exist_ok=True)
    timestamp = int(time.time())
    summary_path = os.path.join(log_dir, f"summary_{test_id}_{timestamp}.csv")
    print("setup complete")
    # connect to drone
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("connected")
        
    # check gps and home calibrated
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            break
    print("connected, ready!")
    # create and define real time logging variables
    flight_mode_val = None
    armed_val = None
    battery_val = None
    pre_rtl_battery = None
    post_rtl_battery = None
    rtl_trigger_time = None
    disarm_time = None
    landing_position = (0.0, 0.0, 0.0)
    rtl_altitudes = []
    
    async def update_flight_mode():
        nonlocal flight_mode_val
        async for mode in drone.telemetry.flight_mode():
            flight_mode_val = mode
            
    async def update_armed():
        nonlocal armed_val
        async for arm_state in drone.telemetry.armed():
            armed_val = arm_state
            
    async def update_battery():
        nonlocal battery_val
        async for bat in drone.telemetry.battery():
            battery_val = bat.remaining_percent
            
    async def status_listener():
        async for status in drone.telemetry.status_text():
            if status.type.name in ["WARNING", "ERROR", "CRITICAL"]:
                print(f"[LOG][sev={status.severity}] {status.text}")
                
    async def telemetry_collector():
        nonlocal pre_rtl_battery, post_rtl_battery, rtl_trigger_time, disarm_time, landing_position
        async for pv in drone.telemetry.position_velocity_ned():
            ts = time.time()
            flight_mode = flight_mode_val
            armed = armed_val
            pos_n = pv.position.north_m
            pos_e = pv.position.east_m
            pos_d = pv.position.down_m
            battery = battery_val
            
            await asyncio.sleep(1)              # adjust for update speed..testing @ 1s
            
            # metrics to track when RTL called and landing position
            if flight_mode != "RETURN TO LAUNCH" and pre_rtl_battery is None and battery is not None:
                pre_rtl_battery = battery
                
            if flight_mode != "RETURN TO LAUNCH" and rtl_trigger_time is None:
                rtl_trigger_time = ts
                
            if flight_mode == "RETURN TO LAUNCH":
                rtl_altitudes.append(pos_d)    
            
            if rtl_trigger_time and armed is False and disarm_time is None:
                disarm_time = ts
                landing_position = (pos_n, pos_e, pos_d)
                post_rtl_battery = battery
                break
            
            
    async def mission_executor():
        nonlocal rtl_trigger_time
        print("ARMING")
        #arm
        await drone.action.arm()
        await asyncio.sleep(1)
        print("TAKEOFF")
        # takeoff
        await drone.set_maximum_speed(15)
        await drone.offboard.set_position_ned(pos(0, 0, -20))
        await drone.offboard.start()
        await asyncio.sleep(10)
        print("TRANSITION")
        #transition to fixed wing
        await drone.action.transition_to_fixedwing()
        await asyncio.sleep(3)
        

        #distance to deploy (XY coords)
        X = distance / 2.0
        a, b = 2.0, 2.0 * X
        c = (X *X) - (distance * distance)
        disc = b * b - (4 *a *c)
        if disc < 0:
            raise RuntimeError(f"Cannot solve for Y (disc={disc:.2f}) for distance={distance}")
        Y = (-b + math.sqrt(disc)) / (2.0 * a)

        print("WP1")
        
        await drone.offboard.set_position_ned(X, 0.0, -100.0, 0)
        await asyncio.sleep((X/15.0) + 1)
        print("WP2")
        await drone.offboard.set_position_ned(X + Y, Y, -100.0)
        await asyncio.sleep(math.sqrt((2 * Y**2) / 15) + 1)
        
       # await drone.offboard.stop()
        print("RTL")
        await drone.action.return_to_launch()
        rtl_trigger_time =  time.time()
        
        if gps_cut:
            await asyncio.sleep(config[gps_cut_delay], 10)
            print("Simulating GPS loss")
            
    await asyncio.gather(
        update_flight_mode(),
        update_armed(),
        update_battery(),
        status_listener(),
        telemetry_collector(),
        mission_executor()
    )
    print(1)
    ttl = disarm_time - rtl_trigger_time if (rtl_trigger_time and disarm_time) else None
    final_n, final_e, _ = landing_position
    print(2)
    horizontal_error = math.hypot(final_n, final_e)
    battery_used = (pre_rtl_battery - post_rtl_battery) * 100 if (pre_rtl_battery and post_rtl_battery) else None
    max_dev = max([abs(alt - (-100.0)) for alt in rtl_altitudes]) if rtl_altitudes else None
    print(3)
    with open(summary_path, 'w') as f:
        f.write("test_id,distance,gps_cut,rtl_trigger_time,disarm_time,rtl_duration_sec,landing_error_m,battery_used_percent,max_altitude_dev_m\n")
        f.write(f"{test_id},{distance},{gps_cut},{rtl_trigger_time},{disarm_time},{ttl},{horizontal_error},{battery_used},{max_dev}\n")
    
    print("\n========== TEST SUMMARY ==========")
    print(f"Summary written to {summary_path}")
    print("Test complete")
    
if __name__ == "__main__":
    asyncio.run(main())
