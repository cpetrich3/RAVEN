import asyncio
from mavsdk import System, offboard

def pos(n: float, e: float, d: float, yaw: float = 0.0):
    return offboard.PositionNedYaw(n, e, d, yaw)

async def main():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    print("Connected")
    # arm and takeoff to 5m
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(5)

    # increase agl from 5m to 10m
    await drone.offboard.set_position_ned(pos(0, 0, -10))
    await drone.offboard.start()
    print("Holding 10 m")
    await asyncio.sleep(2)

    # waypoint 1 - head north 20m, climb 20m
    print("North 20m  | climb to 20m")
    await drone.offboard.set_position_ned(pos(20, 0, -20))
    await asyncio.sleep(10)		#wait for UAV to reach waypoint...can adjust

    # waypoint 2 - head east 20m, hold altitude
    print("East 20m   | hold 20m")
    await drone.offboard.set_position_ned(pos(20, 20, -20))
    await asyncio.sleep(10)

    # waypoint 3 - head south 20m, descending to 15m
    print("South 20m  | descend to 15m")
    await drone.offboard.set_position_ned(pos(0, 20, -15))
    await asyncio.sleep(10)

    # waypoint 4 - head west 20m, hold altitude
    print("West 20m   | hold 15m")
    await drone.offboard.set_position_ned(pos(0, 0, -15))
    await asyncio.sleep(10)

    # landing
    print("Landing…")
    await drone.offboard.stop()
    await drone.action.land()
    await asyncio.sleep(10)    # time to wait for landing

if __name__ == "__main__":
    asyncio.run(main())
