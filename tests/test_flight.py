import asyncio
from mavsdk import System, offboard

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected")
            break

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    print("Starting Offboard Mode...")
    await drone.offboard.set_position_ned(
        offboard.PositionNedYaw(0.0, 0.0, -5.0, 0.0)  # start at 5m altitude
    )
    await drone.offboard.start()
    await asyncio.sleep(5)

    print("Climbing to 10m altitude...")
    await drone.offboard.set_position_ned(
        offboard.PositionNedYaw(0.0, 0.0, -10.0, 0.0)
    )
    await asyncio.sleep(5)

    print("Descending to 3m altitude...")
    await drone.offboard.set_position_ned(
        offboard.PositionNedYaw(0.0, 0.0, -3.0, 0.0)
    )
    await asyncio.sleep(5)

    print("Landing...")
    await drone.action.land()

if __name__ == "__main__":
    asyncio.run(run())
