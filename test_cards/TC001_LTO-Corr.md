# Test Card: TC001_LTO-Corr

**Test Name**		Square Path Flight Profile and LTO Accuracy
**Revision**		1.0
**Date**		2025-05-31
**Prepared By**		Cole Petrich
**System Under Test**	Gazebo Iris (quadcopter); PX4 v1.14 SITL
**Control Script**	square_test.py
**Iterations**		5 identical trials

## 1 - Objective
Verify quadcopter navigation and altitude control by flying a 3d square flight pattern under conditions:

1. Take off to 5m AGL
2. Fly a 20m x 20m square with altitude changes
3. Land within 0.2m XY of takeoff location


## 2 - Flight Profile
| Step		| Action Description		| Coordinates	|
|:--------------|:-----------------------------:|--------------:|
| **T0**	| Takeoff			| (0, 0, -5)	|
| **T1**	| Fly North 20m; climb to 20m	| (20, 0, -20)	|
| **T2**	| Fly East 20m; hold 20m	| (20, 20, -20)	|
| **T3**	| Fly South 20m; descend to 15m	| (0, 20, -15)	|
| **T4**	| Fly  West 20m; hold 15m	| (0, 0, -15)	|
| **T5**	| Descend and land		| (0, 0, 0)	|


All coordinates using local NED frame. Z is negative up.

## 2.1 - Expected Results
- All setpoints reached precisely (limited overshoots, altitude accurately maintained), mission completed with limited deviations
- Stable flight, esp. during landing, transitions, landing
- Precise landing within 0.2m tolerance
- No failsafes triggering during flight


## 3 - Setup and Environment
**Startup cmd**		HEADLESS=0 make px4_sitl gazebo_iris
**PX4 build**		make px4_sitl gazebo_iris
**Start position**	(0, 0, 0) NED
**Flight path**		Autonomous square via offboard points set in square_test.py
**Disturbances**	None
**Parameters**		Default MPC and EKF params
**Logging**		PX4 .ulg files (auto-record)

## 4 - Pass / Fail Criteria
| Metric			| Requirement			| Data Source		|
| :-----------------------------| :----------------------------:| ---------------------:|
| XY error at each waypoint	| <= 1.0m from command		| vehicle_local_position|
| Altitude error at waypoints	| <= 0.5m			| vehicle_local_position|
| Final landing XY error	| <= 0.2m from takeoff point	| vehicle_local_position|
| Hover oscillation at each leg	| <= 0.2m peaks			| vehicle_local_position|
| Max velocity during corners	| <= 2.0m/s			| vehicle_local_position|
| Failesafe triggered		| False				| vehicle_status	|

## 5 - Test Procedure
1. Launch PX4 and Gazebo
make px4_sitl gazebo_iris

2. Run mission script
python3 ~/dev/raven/tests/square_test.py

3. Confirm vehicle completes all segments and lands
4. Archive log
5. Repeat for 5 trials

## 6 - Data Logging
Convert logs from .ulg to csv

Analyze and plot
	3D trajectory with overlayed setpoints
	Position errors vs. setpoints
	Altitude vs. time
	Velocity at corners
	Landing precision scatter plot

Generate report
	TC001_Debrief.pdf with charts, table of metrics, and verdict

## 7 - Risks and Mitigations
| Risk					| Mitigation					|
|:--------------------------------------|----------------------------------------------:|
| Setpoint dropouts/timeouts		| Script has heartbeat and pre arm checks	|
| Log overwrites			| Unique naming convention aftereach run	|
| Waypoint overshoot			| Smooth velocity transitions between corners	|
| Parameter drift			| Save / load config before run			|

## 8 - Deliverables
TC001_LTO-Corr (this test card)
Five .ulg logs and converted .csv files
Flight logs analysis (TC001_analysis.py)
Report (TC001_Debrief.pdf)

## 9 - Participants
Cole Petrich - Test Engineer

## 10 - Notes / Lessons Learned
- All missions completed successfully
- Landing location all within 0.2m radius from takeoff location
- Observed overshooting waypoints
- Altitude hold steady with minimal oscillations (only occurred when holding altitude between WP1 and WP2)

- Figured out PX4 workflow, succesfully starting sim and controlling flight operations through python script
- Performed analysis based off flight logs, using vehivle_location and vehicle_Status logs; aim to incorporate other logs next to observe attitude, battery usage, transitions

- Future:
	- Introduce wind
	- Use VTOL aircraft (standard_vtol)
	- Increase range of test flights (300m, 700m, 1200m)
	- Cross examine other logs to better understand mission through data
	- Add real-time log reading into python script to catch errors in real-time
	- Automate workflow end-to-end (sim --> analysis/graph outputs)

