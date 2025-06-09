# Test Card: TC002_VTOL-RTL

**Test Name**          Long Range Return to Launch in Adverse Wind Conditions (Emergency Return Evaluation)
**Revision**           1.0
**Date**               2025-06-01 
**Prepared By**        Cole Petrich
**System Under Test**  PX4 v1.14 SITL, standard_vtol
**Control Script**     TC002_rtl_test.py
**Iterations**         15 trials across 5 scenarios

## 1 - Objective
Test how the VTOL handles RTL under wind and degraded GPS. Focus on landing accuracy, system stability, and whether RTL performs reliably across runs.

## 2 - Flight Profile
| Step   | Action Description				| Coordinates     |
|:-------|:---------------------------------------------|----------------:|
| **T0** | Takeoff to 20m				| (0, 0, -20)     |
| **T1** | Transition to forward flight mode		| (0, 0, -20)     |
| **T2** | Fly North X meters, climb to 100m		| (X, 0, -100     |
| **T3** | Fly NE to reach full distance		| (X+Y, Y, -100   |
| **T4** | Activate RTL					| Auto from FC	  |
| **T5** | Return, descend, and land			| (0, 0, 0)	  |

Repeat for 300m, 700m, and 1200m total ranges. All coordinates in local NED. Z is negative up.

## 2.1 - Expected Results
- RTL activates reliably and returns to launch with precision (<= 0.50m displacement)
- Aircraft maintains heading and altitude stability during RTL in wind
- Wind introduces lateral drift, but system corrects accordingly
- Power usage scales reasonably with distance
- No erratic attitude changes or failsafes trigger

## 2.2 - Test Cases and Trials
Each test case listed below will be executed in 3 repeated trials using a distinct sim seed to assess repeatability and system sensitivity to random variation.

| Test ID | Scenario                         | Trials | Distance (m) | Wind      | GPS Loss |
|---------|----------------------------------|--------|--------------|-----------|----------|
| TC002-A | Nominal RTL from 300m            | 3      | 300          | None      | No       |
| TC002-B | RTL from 1200m, crosswind        | 3      | 1200         | Crosswind | No       |
| TC002-C | RTL from 700m	             | 3      | 700          | Tailwind  | No       |
| TC002-D | RTL from 700m, GPS lost (no rec) | 3      | 700          | Headwind  | Yes      |
| TC002-E | RTL from 700m, GPS recovers      | 3      | 700          | Crosswind | Yes (rec)|

Total trials: **15**

## 3 - Setup and Environment
**PX4 build**           make px4_sitl gazebo standard_vtol
**Start position**	(0, 0, 0) NED - transition altitude: 20m to forward climb to 100m      
**Flight path**         TC002_rtl_test.py
**Disturbances**        Crosswind (8-15 m/s), wind gusts/variance
**Parameters**          Default PX4 MPC and EKF2, with tuning as needed for wind
**Logging**             .ulg auto-recorded, converted to .csv; Real-time log analysis

Simulation uses Gazebo with wind injected via world plugin/PX4 environment. Default lat/lon position used unless stated otherwise.

## 4 - Pass / Fail Criteria
| Metric			| Requirement					| Data Source            |
| :----------------------------	|:---------------------------------------------:|-----------------------:|
| XY error on return		| <= 0.50m final landing error			| vehicle_local_position |
| Altitude stability (RTL)	| <= +- 2.5m during RTL cruise			| vehicle_local_position |
| Time to disarm after RTL	| Logged and within expected threshold		| vehicle_status	 |
| Battery usage			| Reasonable scaling by distance		| battery status	 |
| Drift outbound vs. return	| Drift ration <= 1.5x outbound vs. return	| vehicle_local_position |
| Failsafe Trigger		| False						| vehicle status	 |

## 5 - Test Procedure
1. Lauch PX4 SITL with windy
	make px4_sitl gazebo_Standard_vtol__windy
2. Load test matrix from '~/dev/raven/matrix/TC002_matric.yaml'
3. For each test case ID:
	- execute 3 trials using different 'sim_seed' values ('12345, 98765, 43761)
4. Run control script with set distance
	python3 ~/dev/raven/tests/TC002_rtl_test.py --distance [300|700|1200]
5. Monitor flight, RTL trigger, and behavior in sim
6. Save logs after each flight

## 6 - Data Logging

- .ulg to .csv using log converter
- Real time logging script outputs (drift magnitude, battery percentage change, time from RTL trigger to disarm, PASS/FAIL on predefined criteria)
- Offline plots 
	full trajectory with setpoints and wind vector overlay
	XY error scatter on landing
	Altitude profiles across phases
	Battery depletion curves
	Drift ratio summary across trials
- test_tracker.csv:
	- appended in real time during batch execution
	- Cols: [test_id, trial_id, sim_seed, offset_m, heading_error_deg, rtl_duration_s, verdict, notes]

Results to be compiled into ~/raven/analysis/debriefs/TC002_Debrief.pdf
See analysis/TC002_RTL_analysis.py for analysis script

## 7 - Risks and Mitigations
| Risk					| Mitigation					|
|:--------------------------------------|----------------------------------------------:|
| Excessive wind drift			| Wind parameter tune or RTL altitude increase	|
| Unexpected transitions/failsafe	| Add retry/monitor logic in control script	|
| High battery drain			| Monitor in real-time, abort id unsafe		|
| Log parsing failure			| Error handling, validate after each run	|

## 8 - Deliverables
- Test card: TC002_VTOL-RTL.md
- .ulg logs and converted .csv's
- Auto-analysis plots and runtime logs
- Summary report: TC002_Debrief.pdf

## 9 - Participants
Cole Petrich - Test Engineer, Analyst, Developer

## 10 - Notes / Lessons Learned
- PX4 RTL logic relies heavily on GPS
- Wind impacts are managable with current PID tuning and trajectory planning (stock on PX4)
- Sensor anaomalies (rapid spikes) do not necessarily affect performance but should be monitored
- Realistic test condiditions (in sim) expose critical navigation edge cases not apparent in ideal runs
- Failsafe thresholds and home accuracy logic should be hardened for degraded GPS signal environments

System reliably executes RTL with under 0.1m error under normal conditions. GPS loss exposes drifts without failsafe triggered. This test campaign defines the cieling and floor of RTL performance, setting the stage for closed-loop robustnes improvements

