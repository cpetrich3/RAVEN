# RAVEN - Reusable Autonomous Vehicle Evaluation Network
## A PX4 Testing and Analysis Suite
A scalable UAV test suite for PX4-based systems, built for simulation and future deployment in the field.

## Overview
RAVEN is a test automation and post-flight analysis framework designed to evaluate VTOL systems in complex environments using PX4 SITL simulations.  
It focuses on validating autonomy logic (like Return-to-Launch) under real-world stressors such as wind, long-range flight, and GPS degradation.


The framework is lightweight, modular, and built to evolve across multiple test campaigns.

### Highlights  
- **Structured Test Cards:** Define repeatable UAV tests via YAML-based templates  
- **Automated Execution:** MAVSDK-Python test scripts with support for GPS loss, wind injection, and long-range missions  
- **Post-Flight Analysis:** Pandas-based tools to extract, visualize, and summarize key performance metrics  
- **Streamlit Dashboard (WIP):** In-progress interactive interface to explore results and evaluate new tests  

## Project Structure
- 'logs/'       - Archived `.ulg` or `.log` files from SITL tests  
- 'utils/'      - Bash/Python scripts to automate tests or run PX4  
- 'test_cards/' - Structured descriptions of each test case  
- 'tests/'      - Test scripts using MAVSDK to run trials
- 'analysis/'   - Scripts to process & visualize logs
- 'matrix/'     - Test matrix YAML files
- 'dashboard'   - Streamlit viz app (in progress)

## Current Test Campaigns

### TC001 - Basic Square Landing Accuracy
Established a controlled baseline for landing accuracy in a zero-wind setting.
Demonstrates sub-0.15m landing precision.
[Read the full debrief here](https://github.com/cpetrich3/RAVEN/blob/main/analysis/debriefs/TC001_Debrief.pdf)

### TC002 - RTL Under Wind and GPS Loss
Evaluated RTL behavior across 5 scenarios with long range return, crosswinds, and GPS degradation.
Exposes GPS as a single point of failure and highlights boundaries in performance.
[Read the full debrief here](https://github.com/cpetrich3/RAVEN/blob/main/analysis/debriefs/TC002_Debrief.pdf)

### TC003 - In Progress
- GPS Jamming / spoofing
- Sensor dropout
- High-wind environment
- Failsafe logic exploration

## Test Philosophy
RAVEN was built to simulate real-world UAV failure cases and analyze the response.
Its the start of a test framework that can scale from SITL to Hardware to Field.

## About the Author
Cole Petrich - UAV systems engineer, obsessed with autonomy, and extracting signal from noise.
Built this in evenings and weekends with the goal of learning by doing.

![GitHub repo size](https://img.shields.io/github/repo-size/cpetrich3/RAVEN)
![Last Commit](https://img.shields.io/github/last-commit/cpetrich3/RAVEN)
