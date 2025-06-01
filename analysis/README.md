# RAVEN - Analysis Quick Start

#1 Create and activate a venv
python -m venv venv
source venv/bin/activate
install requirements: pandas, numpy, tabulate, pyulog

# 2  Run KPIs on every .ulg inside a trial folder
python analysis/extract_kpis.py logs/2025-06-01_squareWind \
       --output analysis/squareWind.csv

# 3  Append further runs
python analysis/extract_kpis.py logs/2025-06-02_squareWind \
       --output analysis/squareWind.csv --append
