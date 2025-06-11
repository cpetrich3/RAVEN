#!/bin/bash

# Create destination folder if it doesn't exist
mkdir -p ~/dev/raven/logs

# Copy all .ulg files from PX4 log folder to raven logs
cp -r ~/dev/PX4-Autopilot/build/px4_sitl_default/rootfs/log/*/ ~/dev/raven/logs/
