#!/bin/bash

for scenario in `ls scenarios/scenario_*.yaml`; 
	do python scenario_editor.py --capture outputs/${${scenario##*/}%.*}.mp4; 
done
