#!/bin/bash

for file in RollingMetric-*-*-0.pkl; do
	new_file=${file%-0.pkl}-2.pkl

	mv "$file" "$new_file"
done
