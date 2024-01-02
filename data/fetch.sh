#!/bin/bash

STATION_ID=98230

for PARAM in {0..30}; do
	curl -f "https://opendata-download.smhi.se/stream?type=metobs&parameterIds=$PARAM&stationId=$STATION_ID&period=corrected-archive" -o "$STATION_ID-$PARAM-archive.csv"
	curl -f "https://opendata-download.smhi.se/stream?type=metobs&parameterIds=$PARAM&stationId=$STATION_ID&period=latest-months" -o "$STATION_ID-$PARAM-latest.csv"
done
