#!/bin/bash

STATION_ID=97100

mkdir -p "./tmp"

for PARAM in {0..100}; do
	curl -f "https://opendata-download.smhi.se/stream?type=metobs&parameterIds=$PARAM&stationId=$STATION_ID&period=corrected-archive" -o "./tmp/$STATION_ID-$PARAM-archive.csv"
	curl -f "https://opendata-download.smhi.se/stream?type=metobs&parameterIds=$PARAM&stationId=$STATION_ID&period=latest-months" -o "./tmp/$STATION_ID-$PARAM-latest.csv"
done

grep -il "1 gång/tim;" ./tmp/*.csv | xargs -I % cp % .
sed -i '/Datum/,$!d' *.csv