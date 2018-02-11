#!/bin/bash
mapping='0p0 0p1 0p2 0p3 0p4 0p5 0p6 0p7 0p8 0p9 1p0'
mp1='0p1'
mp2='1p0'
for p1 in $mp1
do
	for p2 in $mp2
	do 
		for p3 in $mapping
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 1
		done
	done
done

for p1 in $mp1
do
	for p2 in $mp2
	do 
		for p3 in $mapping
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 2
		done
	done
done

