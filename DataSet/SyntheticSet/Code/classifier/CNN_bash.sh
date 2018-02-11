#!/bin/bash
mapping='0p0 0p1 0p2 0p3 0p4 0p5 0p6 0p7 0p8 0p9 1p0'

#type4
P1='0p5'
P2='0p5'
P3='0p2 0p3 0p4 0p5 0p6 0p7 0p8'

for p1 in $P1
do
	for p2 in $P2
	do 
		for p3 in $P3
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 1
		done
	done
done

#type5
P1='0p7'
P2='0p5'
P3='0p2 0p3 0p4 0p5 0p6 0p7 0p8'

for p1 in $P1
do
	for p2 in $P2
	do 
		for p3 in $P3
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 1
		done
	done
done

#type6
P1='0p5'
P2='0p7'
P3='0p2 0p3 0p4 0p5 0p6 0p7 0p8'

for p1 in $P1
do
	for p2 in $P2
	do 
		for p3 in $P3
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 1
		done
	done
done

#type7
P1='1p0'
P2='0p1'
P3='0p2 0p3 0p4 0p5 0p6 0p7 0p8'

for p1 in $P1
do
	for p2 in $P2
	do 
		for p3 in $P3
		do
			python classifier_Synthetic_all.py --p1 $p1 --p2 $p2 --p3 $p3 --nc 8 --configure 1
		done
	done
done