#!/bin/bash

echo "Enter Principal (P):"
read P
echo "Enter Rate of Interest (R):"
read R
echo "Enter Time (T):"
read T

SI=$(( (P * R * T) / 100 ))
echo "Simple Interest is: $SI"
