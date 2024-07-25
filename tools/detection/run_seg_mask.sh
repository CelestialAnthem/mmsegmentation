#!/bin/bash

python tools/detection/mask_to_det.py 0 5200 0
python tools/detection/mask_to_det.py 5200 10400 1
python tools/detection/mask_to_det.py 10400 15600 2
python tools/detection/mask_to_det.py 15600 20800 3
python tools/detection/mask_to_det.py 20800 26000 4
python tools/detection/mask_to_det.py 26000 31200 5
python tools/detection/mask_to_det.py 31200 36400 6
python tools/detection/mask_to_det.py 36400 42000 7

wait