#!/bin/bash

# --------------------------------------------------------------------
# Specific Sequence and Scene

seq=02
scene=001408
export BIN_PATH=/home/semanticKITTI/dataset/sequences/

python assetValidDemoV4.py -sequence $seq -scene $scene

# --------------------------------------------------------------------
# Random Sequence and Scene

# python assetValidDemoV4.py

# --------------------------------------------------------------------



