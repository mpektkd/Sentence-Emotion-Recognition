#!/bin/bash

python main.py

python createdata.py BiH_AttentionLSTMbestmodel.pt

python bonus.py

python bonus_main.py