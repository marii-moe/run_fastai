#!/bin/bash
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install
jupyter nbextensions_configurator enable
./run_jupyter.sh
