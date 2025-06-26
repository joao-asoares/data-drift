#!/bin/bash

uv init
uv venv

uv add scikit-multiflow
uv add pandas
uv add pyarrow
uv add fastparquet
uv add numpy==1.23.5

uv run analyze_page_hinkley.py