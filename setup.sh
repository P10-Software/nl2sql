#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run database setup
python3 -m src.database.setup_database
