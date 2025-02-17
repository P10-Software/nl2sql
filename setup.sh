#!/bin/bash

# Load environment variables
export $(grep -v '^#' .env | xargs)

# Install dependencies
pip install -r requirements.txt
sudo apt update && sudo apt install -y postgresql postgresql-contrib

# Start PostgreSQL service
sudo service postgresql start

# Change PostgreSQL user password
sudo -u postgres psql -c "ALTER USER postgres PASSWORD '$PG_PASSWORD';"

# Run database setup
python3 -m src.database.setup_database
