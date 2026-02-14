#!/bin/bash
set -e

# Remove existing .env file if it exists
rm -f .env
rm -rf ./logs ./plugins ./config

# Stop and remove containers, networks, and volumes
docker compose down -v

# Create required Airflow directories (working_data for DAG outputs e.g. cluster_report.txt)
mkdir -p ./logs ./plugins ./config ./working_data

# Write the current user's UID into .env
echo "AIRFLOW_UID=$(id -u)" > .env

# Run airflow CLI to show current config
docker compose run --rm airflow-cli airflow config list
