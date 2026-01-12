# Airflow Docker Quickstart

## Prereqs
- Docker + Docker Compose plugin
- `make` installed

## One-time init
- `make init` (migrates DB, creates admin user)

## Start/stop
- CPU: `make up`
- GPU: `make up-gpu`
- Stop: `make down`
- Clean all (incl. volumes): `make clean`

## Status & logs
- Status: `make ps`
- Logs (web + scheduler): `make logs`

## Restart
- Web: `make restart-web`
- Scheduler: `make restart-scheduler`

## UI login
- URL: http://localhost:8080
- User: admin / Password: admin

## Notes
- Project name set via COMPOSE_PROJECT_NAME in `.env` (default: `airflow-batch`).
- GPU mode expects NVIDIA Container Toolkit installed.
