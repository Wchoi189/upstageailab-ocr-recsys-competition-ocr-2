# Airflow Batch Processor (Upstage APIs)

Airflow 3.1.5 (Python 3.11) project scaffold to batch process data via Upstage Document Parse / KIE APIs with local and AWS execution modes.

## Quick Start

1. Build the image (single CUDA + Airflow)

```bash
docker build -t airflow-cuda:3.1.5 -f docker/Dockerfile airflow-batch-processor
```

2. Initialize metadata DB and admin user

```bash
cd airflow-batch-processor/docker
docker compose run --rm airflow-init
```

3. Start services (compose will build the image as needed)

```bash
docker compose up -d airflow-webserver airflow-scheduler postgres
# Open http://localhost:8080  (admin / admin)
```

4. GPU (optional)
- Ensure NVIDIA Container Toolkit is installed
- Use GPU override compose file (enables NVIDIA devices):

```bash
cd airflow-batch-processor/docker
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --compatibility up -d
```

- This project uses a single image that includes CUDA libs and Airflow. For an alternative CUDA base used elsewhere in the repo, see [docker/Dockerfile](../../docker/Dockerfile).

## Volumes & Shared Data
- DAGs, plugins, and src are bind-mounted for live development.
- Config and data are shared as:
	- Host: `airflow-batch-processor/config` → Container: `/opt/airflow/config`
	- Host: `airflow-batch-processor/data` → Container: `/opt/airflow/data`
- Logs persist in a named volume `airflow_logs`.
- Environment variables are loaded from `airflow-batch-processor/.env` via `env_file`.

Create the shared folders if needed:

```bash
mkdir -p airflow-batch-processor/{config,data}
cp airflow-batch-processor/.env.example airflow-batch-processor/.env
```

Optional (DockerOperator use): mount Docker socket to the scheduler for launching sibling containers:

```yaml
# airflow-scheduler:
#   volumes:
#     - /var/run/docker.sock:/var/run/docker.sock
```

## Environment
- Copy `.env.example` to `.env` and set `UPSTAGE_API_KEY`

## Structure
- `dags/` — DAGs (see `batch_processor_dag.py`)
- `src/` — Python modules (API clients, processors)
- `config/` — YAML configs for runtime
- `plugins/` — (optional) Airflow plugins/operators
- `docker/` — Dockerfile and docker-compose

## Notes
- The Upstage client is a scaffold; adapt endpoint/routes and auth to your account.
- For AWS Batch, add a separate DAG/operator and boto3 integration.
