COMPOSE ?= docker compose
COMPOSE_GPU ?= $(COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml --compatibility

.PHONY: up up-core up-gpu up-gpu-core down logs ps init restart-web restart-scheduler clean

up:
	$(COMPOSE) up -d

up-core:
	$(COMPOSE) up -d airflow-webserver airflow-scheduler postgres

up-gpu:
	$(COMPOSE_GPU) up -d

up-gpu-core:
	$(COMPOSE_GPU) up -d airflow-webserver airflow-scheduler postgres

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f airflow-webserver airflow-scheduler

ps:
	$(COMPOSE) ps

init:
	$(COMPOSE) run --rm airflow-init

restart-web:
	$(COMPOSE) restart airflow-webserver

restart-scheduler:
	$(COMPOSE) restart airflow-scheduler

clean:
	$(COMPOSE) down -v
