start_service:
	@docker compose -p ddi_chat -f deployment/docker-compose.yaml up -d

stop_service:
	@docker compose -p ddi_chat -f deployment/docker-compose.yaml down

build:
	@docker build -t ddi-chat:latest .

start_dev:
	@docker compose -p ddi_chat -f deployment/docker-compose.yaml up -d vector_database
	@streamlit run main.py --server.fileWatcherType watchdog
