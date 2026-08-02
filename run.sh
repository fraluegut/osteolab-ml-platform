#!/usr/bin/env bash
set -e

COMPOSE_DIR="$(cd "$(dirname "$0")/mlops/docker" && pwd)"

start_docker_daemon() {
    echo "Docker daemon no está corriendo. Intentando arrancarlo..."

    if command -v systemctl &>/dev/null && systemctl is-active docker &>/dev/null; then
        return 0
    fi

    if command -v service &>/dev/null; then
        sudo service docker start 2>/dev/null && return 0
    fi

    # WSL2 o entorno sin systemd: arrancar dockerd directamente
    sudo dockerd &>/tmp/dockerd.log &
    DOCKERD_PID=$!

    echo "Esperando a que Docker arranque..."
    for i in $(seq 1 30); do
        if docker info &>/dev/null; then
            echo "Docker arrancado correctamente (PID $DOCKERD_PID)"
            return 0
        fi
        sleep 1
    done

    echo "ERROR: No se pudo arrancar Docker después de 30s."
    echo "Revisa /tmp/dockerd.log para más detalles."
    exit 1
}

show_urls() {
    echo ""
    echo "=== Servicios disponibles ==="
    echo "App + Web (landing/demo):  http://localhost:8017"
    echo "MLflow (tracking):         http://localhost:5000"
    echo "Streamlit UI:              http://localhost:8501"
    echo "=============================="
}

# Comprobar si Docker está corriendo
if ! docker info &>/dev/null; then
    start_docker_daemon
fi

echo "Docker OK: $(docker --version)"
echo ""

MODE="${1:-}"
if [[ -z "$MODE" ]]; then
    echo "¿Cómo querés levantar el entorno?"
    echo "  1) Normal (usar imágenes ya construidas)"
    echo "  2) Reinstalar todo (rebuild de imágenes desde cero, sin cache)"
    read -rp "Elegí una opción [1/2]: " MODE
fi

cd "$COMPOSE_DIR"

case "$MODE" in
    2|reinstall|rebuild)
        echo "Reinstalando todo (rebuild sin cache)..."
        docker compose build --no-cache
        docker compose up -d
        ;;
    *)
        echo "Levantando entorno normal..."
        docker compose up -d
        ;;
esac

echo ""
echo "Esperando a que los servicios respondan..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8017/ &>/dev/null; then
        break
    fi
    sleep 1
done

show_urls

echo ""
echo "Para ver logs: docker compose -f $COMPOSE_DIR/docker-compose.yml logs -f"
echo "Para parar:    docker compose -f $COMPOSE_DIR/docker-compose.yml down"
