#!/usr/bin/env bash
set -e

COMPOSE_DIR="$(cd "$(dirname "$0")/docker" && pwd)"

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

# Comprobar si Docker está corriendo
if ! docker info &>/dev/null; then
    start_docker_daemon
fi

echo "Docker OK: $(docker --version)"
echo ""

cd "$COMPOSE_DIR"
docker compose up --build "$@"
