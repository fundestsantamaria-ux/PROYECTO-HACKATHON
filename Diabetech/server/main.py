from nodeC.server import server
from nodex.client import client
from coordination import coordinate
from utils import save_metrics, unificar_metricas_csv
import os
import time
import sys

#################### MAIN PARAMETERS ###########################

IPS = [
    "192.168.0.116",  # PC 1
    "192.168.0.116",  # PC 2
    "192.168.0.116",  # PC 3
    "192.168.0.116"   # PC 4
]

# Puertos del HOST (externos)
PORTS = os.getenv("PORTS", "5000,5001,5002,5003").split(",")

BIND_PORT = int(os.getenv("BIND_PORT", 5000))
DOCKER_PORT = int(os.getenv("DOCKER_PORT", 5000))

# MUY IMPORTANTE: IDs 1 y 2 (no 0 y 1)
NODE_ID = int(os.getenv("NODE_ID"))       # 1 en la primera PC, 2 en la segunda
MODE = int(os.getenv("MODE", 1))

# Construimos las direcciones ip:puerto de TODOS los nodos
NETWORK_ADDRESSES = [
    f"{ip}:{port.strip()}" for ip, port in zip(IPS, PORTS)
]

# Dirección de ESTE nodo (NODE_ID es 1-based)
DOCKER_ADDRESS = NETWORK_ADDRESSES[NODE_ID - 1]

# Peers = todos menos yo
PEERS = [
    addr for i, addr in enumerate(NETWORK_ADDRESSES, start=1)
    if i != NODE_ID
]

ROUNDS = 3
SUB_ROUNDS = 3

if MODE == 0 and ROUNDS > 1:
    print("Centralized modo solo puede tener ROUNDS=1", flush=True)
    sys.exit(1)

NCLIENTS = len(NETWORK_ADDRESSES)
NODE_DIR = f"nodo{NODE_ID}"

PARAMS = {
    "hidden_layers": [(32, 0.4), (16, 0.3)],
    "activation": "relu",
    "optimizer": "adam"
}

#################################################################

if __name__ == '__main__':

    # Espera inicial para que Docker estabilice la red
    time.sleep(5)

    init = time.time()
    for round in range(ROUNDS): 
        print(f"\n>>> INICIO RONDA {round} <<<", flush=True)

        if MODE == 1:
            print("Semi-Descentrilized Modo Configurado", flush=True)
            # 1. COORDINACIÓN
            id_nodeserver = coordinate(PEERS, round)
            server_ip = NETWORK_ADDRESSES[int(id_nodeserver) - 1]
            port_ip = int(server_ip.split(':')[1])
            nodo_ip = server_ip.split(':')[0]
            print(f"Selected node ID: {id_nodeserver} address: {nodo_ip}:{port_ip}", flush=True)
        else:
            print("Centralizado Modo Configurado", flush=True)
            id_nodeserver = 1
            server_ip = NETWORK_ADDRESSES[int(id_nodeserver) - 1]
            port_ip = int(server_ip.split(':')[1])
            nodo_ip = server_ip.split(':')[0]

        time.sleep(5)  # Pequeña espera antes de iniciar la siguiente fase

        f1scores: list[dict[str, float]] = []
        accs: list[dict[str, float]] = []
        get_times: list[dict[str, float]] = []
        send_times: list[dict[str, float]] = []
        
        # 2. ENTRENAMIENTO
        if server_ip == DOCKER_ADDRESS:
            # Soy el servidor
            print(f"[MAIN] Iniciando Servidor FL (Esperando {NCLIENTS - 1} clientes)...", flush=True)
            server(BIND_PORT, SUB_ROUNDS + 1, NCLIENTS - 1, PARAMS, f1scores, accs, get_times, send_times)
        else:
            time.sleep(5)
            print(f"[MAIN] Conectando al servidor {nodo_ip}:{port_ip}...", flush=True)
            client(nodo_ip, port_ip, SUB_ROUNDS + 1)

        save_metrics(f1scores, accs, get_times, send_times, NODE_ID)

        print(f"Round {round} completed!!!", flush=True)
        print("=" * 60, '\n', flush=True)

        unificar_metricas_csv(NODE_ID)

    end = time.time()
    
    print("=" * 60, '\n')
    print("Federated training completed successfully!!!", flush=True)
    print(f"Training Time: {end - init}s")
