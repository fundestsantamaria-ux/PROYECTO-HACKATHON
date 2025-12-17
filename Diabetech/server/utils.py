import csv
import os
import random

def unificar_metricas_csv(node_id):
    """
    Une:
    - CSV principal de modelos
    - TXT de f1scores
    - TXT de accuracies
    - TXT de get_times
    - TXT de send_times

    Todos con formato:
        node_id,value

    Se ignora el nodo actual (node_id).
    Archivos vacíos se omiten automáticamente.
    """
    csv_filename = f"models_path_{node_id}.csv"
    txt_files = {
        "f1": f"f1scores{node_id}.txt",
        "acc": f"accs{node_id}.txt",
        "get_time": f"get_times{node_id}.txt",
        "send_time": f"send_times{node_id}.txt"
    }
    output_filename = f"full_metrics_node_{node_id}.csv"

    # Validar que exista CSV principal
    if not os.path.exists(csv_filename):
        print(f"[!] Falta el archivo: {csv_filename}")
        return

    try:
        # CSV principal
        with open(csv_filename, "r", encoding="utf-8") as f_csv:
            reader = csv.reader(f_csv)
            header_csv = next(reader)
            rows_csv = list(reader)

        # Cargar TXT y agrupar por nodo, ignorando archivos vacíos
        metrics_data = {}
        for key, filename in txt_files.items():
            metrics_data[key] = {}
            if not os.path.exists(filename) or os.path.getsize(filename) == 0:
                print(f"[!] Archivo vacío o no existe, se ignora: {filename}")
                continue
            with open(filename, "r", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if len(row) < 2:
                        continue  # Ignorar filas mal formateadas
                    node, value = row
                    if int(node) == int(node_id):
                        continue
                    if node not in metrics_data[key]:
                        metrics_data[key][node] = []
                    metrics_data[key][node].append(value)

        # Construir headers dinámicos
        full_header = header_csv
        for key in txt_files.keys():
            for node in sorted(metrics_data[key].keys(), key=int):
                full_header.append(f"{key}_node_{node}")

        # Construir filas finales
        final_rows = []
        num_rows = len(rows_csv)
        for i in range(num_rows):
            row = list(rows_csv[i])  # copiar fila CSV
            for key in txt_files.keys():
                for node in sorted(metrics_data[key].keys(), key=int):
                    values = metrics_data[key][node]
                    # Usar valor si existe, si no '', para filas faltantes
                    row.append(values[i] if i < len(values) else '')
            final_rows.append(row)

        # Escribir archivo final
        with open(output_filename, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(full_header)
            writer.writerows(final_rows)

        print(f"[✓] Archivo generado: {output_filename}")
        print(f"    Columnas totales: {len(full_header)}")

    except Exception as e:
        print(f"[!] Error procesando nodo {node_id}: {e}")


def save_metrics(f1scores, accs, get_times, send_times, NODE_ID):
    print(f1scores)
    with open(f'f1scores{NODE_ID}.txt', "a+") as f:
        for d in f1scores:               # cada d es un diccionario
            for node, scores in d.items():
                if isinstance(scores, list):
                    line = ",".join(str(x) for x in scores)
                else:
                    line = str(scores)
                f.write(f"{node},{line}\n")
    
    print(accs)
    with open(f'accs{NODE_ID}.txt', "a+") as f:
        for d in accs:
            for node, scores in d.items():
                if isinstance(scores, list):
                    line = ",".join(str(x) for x in scores)
                else:
                    line = str(scores)
                f.write(f"{node},{line}\n")
    
    print(get_times)
    with open(f'get_times{NODE_ID}.txt', "a+") as f:
        for d in get_times:
            for node, scores in d.items():
                if isinstance(scores, list):
                    line = ",".join(str(x) for x in scores)
                else:
                    line = str(scores)
                f.write(f"{node},{line}\n")
    
    print(send_times)
    with open(f'send_times{NODE_ID}.txt', "a+") as f:
        for d in send_times:
            for node, scores in d.items():
                if isinstance(scores, list):
                    line = ",".join(str(x) for x in scores)
                else:
                    line = str(scores)
                f.write(f"{node},{line}\n")


def checkConvergence(scores: list[dict[str, float]], patience: int, threshold: float = 0.01) -> bool:
    if len(scores) < patience:
        return False

    old = scores[-patience]   # dict: nodo → valor
    new = scores[-1]          # dict: nodo → valor

    # Asegurar que ambos diccionarios tienen mismas keys (por seguridad)
    common_nodes = old.keys() & new.keys()

    for node in common_nodes:
        diff = new[node] - old[node]

        if abs(diff) > threshold:
            return False

    return True

def select_leader(nodes, round):
    """
    Selecciona un nodo líder basado en probabilidad ponderada por sus capacidades.
    Usa la ronda como semilla para asegurar que todos los nodos elijan al mismo ganador.
    """
    
    # 1. Calcular el puntaje (score) para cada nodo
    scores = []
    for x in nodes:
        score = (
            0.5 * (0.5 * x['net_up'] + 0.5 * x['net_down']) +
            0.3 * x['ram'] +
            0.35 * x['cpu_mhz'] +
            0.2 * int(x['gpu']) +
            0.1 * (1 / int(x['id']))
        )
        scores.append(score)

    # 2. Configurar la semilla compartida
    random.seed(round)

    # 3. Selección ponderada (Weighted Choice)
    ganador = random.choices(nodes, weights=scores, k=1)[0]
    
    random.seed(None) 
    
    return ganador