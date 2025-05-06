import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from model import create_model, save_model, load_model

np.random.seed(42)
tf.random.set_seed(42)

def load_client_weights(client_ids):
    client_weights = []
    
    for client_id in client_ids:
        weights_file = f"pesos_cliente_{client_id}.pkl"
        if os.path.exists(weights_file):
            with open(weights_file, 'rb') as f:
                weights = pickle.load(f)
            client_weights.append(weights)
            print(f"Pesos cargados para el cliente {client_id}")
        else:
            print(f"¡Advertencia! No se encontraron pesos para el cliente {client_id}")
    
    return client_weights

def load_test_data():
    test_data_file = "test_data/test_data.pkl"
    with open(test_data_file, 'rb') as f:
        x_test, y_test = pickle.load(f)
    
    return x_test, y_test

def fedavg(client_weights):
    if not client_weights:
        return None
    
    avg_weights = [np.zeros_like(w) for w in client_weights[0]]
    
    for weights in client_weights:
        for i, w in enumerate(weights):
            avg_weights[i] += w
    
    n_clients = len(client_weights)
    avg_weights = [w / n_clients for w in avg_weights]
    
    return avg_weights

def fedprox(client_weights, mu=0.01, global_weights=None):
    if not client_weights:
        return None
    
    if global_weights is None:
        return fedavg(client_weights)
    
    prox_weights = [np.zeros_like(w) for w in client_weights[0]]
    
    for weights in client_weights:
        for i, w in enumerate(weights):
            regularized_w = w - mu * (w - global_weights[i])
            prox_weights[i] += regularized_w
    
    n_clients = len(client_weights)
    prox_weights = [w / n_clients for w in prox_weights]
    
    return prox_weights

def fedmed(client_weights):
    if not client_weights:
        return None
    
    med_weights = []
    
    for i in range(len(client_weights[0])):
        layer_weights = [weights[i] for weights in client_weights]
        
        stacked = np.stack(layer_weights, axis=0)
        median_weights = np.median(stacked, axis=0)
        
        med_weights.append(median_weights)
    
    return med_weights

def evaluate_model(model, x_test, y_test, title="Modelo Global"):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Pérdida en prueba: {test_loss:.4f}")
    print(f"Precisión en prueba: {test_accuracy:.4f}")
    return test_accuracy

def main():
    client_ids = [1, 2, 3, 4]
    
    client_weights = load_client_weights(client_ids)
    
    if not client_weights:
        print("No se encontraron pesos de clientes. Saliendo...")
        return
    
    x_test, y_test = load_test_data()
    print(f"Datos de prueba cargados: {len(x_test)} muestras")
    
    base_model = create_model()
    
    print("\n=== Agregación con FedAvg ===")
    fedavg_weights = fedavg(client_weights)
    base_model.set_weights(fedavg_weights)
    fedavg_accuracy = evaluate_model(base_model, x_test, y_test, "FedAvg")
    save_model(base_model, "modelo_global_fedavg.h5")
    
    print("\n=== Agregación con FedProx ===")
    fedprox_weights = fedprox(client_weights, mu=0.01, global_weights=fedavg_weights)
    base_model.set_weights(fedprox_weights)
    fedprox_accuracy = evaluate_model(base_model, x_test, y_test, "FedProx")
    save_model(base_model, "modelo_global_fedprox.h5")
    
    print("\n=== Agregación con FedMed ===")
    fedmed_weights = fedmed(client_weights)
    base_model.set_weights(fedmed_weights)
    fedmed_accuracy = evaluate_model(base_model, x_test, y_test, "FedMed")
    save_model(base_model, "modelo_global_fedmed.h5")
    
    print("\n=== Comparación de Métodos ===")
    methods = ["FedAvg", "FedProx", "FedMed"]
    accuracies = [fedavg_accuracy, fedprox_accuracy, fedmed_accuracy]
    
    plt.figure(figsize=(10, 6))
    plt.bar(methods, accuracies, color=['blue', 'green', 'red'])
    plt.title('Comparación de Precisión por Método de Agregación')
    plt.xlabel('Método')
    plt.ylabel('Precisión')
    plt.ylim(0, 1.0)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.savefig("comparacion_metodos.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    best_idx = np.argmax(accuracies)
    best_method = methods[best_idx]
    print(f"\nEl mejor método de agregación es: {best_method} con precisión: {accuracies[best_idx]:.4f}")
    
    if best_method == "FedAvg":
        best_weights = fedavg_weights
    elif best_method == "FedProx":
        best_weights = fedprox_weights
    else:
        best_weights = fedmed_weights
    
    base_model.set_weights(best_weights)
    save_model(base_model, "modelo_global_final.h5")
    print(f"Modelo global final guardado como: modelo_global_final.h5")

if __name__ == "__main__":
    main()