import json
import numpy as np
from datetime import datetime, timedelta
import sys
import platform
import cupy as cp
import subprocess
import time
import psutil
import os

try:
    xp = cp
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
    output = output.decode('utf-8').strip()
    gpu_name, vram, gpu_temp = output.split(',')
    print(f"Using GPU: {gpu_name} | VRAM: {vram} MB")
except:
    xp = np
    print(f"NVIDIA GPU not found! Using CPU: {platform.processor()}")

from nn import TraditionalNeuralNetwork

def load_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    inputs = []
    targets = []

    for item in data:
        binary_rep = [int(x) for x in f"{item['number']:032b}"]
        inputs.append(binary_rep)
        targets.append([1, 0] if item['is_even'] else [0, 1])

    return xp.array(inputs, dtype=xp.float32), xp.array(targets, dtype=xp.float32)

def save_trained_network(network, filepath):
    with open(filepath, 'w') as file:
        json.dump(network.to_dict(), file)

def get_system_info():
    cpu = platform.processor()
    ram = psutil.virtual_memory()
    storage = psutil.disk_usage('/')
    drives = psutil.disk_partitions()
    
    system_info = {
        "CPU": cpu,
        "RAM": f"{ram.total / (1024**3):.2f} GB",
        "Storage": {
            "Total": f"{storage.total / (1024**3):.2f} GB",
            "Free": f"{storage.free / (1024**3):.2f} GB"
        },
        "Drives": [{"Device": d.device, "Mountpoint": d.mountpoint, "FStype": d.fstype} for d in drives],
        "OS": f"{platform.system()} {platform.release()}",
        "Python": platform.python_version()
    }
    
    if gpu_name:
        system_info["GPU"] = {
            "Name": gpu_name,
            "VRAM": f"{vram} MB"
        }
    
    return system_info

def log_model_metadata(network, loss, dataset_size, epochs, filepath, log_filepath, total_train_time, avg_temp, avg_epoch_time, avg_cpu_usage, avg_gpu_usage, avg_ram_usage):
    now = datetime.now()
    time_created = now.strftime("%I:%M%p %B %dth %Y | %H:%M %d/%m/%Y | %s")
    
    system_info = get_system_info()
    
    metadata = f"""
- Model:
Input Neurons: {network.inputLayerNeuronCount}
Output Neurons: {network.outputLayerNeuronCount}
Hidden Layers: {network.hiddenLayerCount}

Hidden Layer Neurons:
""" + '\n'.join([f"- {count}" for count in network.hiddenLayerNeuronCounts]) + f"""

Loss: {loss}
Dataset: {dataset_size} Numbers
Epochs Trained: {epochs}

Time took training: {total_train_time}
Time Created: {time_created}
File: {filepath}

Training Statistics:
- Average Temperature: {avg_temp:.2f}°C
- Average Time per Epoch: {avg_epoch_time:.2f}s
- Average CPU Usage: {avg_cpu_usage:.2f}%
- Average GPU Usage: {avg_gpu_usage:.2f}%
- Average RAM Usage: {avg_ram_usage:.2f}%

System Information:
- CPU: {system_info['CPU']}
- RAM: {system_info['RAM']}
- Storage:
  - Total: {system_info['Storage']['Total']}
  - Free: {system_info['Storage']['Free']}
- Drives:
""" + '\n'.join([f"  - {d['Device']} ({d['FStype']}) mounted at {d['Mountpoint']}" for d in system_info['Drives']]) + f"""
- OS: {system_info['OS']}
- Python: {system_info['Python']}
"""

    if 'GPU' in system_info:
        metadata += f"""- GPU: {system_info['GPU']['Name']}
- VRAM: {system_info['GPU']['VRAM']}
"""

    with open(log_filepath, 'a') as log_file:
        log_file.write(metadata)

def train(network, learning_rate=0.01, epochs=200, batch_size=32):
    start_time = time.time()
    total_time = 0
    total_temp = 0
    total_cpu_usage = 0
    total_gpu_usage = 0
    total_ram_usage = 0

    for epoch in range(epochs):
        # os.system('python .\\src\\generate_dataset.py') # Enable for a new dataset every epoch
        inputs, targets = load_data('.\\data\\datasets\\numbers.json')

        epoch_start_time = time.time()
        total_loss = 0

        if epoch >= 30:
            learning_rate *= 0.98

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_targets = targets[i:i+batch_size]
            
            for x, y in zip(batch_inputs, batch_targets):
                output = xp.array(network.forward(x))
                deltas = y - output
                total_loss += xp.mean((y - output)**2)

                for j, neuron in enumerate(network.outputLayer.getNeurons()):
                    neuron.setBias(float(neuron.getBias() + learning_rate * deltas[j]))
                    for k in range(len(network.outputLayer.weights[j])):
                        network.outputLayer.weights[j][k] += float(learning_rate * deltas[j] * network.hiddenLayers[-1].neurons[k].output)
                
                for l in reversed(range(len(network.hiddenLayers))):
                    layer = network.hiddenLayers[l]
                    next_layer = network.outputLayer if l == len(network.hiddenLayers) - 1 else network.hiddenLayers[l + 1]

                    for j, neuron in enumerate(layer.getNeurons()):
                        error = xp.sum(xp.array([next_layer.weights[k][j] * deltas[k] for k in range(len(deltas))]))
                        delta = float(error * neuron.output * (1 - neuron.output))
                        neuron.setBias(float(neuron.getBias() + learning_rate * delta))

                        for k in range(len(layer.weights[j])):
                            prev_output = network.inputLayer.getNeurons()[k].output if l == 0 else network.hiddenLayers[l - 1].neurons[k].output
                            layer.weights[j][k] += float(learning_rate * delta * prev_output)

                    deltas = xp.array([float(error * neuron.output * (1 - neuron.output)) for neuron in layer.getNeurons()])

        avg_loss = total_loss / len(inputs)
        epoch_time = time.time() - epoch_start_time
        total_time += epoch_time
        avg_epoch_time = total_time / (epoch + 1)
        eta = avg_epoch_time * (epochs - epoch - 1)

        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        if gpu_name:
            gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"])
            gpu_temp, gpu_usage = map(float, gpu_info.decode('utf-8').strip().split(','))
        else:
            gpu_usage = 0
            gpu_temp = 0

        total_temp += gpu_temp
        total_cpu_usage += cpu_usage
        total_gpu_usage += gpu_usage
        total_ram_usage += ram_usage

        if xp == cp: 
            temp = f"GPU Temp: {gpu_temp:.1f}°C | {(gpu_temp*9/5)+32:.1f}°F"
        else: 
            temp = ""

        print("---------------------------------------------")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Loss: {avg_loss:.6f}")
        print(f"{temp}")
        print(f"CPU Usage: {cpu_usage:.2f}% | GPU Usage: {gpu_usage:.2f}% | RAM Usage: {ram_usage:.2f}%")
        print(f"Epoch Time: {epoch_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print(f"ETA: {timedelta(seconds=int(eta))}")
        print(f"Learning Rate: {learning_rate:.6f}")
        print("---------------------------------------------")
    
    avg_temp = total_temp / epochs
    avg_cpu_usage = total_cpu_usage / epochs
    avg_gpu_usage = total_gpu_usage / epochs
    avg_ram_usage = total_ram_usage / epochs
    
    return avg_loss, total_time, avg_temp, avg_epoch_time, avg_cpu_usage, avg_gpu_usage, avg_ram_usage

def main(inputLayerNeuronCount=32, outputLayerNeuronCount=2, hiddenLayerCount=4, hiddenLayerNeuronCounts=[64, 48, 32, 16], learning_rate=0.01, epochs=200, batch_size=32):
    network = TraditionalNeuralNetwork(
        inputLayerNeuronCount,
        outputLayerNeuronCount,
        hiddenLayerCount,
        hiddenLayerNeuronCounts
    )
    loss, total_training_time, avg_temp, avg_epoch_time, avg_cpu_usage, avg_gpu_usage, avg_ram_usage = train(network, learning_rate, epochs, batch_size)
    save_trained_network(network, '.\\data\\model.json')
    log_model_metadata(network, loss, 1000, epochs, '.\\data\\model.json', '.\\data\\models.txt', total_training_time, avg_temp, avg_epoch_time, avg_cpu_usage, avg_gpu_usage, avg_ram_usage)
    print("---------------------------------------------")
    print(f"Total training time: {timedelta(seconds=int(total_training_time))}")
    print(f"Average time per epoch: {avg_epoch_time:.2f}s")
    print(f"Average GPU temperature: {avg_temp:.2f}°C")
    print(f"Average CPU usage: {avg_cpu_usage:.2f}%")
    print(f"Average GPU usage: {avg_gpu_usage:.2f}%")
    print(f"Average RAM usage: {avg_ram_usage:.2f}%")
    print("---------------------------------------------")

if __name__ == "__main__":
    main(32, 2, 4, [64, 48, 32, 16], 0.01, 200, 32)