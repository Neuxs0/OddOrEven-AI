import time
import random
import numpy as np

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

neuronID = 0
layerID = 0

class Neuron:
    def __init__(self):
        global neuronID
        neuronID += 1
        t = time.localtime()

        self.bias = random.uniform(-1.00, 1.00)
        self.id = neuronID
        self.output = None

        self.creationTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
        self.updatedTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
    
    def passData(self, inputs, weights):
        inputs = xp.array(inputs)
        weighted_sum = xp.dot(weights, inputs) + self.bias
        self.output = 1 / (1 + xp.exp(-weighted_sum))
        return self.output
    
    def getID(self):
        return self.id
    
    def getBias(self):
        return self.bias
    
    def setBias(self, newBias):
        t = time.localtime()
        self.bias = newBias
        self.updatedTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"

class NetworkLayer:
    def __init__(self, neuronCount, inputConnections):
        global layerID
        layerID += 1

        t = time.localtime()

        self.id = layerID
        self.neuronCount = neuronCount
        self.neurons = [Neuron() for _ in range(neuronCount)]
        self.weights = xp.random.uniform(-1.00, 1.00, (neuronCount, inputConnections))

        self.creationTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
        self.updatedTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
    
    def getNeurons(self):
        return self.neurons
    
    def getNeuron(self, id):
        return self.neurons[id]
    
    def getNeuronCount(self):
        return self.neuronCount
    
    def getID(self):
        return self.id

class TraditionalNeuralNetwork:
    def __init__(self, inputLayerNeuronCount, outputLayerNeuronCount, hiddenLayerCount, hiddenLayerNeuronCounts=None):
        t = time.localtime()

        if hiddenLayerNeuronCounts is None:
            hiddenLayerNeuronCounts = [16] * hiddenLayerCount
        elif len(hiddenLayerNeuronCounts) < hiddenLayerCount:
            hiddenLayerNeuronCounts.extend([hiddenLayerNeuronCounts[-1]] * (hiddenLayerCount - len(hiddenLayerNeuronCounts)))
        elif len(hiddenLayerNeuronCounts) > hiddenLayerCount:
            hiddenLayerNeuronCounts = hiddenLayerNeuronCounts[:hiddenLayerCount]

        self.inputLayer = NetworkLayer(inputLayerNeuronCount, inputLayerNeuronCount)
        self.hiddenLayers = [
            NetworkLayer(
                hiddenLayerNeuronCounts[i],
                inputLayerNeuronCount if i == 0 else hiddenLayerNeuronCounts[i - 1]
            ) for i in range(hiddenLayerCount)
        ]
        self.outputLayer = NetworkLayer(outputLayerNeuronCount, hiddenLayerNeuronCounts[-1])

        self.creationTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
        self.updatedTime = f"{t.tm_hour}:{t.tm_min} {t.tm_mday}/{t.tm_mon}/{t.tm_year}"
    
    def forward(self, inputs):
        inputs = xp.array(inputs)
        input_outputs = []
        for neuron, weights in zip(self.inputLayer.getNeurons(), self.inputLayer.weights):
            input_outputs.append(neuron.passData(inputs, weights))
        
        for layer in self.hiddenLayers:
            new_inputs = []
            for neuron, weights in zip(layer.getNeurons(), layer.weights):
                new_inputs.append(neuron.passData(input_outputs, weights))
            input_outputs = new_inputs
        
        output_outputs = []
        for neuron, weights in zip(self.outputLayer.getNeurons(), self.outputLayer.weights):
            output_outputs.append(float(neuron.passData(input_outputs, weights)))
        
        return output_outputs
    
    @classmethod
    def from_dict(cls, data):
        inputLayerData = data['inputLayer']
        hiddenLayersData = data['hiddenLayers']
        outputLayerData = data['outputLayer']

        network = cls(
            inputLayerNeuronCount=len(inputLayerData['neurons']),
            outputLayerNeuronCount=len(outputLayerData['neurons']),
            hiddenLayerCount=len(hiddenLayersData),
            hiddenLayerNeuronCounts=[len(layer['neurons']) for layer in hiddenLayersData]
        )

        network.inputLayer = NetworkLayer(
            neuronCount=len(inputLayerData['neurons']),
            inputConnections=len(inputLayerData['weights'][0])
        )
        network.inputLayer.weights = xp.array(inputLayerData['weights'])
        for i, bias in enumerate(inputLayerData['biases']):
            network.inputLayer.neurons[i].setBias(bias)

        network.hiddenLayers = []
        for layerData in hiddenLayersData:
            layer = NetworkLayer(
                neuronCount=len(layerData['neurons']),
                inputConnections=len(layerData['weights'][0])
            )
            layer.weights = xp.array(layerData['weights'])
            for i, bias in enumerate(layerData['biases']):
                layer.neurons[i].setBias(bias)
            network.hiddenLayers.append(layer)

        network.outputLayer = NetworkLayer(
            neuronCount=len(outputLayerData['neurons']),
            inputConnections=len(outputLayerData['weights'][0])
        )
        network.outputLayer.weights = xp.array(outputLayerData['weights'])
        for i, bias in enumerate(outputLayerData['biases']):
            network.outputLayer.neurons[i].setBias(bias)

        return network

    def to_dict(self):
        return {
            'inputLayer': {
                'neurons': [neuron.getID() for neuron in self.inputLayer.getNeurons()],
                'biases': [neuron.getBias() for neuron in self.inputLayer.getNeurons()],
                'weights': self.inputLayer.weights.tolist()
            },
            'hiddenLayers': [
                {
                    'neurons': [neuron.getID() for neuron in layer.getNeurons()],
                    'biases': [neuron.getBias() for neuron in layer.getNeurons()],
                    'weights': layer.weights.tolist()
                }
                for layer in self.hiddenLayers
            ],
            'outputLayer': {
                'neurons': [neuron.getID() for neuron in self.outputLayer.getNeurons()],
                'biases': [neuron.getBias() for neuron in self.outputLayer.getNeurons()],
                'weights': self.outputLayer.weights.tolist()
            }
        }
