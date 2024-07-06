import json
from nn import TraditionalNeuralNetwork

MAX_NUMBER = 4294967295 # Max due to the limitations of 32 bit unsigned integers

def load_trained_network(filepath):
    with open(filepath, 'r') as file:
        network_data = json.load(file)
    
    network = TraditionalNeuralNetwork.from_dict(network_data)
    return network

def main():
    try:
        network = load_trained_network('.\\data\\model1.json')
    except:
        exit("Exitting... Something went wrong while getting the trained ai model file...")

    while True:
        user_input = input("Enter a number or type /bye to exit.\n> ").strip()
        
        if user_input.lower() == '/bye':
            print("Exiting the program. Goodbye!")
            break

        try:
            number = int(user_input)
        except:
            print("Invalid input: Please enter a valid number.")
            continue
        
        if number > MAX_NUMBER:
            print(f"Invalid input: Number exceeds the maximum allowed value of {MAX_NUMBER}.")
            continue
        
        binary_rep = [int(x) for x in f"{number:032b}"]
        try:
            output = network.forward(binary_rep)
            is_even = output[0] > output[1]
            print(f"\nNetwork Output: {'even' if is_even else 'odd'}.\nIs even certaincy: {output[0]}%\nIs odd certaincy: {output[1]}%\n")
        except ValueError as e:
            print(f"Error processing the number: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
