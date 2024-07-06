import json
import random

def generate_numbers_dataset(num_entries, filename):
    data = []
    for _ in range(num_entries):
        number = random.randint(1, 4294967295)
        is_even = (number % 2 == 0)
        data.append({"number": number, "is_even": is_even})
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

generate_numbers_dataset(1000, '.\\data\\datasets\\numbers.json')

print("Dataset generated and saved to Numbers.json")