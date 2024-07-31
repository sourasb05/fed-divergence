import json

# Path to the JSON file
file_path = '/proj/bhuyan24/fed-divergence/data/data/100_3/test/cifa_test.json'
clients = []
train_data = {}
test_data = {}
num_samples = []
# Open and read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)
    clients.extend(data['users'])
    train_data.update(data['user_data'])
    num_samples.extend(data['num_samples'])

# Print the loaded data
print(clients)
print(num_samples)