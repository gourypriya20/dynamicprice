import csv

def read_csv_to_dict(file_path):
    data_dict = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data_dict.append(row)
    return data_dict

file_path = "dynamic pricing\car.csv"
data = read_csv_to_dict(file_path)
for i in data:
    print(i)