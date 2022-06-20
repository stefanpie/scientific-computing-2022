from collections import Counter 

with open("./input_sample.txt", "r") as f:
    data = f.readlines()[0]

data = list(data)
count = Counter(data)

solution = f'{count["A"]} {count["C"]} {count["G"]} {count["T"]}'
print(solution)
