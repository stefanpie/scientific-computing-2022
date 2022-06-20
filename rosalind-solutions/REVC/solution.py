with open("./input.txt", "r") as f:
    data = f.readlines()[0]

dict_base_paris = {"A": "T",
                    "T": "A",
                    "C": "G",
                    "G": "C"}

solution = "".join([dict_base_paris[c] for c in data])[::-1]
print(solution)


with open("./output.txt", "w") as f:
    f.write(solution)