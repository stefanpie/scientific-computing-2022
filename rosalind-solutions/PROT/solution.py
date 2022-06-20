import re
from pprint import pprint as pp

with open("./rna_codon_table.txt", "r") as f:
    translation_file = f.read()

translation_table = {}
for match in re.findall(r"([a-zA-Z]+) ([a-zA-Z]+)", translation_file):
    translation_table[match[0]] = match[1]

pp(translation_table)



with open("./input.txt", "r") as f:
    data = f.readlines()[0].strip()

n = 3
data_split = [data[i:i + n] for i in range(0, len(data), n)]

data_translated = [translation_table[chunk] for chunk in data_split]
data_translated = "".join([i for i in data_translated if i != "Stop"])


with open("./output.txt", "w") as f:
    f.write(data_translated)
