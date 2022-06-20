from pprint import pprint as pp

with open("./input.txt", "r") as f:
    file_lines = f.readlines()
    file_lines = [l.strip() for l in file_lines]
  
dna_strings = {}
current_id = None
for i, line in enumerate(file_lines):
    if line[0] == ">":
        dna_strings[line[1:]] = ""
        current_id = line[1:]
    else:
        dna_strings[current_id] += line

pp(dna_strings)

gc_count = {k: (v.count("G") + v.count("C")) *100 / len(v)
            for k, v in dna_strings.items()}

max_key = max(gc_count, key=gc_count.get)
print(max_key)
print(gc_count[max_key])
