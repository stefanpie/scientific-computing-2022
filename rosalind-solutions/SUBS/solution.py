import re

with open("./input.txt", "r") as f:
    sequence, sub = [l.strip() for l in f.readlines()][0:2]

print(sequence)
print(sub)

indexes = []

for i in range(len(sequence)-len(sub)+1):
    if sub == sequence[i:i+len(sub)]:
        indexes.append(i+1)

solution = " ".join([str(i) for i in indexes])
print(solution)