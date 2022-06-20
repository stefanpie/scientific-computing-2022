with open("./input.txt", "r") as f:
    s1, s2 = [l.strip() for l in f.readlines()][0:2]

print(sum([c1 != c2 for c1, c2 in zip(s1,s2)]))