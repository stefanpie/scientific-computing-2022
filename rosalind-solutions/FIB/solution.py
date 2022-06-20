
with open("./input.txt", "r") as f:
    n, k = [int(i)for i in f.readlines()[0].split(" ")]



population = [1,1]

print(n, k)

for month in range(n):
    population.append(population[-1] + population[-2]*k)

print(population[n-1])
