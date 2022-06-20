from itertools import combinations, permutations, product

with open("./input.txt", "r") as f:
    k, m, n = [int(i)for i in f.readlines()[0].split(" ")]


k_organisms = {"alleles": ["Y", "Y"],
               "count": k}

m_organisms = {"alleles": ["Y", "y"],
               "count": m}

n_organisms = {"alleles": ["y", "y"],
               "count": n}

population = [*[k_organisms["alleles"]]*k_organisms["count"],
              *[m_organisms["alleles"]]*m_organisms["count"],
              *[n_organisms["alleles"]]*n_organisms["count"]]
# population = {"k": k_organisms, "m": m_organisms, "n": n_organisms}
# print(population)

combos = list(combinations(population, 2))

all_outcomes = []

for combo in combos:
    outcome = list(product(combo[0], combo[1]))
    for o in outcome:
        all_outcomes.append("".join(o))
    # print(outcome)

# print(all_outcomes)

prob = len(list(filter(lambda x: "Y" in x, all_outcomes))) / len(all_outcomes)
print(prob)
