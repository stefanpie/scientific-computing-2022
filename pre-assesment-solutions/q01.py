import os
from pprint import pp
from pprint import pprint as pp


def read_data(fp):
    number = None
    with open(fp, "r") as f:
        number = f.readlines()
    number = [x.strip() for x in number]
    number = "".join(number)
    number = [int(x) for x in number]
    return number


def find_greatest_product(number, size):
    greatest_product = 0
    best_seq = None
    for i in range(0, len(number) - size + 1):
        seq = number[i : i + size]
        product = 1
        for x in seq:
            product *= x
        if product > greatest_product:
            greatest_product = product
            best_seq = seq
    return greatest_product, best_seq


if __name__ == "__main__":
    data_fp = "./data/big_number.txt"
    number = read_data(data_fp)
    greatest_product, seq = find_greatest_product(number, 13)
    seq_text = "".join([str(x) for x in seq])
    print(f"Greatest product: {greatest_product}")
    print(f"Sequence: {seq_text}")
