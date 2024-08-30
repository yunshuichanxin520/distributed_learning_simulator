import itertools

def generate_subsets(N):
    subsets = []
    for r in range(len(N)+1):
        subsets.extend(itertools.combinations(N, r))
    return subsets

def generate_subset_pairs(N):
    subsets_S = generate_subsets(N)
    subsets_T = generate_subsets(N)
    subset_pairs = [(S, T) for S in subsets_S for T in subsets_T if set(S).isdisjoint(T)]
    return subset_pairs

N = {1, 2, 3}
subset_pairs = generate_subset_pairs(N)
# 打印结果
for pair in subset_pairs:
    print(f"v({pair[0]}, {pair[1]})")

print("总共生成了 {} 组子集对".format(len(subset_pairs)))


# subsets = generate_subsets(N)
# for pair in subsets:
#     print(pair)