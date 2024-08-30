import itertools


# 生成集合 N 的所有子集
def generate_subsets(N):
    subsets = []
    for r in range(len(N) + 1):
        subsets.extend(itertools.combinations(N, r))
    return subsets


# 生成集合 N 的所有不相交子集对
def generate_subset_pairs(N):
    subsets_S = generate_subsets(N)
    subsets_T = generate_subsets(N)
    subset_pairs = [(S, T) for S in subsets_S for T in subsets_T if set(S).isdisjoint(T)]
    return subset_pairs


# 生成 N \ (S ∪ T) 的所有子集
def generate_remaining_subsets(N, S, T):
    remaining = N - set(S) - set(T)
    return generate_subsets(remaining)


# 计算 v(S ∪ A) 的累计效用
def calculate_cumulative_utility(S, remaining_subsets, v_S):
    cumulative_utility = 0
    for subset in remaining_subsets:
        union_set = set(S).union(subset)
        cumulative_utility += v_S[tuple(sorted(union_set))]
    return cumulative_utility


# 主函数
def main(n):
    N = set(range(1, n + 1))
    # 生成 N 的所有子集及其效用 v(S)
    all_subsets = generate_subsets(N)
    v_S = {subset: 1 for subset in all_subsets}  # 这里假设 v(S) 都为 1，可以根据需要修改

    # 生成所有不相交子集对 (S, T)
    subset_pairs = generate_subset_pairs(N)

    # 计算 v(S, T)
    v_ST = {}
    for S, T in subset_pairs:
        remaining_subsets = generate_remaining_subsets(N, S, T)
        cumulative_utility = calculate_cumulative_utility(S, remaining_subsets, v_S)
        v_ST[(S, T)] = cumulative_utility / (2 ** len(N - set(S) - set(T)))

    # 打印结果
    for (S, T), value in v_ST.items():
        print(f"v({S}, {T}) = {value}")

# 运行主函数，n 为集合的大小
main(3)  # 你可以修改这个值来测试不同大小集合的所有子集对

