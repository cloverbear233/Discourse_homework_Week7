#1.分治思想实现二分法
def dfs(nums: list[int], target: int, i: int, j: int) -> int:
    """二分查找：问题 f(i, j)"""
    # 若区间为空，代表无目标元素，则返回 -1
    if i > j:
        return -1
    # 计算中点索引 m
    m = (i + j) // 2
    if nums[m] < target:
        # 递归子问题 f(m+1, j)
        return dfs(nums, target, m + 1, j)
    elif nums[m] > target:
        # 递归子问题 f(i, m-1)
        return dfs(nums, target, i, m - 1)
    else:
        # 找到目标元素，返回其索引
        return m


def binary_search(nums: list[int], target: int) -> int:
    """二分查找"""
    n = len(nums)
    # 求解问题 f(0, n-1)
    return dfs(nums, target, 0, n - 1)

#汉诺塔问题
def move(src: list[int], tar: list[int]):
    """移动一个圆盘"""
    # 从 src 顶部拿出一个圆盘
    pan = src.pop()
    # 将圆盘放入 tar 顶部
    tar.append(pan)

def dfs(i: int, src: list[int], buf: list[int], tar: list[int]):
    """求解汉诺塔问题 f(i)"""
    # 若 src 只剩下一个圆盘，则直接将其移到 tar
    if i == 1:
        move(src, tar)
        return
    # 子问题 f(i-1) ：将 src 顶部 i-1 个圆盘借助 tar 移到 buf
    dfs(i - 1, src, tar, buf)
    # 子问题 f(1) ：将 src 剩余一个圆盘移到 tar
    move(src, tar)
    # 子问题 f(i-1) ：将 buf 顶部 i-1 个圆盘借助 src 移到 tar
    dfs(i - 1, buf, src, tar)

def solve_hanota(A: list[int], B: list[int], C: list[int]):
    """求解汉诺塔问题"""
    n = len(A)
    # 将 A 顶部 n 个圆盘借助 B 移到 C
    dfs(n, A, B, C)

#2回溯算法通常采用“深度优先搜索”来遍历解空间。在“二叉树”章节中，我们提到前序、中序和后序遍历都属于深度优先搜索。
1	class TreeNode:
2	    """二叉树节点类"""
3
4	    def __init__(self, val: int = 0):
5	        self.val: int = val  # 节点值
6	        self.left: TreeNode | None = None  # 左子节点引用
7	        self.right: TreeNode | None = None  # 右子节点引用
8
9	def list_to_tree_dfs(arr: list[int], i: int) -> TreeNode | None:
10	    """将列表反序列化为二叉树：递归"""
11	    # 如果索引超出数组长度，或者对应的元素为 None ，则返回 None
12	    if i < 0 or i >= len(arr) or arr[i] is None:
13	        return None
14	    # 构建当前节点
15	    root = TreeNode(arr[i])
16	    # 递归构建左右子树
17	    root.left = list_to_tree_dfs(arr, 2 * i + 1)
18	    root.right = list_to_tree_dfs(arr, 2 * i + 2)
19	    return root
20
21	def list_to_tree(arr: list[int]) -> TreeNode | None:
22	    """将列表反序列化为二叉树"""
23	    return list_to_tree_dfs(arr, 0)
24
25
26	def pre_order(root: TreeNode):
27	    """前序遍历：例题一"""
28	    if root is None:
29	        return
30	    if root.val == 7:
31	        # 记录解
32	        res.append(root)
33	    pre_order(root.left)
34	    pre_order(root.right)
35
36
37	"""Driver Code"""
38	if __name__ == "__main__":
39	    root = list_to_tree([1, 7, 3, 4, 5, 6, 7])
40
41	    # 前序遍历
42	    res = list[TreeNode]()
43	    pre_order(root)
44#给定一棵二叉树，搜索并记录所有值为7的节点，请返回节点列表。
45	    print("\n输出所有值为 7 的节点")
46	    print([node.val for node in res])

#尝试与回退
def pre_order(root: TreeNode):
    """前序遍历：例题二"""
    if root is None:
        return
    # 尝试
    path.append(root)
    if root.val == 7:
        # 记录解
        res.append(list(path))
    pre_order(root.left)
    pre_order(root.right)
    # 回退
    path.pop()

#子集和问题
#例如，输入集合{3,4,5}和目标整数9，解为{3,3,3}和{4,5}。需要注意以下两点。
#输入集合中的元素可以被无限次重复选取。子集不区分元素顺序，比如{4,5}和{5,4}是同一个子集。
def backtrack(
    state: list[int],
    target: int,
    total: int,
    choices: list[int],
    res: list[list[int]],
):
    """回溯算法：子集和 I"""
    # 子集和等于 target 时，记录解
    if total == target:
        res.append(list(state))
        return
    # 遍历所有选择
    for i in range(len(choices)):
        # 剪枝：若子集和超过 target ，则跳过该选择
        if total + choices[i] > target:
            continue
        # 尝试：做出选择，更新元素和 total
        state.append(choices[i])
        # 进行下一轮选择
        backtrack(state, target, total + choices[i], choices, res)
        # 回退：撤销选择，恢复到之前的状态
        state.pop()

def subset_sum_i_naive(nums: list[int], target: int) -> list[list[int]]:
    """求解子集和 I（包含重复子集）"""
    state = []  # 状态（子集）
    total = 0  # 子集和
    res = []  # 结果列表（子集列表）
    backtrack(state, target, total, nums, res)
    return res

#方法2
def backtrack(
    state: list[int], target: int, choices: list[int], start: int, res: list[list[int]]
):
    """回溯算法：子集和 I"""
    # 子集和等于 target 时，记录解
    if target == 0:
        res.append(list(state))
        return
    # 遍历所有选择
    # 剪枝二：从 start 开始遍历，避免生成重复子集
    for i in range(start, len(choices)):
        # 剪枝一：若子集和超过 target ，则直接结束循环
        # 这是因为数组已排序，后边元素更大，子集和一定超过 target
        if target - choices[i] < 0:
            break
        # 尝试：做出选择，更新 target, start
        state.append(choices[i])
        # 进行下一轮选择
        backtrack(state, target - choices[i], choices, i, res)
        # 回退：撤销选择，恢复到之前的状态（就是最后得出了choice=target然后，一步一步回退到最开始的状态
        state.pop()

def subset_sum_i(nums: list[int], target: int) -> list[list[int]]:
    """求解子集和 I"""
    state = []  # 状态（子集）
    nums.sort()  # 对 nums 进行排序
    start = 0  # 遍历起始点
    res = []  # 结果列表（子集列表）
    backtrack(state, target, nums, start, res)
    return res