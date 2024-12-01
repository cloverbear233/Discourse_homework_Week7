from collections import deque

if __name__ == "__main__":
# 初始化队列
	    # 在 Python 中，我们一般将双向队列类 deque 看作队列使用
	    # 虽然 queue.Queue() 是纯正的队列类，但不太好用
	que = deque()#队列形式的设置！
# 元素入队
	que.append(1)
	que.append(3)
	que.append(2)
	que.append(5)
	que.appendleft(4)#左加
	que.insert(2,12)#index和数字
	que.extendleft(range(14,21))#从左一个一个塞进去
	print("队列 que =", que)

# 访问队首元素
	front = que[0]
	print("队首元素 front =", front)


# 元素出队
	pop = que.popleft()
	print("出队元素 pop =", pop)
	print("出队后 que =", que)

# 获取队列的长度
	size = len(que)
	print("队列长度 size =", size)

# 判断队列是否为空
	is_empty = len(que) == 0
	print("队列是否为空 =", is_empty)
class TreeNode:
#二叉树节点类
	def __init__(self, val):
		self.val = val
		self.left = None
		self.right = None
class BinarySearchTree:
#二叉搜索树
	def __init__(self):
#构造方法
# 初始化空树
		self.root = None
	def search(self, num: int):
#查找节点
		cur = self.root
# 循环查找，越过叶节点后跳出
		while cur is not None:
# 目标节点在 cur 的右子树中
			if cur.val < num:
				cur = cur.right
# 目标节点在 cur 的左子树中
			elif cur.val > num:
				cur = cur.left
# 找到目标节点，跳出循环
			else:
				break
		return cur

	def insert(self, num: int):
#插入节点
# 若树为空，则初始化根节点
		if self.root is None:
			self.root = TreeNode(num)
			return
# 循环查找，越过叶节点后跳出
		cur, pre = self.root, None
		while cur is not None:
# 找到重复节点，直接返回
			if cur.val == num:
				return
			pre = cur
# 插入位置在 cur 的右子树中
			if cur.val < num:
				cur = cur.right
# 插入位置在 cur 的左子树中
			else:
				cur = cur.left
# 插入节点
			node = TreeNode(num)
			if pre.val < num:
				pre.right = node
			else:
				pre.left = node

if __name__ == "__main__":
# 初始化二叉搜索树
	bst = BinarySearchTree()
	nums = [4, 2, 6, 1, 3, 5, 7]
	for num in nums:
		bst.insert(num)
# 查找节点
	node = bst.search(7)
	print("\n查找到的节点对象为: {}，节点值 = {}".format(node, node.val))

import heapq
# 初始化小顶堆
min_heap, flag = [], 1
# 初始化大顶堆
max_heap, flag = [], -1

# Python 的 heapq 模块默认实现小顶堆
# 考虑将“元素取负”后再入堆，这样就可以将大小关系颠倒，从而实现大顶堆
# 在本示例中，flag = 1 时对应小顶堆，flag = -1 时对应大顶堆

# 元素入堆
heapq.heappush(max_heap, flag * 1)
heapq.heappush(max_heap, flag * 3)
heapq.heappush(max_heap, flag * 2)
heapq.heappush(max_heap, flag * 5)
heapq.heappush(max_heap, flag * 4)

# 获取堆顶元素
peek: int = flag * max_heap[0] # 5

# 堆顶元素出堆
# 出堆元素会形成一个从大到小的序列
val = flag * heapq.heappop(max_heap) # 5
val = flag * heapq.heappop(max_heap) # 4
val = flag * heapq.heappop(max_heap) # 3
val = flag * heapq.heappop(max_heap) # 2
val = flag * heapq.heappop(max_heap) # 1

# 获取堆大小
size: int = len(max_heap)

# 判断堆是否为空
is_empty: bool = not max_heap

# 输入列表并建堆
min_heap: list[int] = [1, 3, 2, 5, 4]
heapq.heapify(min_heap)

#索引
def left(self, i: int) -> int:
    """获取左子节点的索引"""
    return 2 * i + 1

def right(self, i: int) -> int:
    """获取右子节点的索引"""
    return 2 * i + 2

def parent(self, i: int) -> int:
    """获取父节点的索引"""
    return (i - 1) // 2  # 向下整除