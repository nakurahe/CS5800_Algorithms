'''
    NEU Fall 2024 CS5800 Algorithm
    This is the codebase for all homeworks, pre-learning notes of this course.
    Jiahuan
'''
import heapq
import math
import collections


# Week1 Pre-learning Notes
def insertion_sort_iterative(arr: list) -> list:
    '''
    Given an array arr of n elements, sort the array in ascending order.
    '''
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            # Move elements of arr[0..i-1], that are greater than key,
            # to one position ahead of their current position.
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def insertion_sort_recursive(arr: list, n: int) -> list:
    '''
    Given an array arr of n elements, sort the array in ascending order.
    '''
    if n <= 1:
        return arr
    insertion_sort_recursive(arr, n - 1)
    key = arr[n - 1]
    j = n - 2
    while j >= 0 and key < arr[j]:
        arr[j + 1] = arr[j]
        j -= 1
    arr[j + 1] = key
    return arr


# HW1 Problem 2: Hybrid Sorting
# Used Copilot to generate the code and modified manually.
def hybrid_sort(arr):
    '''
    A hybrid sorting algorithm that combines the quick sort, insertion sort,
        and heap sort.
    The algorithm uses the quick sort to sort the array
        until the depth reaches 2 * math.log2(n), then it switches to the
        heap sort to sort the array.
    When the size of the array is less than or equal to 16,
        the algorithm uses the insertion sort to sort the array.
    '''
    def hybrid_sort_helper(arr, low, high, depth, max_depth) -> list:
        # If the size of the array is less than or equal to 16,
        # simply use the insertion sort to sort the array.
        if high - low <= 16:
            return insertion_sort_iterative(arr[low:high+1])

        # If the depth is greater than the max_depth,
        # use the heap sort to sort the array afterwards.
        if depth > max_depth:
            heapq.heapify(arr[low:high+1])
            return [heapq.heappop(arr[low:high+1]) for _ in range(low, high+1)]

        # Otherwise, use the quick sort to sort the array.
        if low < high:
            pivot = partition(arr, low, high)
            hybrid_sort_helper(arr, low, pivot - 1, depth + 1, max_depth)
            hybrid_sort_helper(arr, pivot + 1, high, depth + 1, max_depth)

    def partition(arr, low, high) -> int:
        # Choose the last element as the pivot.
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        # Return the index of the pivot finally.
        return i + 1

    n = len(arr)
    max_depth = int(2 * math.log2(n))
    return hybrid_sort_helper(arr, 0, n - 1, 0, max_depth)


# Example usage
print("HW1 Problem 2: Hybrid Sorting")
arr = [-1, 1, 3, -5, 7, 9, -11, 13, 15, -17, 19, 21, -23]
print("Sorting the array:", arr)
sorted_arr = hybrid_sort(arr)
print("Result:", sorted_arr)


# HW1 Problem 3: Binary Search
def guessing_number_binary_search(arr: list, target: int) -> int:
    '''
    Given a sorted array arr of n elements,
    search the given element target,
    and return the number of attempts guessing the target.
    '''
    left_index, right_index = 0, len(arr) - 1
    count = 0
    while left_index <= right_index:
        count += 1
        # This is more robust than (left_index + right_index) // 2,
        # which may cause overflow. That is, when left_index and right_index
        # are both very large, their sum may exceed the limit of the integer.
        mid_index = left_index + (right_index - left_index) // 2
        if arr[mid_index] == target:
            return count
        elif arr[mid_index] < target:
            left_index = mid_index + 1
        else:
            right_index = mid_index - 1
    return count


# HW1 Problem 5: Divide and Conquer: Optimal Timing in Investment
# Used Copilot to generate the code and modified manually.
def maxProfit(prices):
    def maxProfitHelper(prices, left, right):
        if left >= right:
            return 0

        mid = (left + right) // 2

        left_profit = maxProfitHelper(prices, left, mid)
        right_profit = maxProfitHelper(prices, mid + 1, right)

        min_left = min(prices[left:mid + 1])
        max_right = max(prices[mid + 1:right + 1])

        cross_profit = max_right - min_left

        return max(left_profit, right_profit, cross_profit)

    if not prices:
        return 0

    return maxProfitHelper(prices, 0, len(prices) - 1)


# Week2 Quiz2
# Used GPT to generate the code and modified manually.
def largest_subarray(nums: list) -> list:
    '''
    Given an array arr of n elements,
    find the subarray with the largest sum.
    '''
    if not nums:
        return []

    current_sum, max_sum = nums[0], nums[0]
    start, end, temp_start = 0, 0, 0

    for i in range(1, len(nums)):
        # If the current sum is less than or equal to 0,
        # then the current sub-array, no matter a new sub-array or
        # an old array appending nums[i-1], is not the optimal sub-array.
        # So we should start a new sub-array from nums[i].
        if current_sum <= 0:
            current_sum = nums[i]
            temp_start = i
        else:
            current_sum += nums[i]

        # If after appending a new element nums[i], the current sum is greater
        # than the max_sum, then we update the max_sum and
        # set start from the last temp_start and update end to current i.
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i

    return nums[start:end+1]


print("\nWeek2 Quiz2")
arr = [-1, 1, 3, -5, 7, 9, -11, 13, 15, -17, 19, 21, -23]
print("The largest subarray from:", arr)
print("Result:", largest_subarray(arr))
print("The sum is:", sum(largest_subarray(arr)))


# Week3 Pre-learning Notes
# Used Copilot to generate the code and modified manually.
def counting_sort(arr: list) -> list:
    '''
    Given an array arr of n elements,
    sort the array in ascending order, by counting sort.
    '''
    if not arr:
        return []

    # Find the maximum value in the array.
    max_value = max(arr)
    # Initialize the count array with the length of max_value + 1.
    count = [0] * (max_value + 1)
    # Count the frequency of each element.
    for num in arr:
        count[num] += 1
    # Calculate the prefix sum of the count array.
    for i in range(1, len(count)):
        count[i] += count[i - 1]
    # Initialize the result array with the length of n.
    result = [0] * len(arr)
    # Fill the result array with the sorted elements.
    for num in reversed(arr):
        result[count[num] - 1] = num
        count[num] -= 1
    return result


def bucket_sort(arr: list) -> list:
    '''
    Given an array arr of n elements,
    sort the array in ascending order, by bucket sort.
    '''
    if not arr:
        return []

    # Find the maximum value in the array.
    max_value = max(arr)
    # Find the minimum value in the array.
    min_value = min(arr)
    # Calculate the number of buckets.
    num_buckets = len(arr)
    # Initialize the buckets with the number of buckets.
    buckets = [[] for _ in range(num_buckets)]
    # Calculate the range of each bucket.
    bucket_range = (max_value - min_value) / num_buckets
    # Fill the buckets with the elements.
    for num in arr:
        bucket_index = min(int(num // bucket_range), num_buckets - 1)
        buckets[bucket_index].append(num)
    # Sort each bucket.
    for i in range(num_buckets):
        buckets[i] = insertion_sort_iterative(buckets[i])
    # Concatenate the sorted buckets.
    result = []
    for bucket in buckets:
        result.extend(bucket)
    return result


# Example usage
arr = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
print("\nWeek3 Pre-learning Notes")
print("Bucket Sort this array:", arr)
arr = bucket_sort(arr)
print("Result:", arr)


def selection_problem(arr: list, k: int) -> int:
    '''
    Given an array arr of n elements and an integer k,
    find the k-th smallest element in the array.
    '''
    def selection_problem_helper(arr, left, right, k):
        if left == right:
            return arr[left]

        pivot_index = partition(arr, left, right)
        if k == pivot_index:
            return arr[k]
        elif k < pivot_index:
            return selection_problem_helper(arr, left, pivot_index - 1, k)
        else:
            return selection_problem_helper(arr, pivot_index + 1, right, k)

    def partition(arr, left, right):
        pivot = arr[right]
        i = left - 1
        for j in range(left, right):
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[right] = arr[right], arr[i + 1]
        return i + 1

    return selection_problem_helper(arr, 0, len(arr) - 1, k - 1)


# HW2 Problem 1: Stooge Sort
def stooge_sort(arr: list) -> list:
    '''
    Given an array arr of n elements,
    sort the array in ascending order, by stooge sort.
    Pseudocode:
        Stooge-Sort(A, p, r)
            if A[p] > A[r]
                swap A[p] and A[r]
            if r - p + 1 > 2:
                q = (r - p + 1) / 3
                Stooge-Sort(A, p, r - q)
                Stooge-Sort(A, p + q, r)
                Stooge-Sort(A, p, r - q)
'''
    def stooge_sort_helper(arr, p, r):
        if arr[p] > arr[r]:
            arr[p], arr[r] = arr[r], arr[p]
        if r - p + 1 > 2:
            q = (r - p + 1) // 3
            stooge_sort_helper(arr, p, r - q)
            stooge_sort_helper(arr, p + q, r)
            stooge_sort_helper(arr, p, r - q)

    stooge_sort_helper(arr, 0, len(arr) - 1)
    return arr


# HW2 Problem 2: Simultaneous Maximum and Minimum
def simultaneous_max_min(arr: list) -> tuple:
    '''
    Given an array arr of n elements,
    find the maximum and minimum elements in the array.
    '''
    if not arr:
        return None, None

    n = len(arr)
    if n % 2 == 0:
        max_num = max(arr[0], arr[1])
        min_num = min(arr[0], arr[1])
        start_index = 2
    else:
        max_num = min_num = arr[0]
        start_index = 1

    for i in range(start_index, n - 1, 2):
        if arr[i] < arr[i + 1]:
            min_num = min(min_num, arr[i])
            max_num = max(max_num, arr[i + 1])
        else:
            min_num = min(min_num, arr[i + 1])
            max_num = max(max_num, arr[i])

    return max_num, min_num


# in-class exercise
# DP: find the smallest number of coins
def too_many_coins(coins: list, target: int) -> int:
    '''
    Given a list of coins and a target value,
    find the minimum number of coins that sum up to the target value.
    '''
    dp = [float('inf')] * (target + 1)
    dp[0] = 0

    for i in range(1, target + 1):
        for coin in coins:
            if i - coin >= 0:
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[target] if dp[target] != float('inf') else -1


# Recursion: find the smallest number of coins
def too_many_coins_recursion(coins: list, target: int) -> int:
    def too_many_coins_recursion_helper(coins, target):
        if target == 0:
            return 0
        if target < 0:
            return float('inf')

        min_coins = float('inf')
        for coin in coins:
            min_coins = min(
                min_coins,
                too_many_coins_recursion_helper(coins, target - coin) + 1)

        return min_coins

    min_coins = too_many_coins_recursion_helper(coins, target)
    return min_coins if min_coins != float('inf') else -1


# Greedy: find the smallest number of coins
def too_many_coins_greedy(coins: list, target: int) -> int:
    coins.sort(reverse=True)
    num_coins = 0

    for coin in coins:
        while target >= coin:
            target -= coin
            num_coins += 1

    return num_coins if target == 0 else -1


# HW3 Problem 1: Longest Increasing Subsequence(c)
def longest_common_subsequence(arr1: list, arr2: list) -> list:
    '''
    Given two arrays arr1 and arr2,
    find the longest common subsequence of the two arrays.
    '''
    m, n = len(arr1), len(arr2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = m, n
    lcs = []
    while i > 0 and j > 0:
        if arr1[i - 1] == arr2[j - 1]:
            lcs.append(arr1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]


# HW3 Problem 1: Longest Increasing Subsequence(d)
def length_of_longest_increasing_subsequence(arr: list) -> int:
    '''
    Given an array arr of n elements,
    find the length of the longest increasing subsequence
        with O(n * logn) time.
    '''
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    # Initialize an empty list to store the LIS candidates
    lis = []

    for num in arr:
        # Find the position to replace or append the current number
        pos = binary_search(lis, num)

        # If pos is equal to the length of lis, append the number
        if pos == len(lis):
            lis.append(num)
        else:
            # Otherwise, replace the element at the found position
            lis[pos] = num

    # The length of lis is the length of the longest increasing subsequence
    return len(lis)


# Example usage
print("\nHW3 Problem 1: Longest Increasing Subsequence")
arr = [2, 4, 5, 3]
print("The length of the longest increasing subsequence in", arr)
print("Result:", length_of_longest_increasing_subsequence(arr))


# HW3 Problem 2: Fractional Knapsack Problem
def fractional_knapsack_solution(items: list, capacity: int) -> float:
    '''
    Given a list of items, each item has a weight and a value,
    and a knapsack with a capacity,
    find the maximum total value that can be put into the knapsack.
    '''
    # Sort the items by the value per weight in descending order.
    items.sort(key=lambda x: x[1] / x[0], reverse=True)

    total_value = 0
    for weight, value in items:
        if capacity == 0:
            break
        # If the weight of the item is less than the capacity,
        # put the whole item into the knapsack.
        if weight <= capacity:
            total_value += value
            capacity -= weight
        # If the weight of the item is greater than the capacity,
        # put the fraction of the item into the knapsack.
        else:
            total_value += value * capacity / weight
            break

    return total_value


# HW3 Problem 3: Sleeping cats (a)
def time_to_wake_up(matrix, m, n, x, y):
    '''
    Given a matrix of m rows and n columns,
    where each cell contains the status of the cat,
    find the minimum time to wake up the cat at the cell (x, y).
    '''
    if matrix[x][y] != -1:
        return 0  # The cat is already awake or the cell is empty

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    queue = []
    visited = set()

    # Initialize the queue with all awake cats
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                queue.append((i, j, 0))  # (row, col, time)
                visited.add((i, j))

    # Perform BFS
    while queue:
        current_row, current_col, time = queue.pop(0)

        for direction in directions:
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
            if (new_row in range(0, m) and new_col in range(0, n)
                    and (new_row, new_col) not in visited
                    and matrix[new_row][new_col] == -1):
                if new_row == x and new_col == y:
                    return time + 1
                queue.append((new_row, new_col, time + 1))
                visited.add((new_row, new_col))

    return -1  # The cat at (x, y) cannot be woken up for some reason


# HW3 Problem 3: Sleeping cats (b)
def time_to_wake_up_all(matrix):
    '''
    Given a matrix of m rows and n columns,
    where each cell contains the status of the cat,
    find the minimum time to wake up all the cats
    '''
    m, n = len(matrix), len(matrix[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = collections.deque()
    awake = set()
    max_time = 0

    # Initialize the queue with all awake cats
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                queue.append((i, j, 0))  # (row, col, time)
                awake.add((i, j))

    # Perform BFS
    while queue:
        current_row, current_col, time = queue.popleft()
        max_time = max(max_time, time)

        for direction in directions:
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
            if (new_row in range(0, m) and new_col in range(0, n)
                    and (new_row, new_col) not in awake
                    and matrix[new_row][new_col] == -1):
                queue.append((new_row, new_col, time + 1))
                awake.add((new_row, new_col))

    # Check if all the cats are awake
    for row in matrix:
        if -1 in row:
            # There is at least one cat that cannot be woken up
            return -1

    return max_time


# HW3 Problem 3: Sleeping cats (c)
def track_wake_up_path(matrix, x, y):
    '''
    Given a matrix of m rows and n columns,
    where each cell contains the status of the cat,
    find the path to wake up the cat at the cell (x, y).
    '''
    if matrix[x][y] != -1:
        return []  # The cat is already awake or the cell is empty

    m, n = len(matrix), len(matrix[0])

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = collections.deque()
    visited = set()
    parent = {}

    # Initialize the queue with all awake cats
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 1:
                queue.append((i, j))
                visited.add((i, j))
                parent[(i, j)] = None

    # Perform BFS
    while queue:
        current_row, current_col = queue.popleft()

        for direction in directions:
            new_row = current_row + direction[0]
            new_col = current_col + direction[1]
            if (new_row in range(0, m) and new_col in range(0, n)
                    and (new_row, new_col) not in visited
                    and matrix[new_row][new_col] == -1):
                queue.append((new_row, new_col))
                visited.add((new_row, new_col))
                parent[(new_row, new_col)] = (current_row, current_col)
                if new_row == x and new_col == y:
                    return construct_path(parent, (x, y))

    return []  # The cat at (x, y) cannot be woken up


def construct_path(parent, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return path[::-1]


# CLRS 20, 21:
def prims_algorithm_adjacency_list(graph):
    '''
    Given a graph represented by an adjacency list,
    find the minimum spanning tree using Prim's algorithm.
    '''
    n = len(graph)
    parent = [-1] * n
    key = [float('inf')] * n
    visited = [False] * n
    key[0] = 0

    for _ in range(n):
        u = min_key(key, visited)
        visited[u] = True

        for v, w in graph[u]:
            if not visited[v] and w < key[v]:
                parent[v] = u
                key[v] = w

    return parent


def min_key(key, visited):
    min_value = float('inf')
    min_index = -1
    for i in range(len(key)):
        if not visited[i] and key[i] < min_value:
            min_value = key[i]
            min_index = i
    return min_index


def kruskals_algorithm(graph):
    '''
    Given a graph represented by an adjacency list,
    find the minimum spanning tree using Kruskal's algorithm.
    '''
    n = len(graph)
    parent = [i for i in range(n)]
    rank = [0] * n
    result = []

    edges = []
    for u in range(n):
        for v, w in graph[u]:
            edges.append((u, v, w))
    edges.sort(key=lambda x: x[2])

    for u, v, w in edges:
        root_u = find(parent, u)
        root_v = find(parent, v)
        if root_u != root_v:
            result.append((u, v, w))
            union(parent, rank, root_u, root_v)

    return result


def find(parent, u):
    if parent[u] != u:
        parent[u] = find(parent, parent[u])
    return parent[u]


def union(parent, rank, u, v):
    if rank[u] > rank[v]:
        parent[v] = u
    elif rank[u] < rank[v]:
        parent[u] = v
    else:
        parent[v] = u
        rank[u] += 1


# HW4 Question 2: Graph Design
def can_satisfy_constraints(n, equalities, inequalities):
    # Initialize Union-Find data structures
    parent = [i for i in range(n)]
    rank = [0] * n

    # Helper functions for Union-Find
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    # Process equality constraints
    for (xi, xj) in equalities:
        union(xi, xj)

    # Check inequality constraints
    for (xi, xj) in inequalities:
        if find(xi) == find(xj):
            return False

    return True


# Example usage
print("\nHW4 Question 2: Graph Design")
n = 5
equalities = [(0, 1), (1, 2)]
inequalities = [(0, 3), (2, 4)]
print("Can satisfy the constraints?", can_satisfy_constraints(
    n, equalities, inequalities))
equalities = [(0, 1), (1, 2), (2, 3), (3, 4)]
inequalities = [(0, 4)]
print("Can satisfy the constraints?", can_satisfy_constraints(
    n, equalities, inequalities))


# HW4 Question 3: Greedy Algorithm
def greedy_min_cost(items: list, max_value: int, max_weight: int) -> int:
    '''
    Given a list of items, each item has a value and a weight,
        and the maximum value one value-restricted box can hold,
        and maximum weight one weight-restricted box can hold,
        calculate the minimum cost to pack these items into boxes.
        One box costs 1 dollar.
    '''
    number_of_items = len(items)
    i, total_box = 0, 0

    while i < number_of_items:
        current_value_item, current_weight_item = i, i
        current_box_value_sum, current_box_weight_sum = 0, 0

        while (current_value_item < number_of_items
               and current_box_value_sum + items[current_value_item][0] <= max_value):
            current_box_value_sum += items[current_value_item][0]
            current_value_item += 1

        while (current_weight_item < number_of_items
               and current_box_weight_sum + items[current_weight_item][1] <= max_weight):
            current_box_weight_sum += items[current_weight_item][1]
            current_weight_item += 1

        if current_value_item > current_weight_item:
            i = current_value_item
        else:
            i = current_weight_item

        total_box += 1

    return total_box


# Example usage
print("\nHW4 Question 3: Greedy Algorithm")
items = [(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)]
max_value = 100
max_weight = 10
print("The minimum cost to pack the items into boxes:", greedy_min_cost(
    items, max_value, max_weight))


# Week8 in-class exercise
def findMostShops(street_shops: list) -> list:
    index_of_shops = []
    n = len(street_shops)
    i = 0
    while i < n:
        if street_shops[i] == 1:
            index_of_shops.append(i)
            i += 3
        else:
            i += 1

    return index_of_shops


print("\nWeek8 in-class exercise")
street_shops = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
print("The maximum number of shops:", findMostShops(street_shops))


def findMostShopsDP(street_shops: list) -> list:
    '''
    Given a list of 0s and 1s, where non-0 digit represents a shop's revenue,
    find the maximum revenue of shops that can be opened.
    '''
    n = len(street_shops)
    # When n is less than 3,
    # return the index of the shop with the maximum revenue.
    if n < 3:
        return street_shops.index(max(street_shops))

    # When n is larger than 3,
    # use dynamic programming to find the maximum revenue.
    dp = [0] * n
    dp[0] = street_shops[0]
    dp[1] = street_shops[1]
    dp[2] = street_shops[2] + dp[0]
    for i in range(3, n):
        dp[i] = max(street_shops[i] + dp[i - 3], dp[i - 1])

    # Find the index of the shops with the maximum revenue.
    index_of_shops = []
    i = n - 1
    while i >= 0:
        if dp[i] == max(dp):
            index_of_shops.append(i)
            i -= 3
        else:
            i -= 1

    return index_of_shops[::-1]


# Example usage
print("\nWeek8 in-class exercise")
street_shops = [0, 11, 0, 1, 0, 5, 9, 1, 0, 100, 0, 1, 1, 0, 0, 0]
print("The maximum number of shops:", findMostShopsDP(street_shops))
