'''
    NEU Fall 2024 CS5800 Algorithm
    This is the codebase for all homeworks, prelearning notes of this course.
    Jiahuan
'''
import heapq
import math


# Week1 Prelearning Notes
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


# HW1 Problme 2: Hybrid Sorting
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
print("HW1 Problme 2: Hybrid Sorting")
arr = [-1, 1, 3, -5, 7, 9, -11, 13, 15, -17, 19, 21, -23]
print("Sorting the array:", arr)
sorted_arr = hybrid_sort(arr)
print("Result:", sorted_arr)


# HW1 Problme 3: Binary Search
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
        # This is more rubust than (left_index + right_index) // 2,
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


# HW1 Problme 5: Divide and Conquer: Optimal Timing in Investment
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
        # then the current subarray, no matter a new subarray or
        # an old array appending nums[i-1], is not the optimal subarray.
        # So we should start a new subarray from nums[i].
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


# Week3 Prelearning Notes
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
print("\nWeek3 Prelearning Notes")
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
