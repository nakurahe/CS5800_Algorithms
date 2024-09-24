# HW1 Problme 2: Hybrid Sorting
# For the hybrid_sort part,
# I used Copilot to generate the code and modified it myself.
import heapq
import math
import random
import time
import pandas as pd


def quick_sort(arr: list) -> list:
    '''
    Quick sort to sort an array arr of n elements in ascending order.
    '''
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)


def insertion_sort(arr: list) -> list:
    '''
    Insertion sort to sort an array arr of n elements in ascending order.
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


def heapsort(arr):
    '''
    Heapsort using heapq module.
    '''
    h = []
    for value in arr:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for _ in range(len(h))]


def hybrid_sort(arr) -> list:
    '''
    A hybrid sorting algorithm that combines the quick sort, insertion sort,
        and heap sort.
    The algorithm uses the quick sort to sort the array
        until the depth reaches 2 * math.log2(n), then it switches to the
        heap sort to sort the array.
    When the size of the array is less than or equal to 16,
        the algorithm uses the insertion sort to sort the array.
    '''
    def _hybrid_sort_helper(arr, depth, max_depth):
        # If the size of the array is less than or equal to 16,
        # simply use the insertion sort to sort the array.
        if len(arr) <= 16:
            return insertion_sort(arr)
        # If the depth is greater than the max_depth,
        # use the heap sort to sort the array afterwards.
        if depth > max_depth:
            return heapsort(arr)

        # Otherwise, use the quick sort to sort the array.
        # aka, if low < high:
        pivot = arr[len(arr) // 2]  # Choose the middle element as the pivot.
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]

        return (_hybrid_sort_helper(left, depth + 1, max_depth) +
                middle +
                _hybrid_sort_helper(right, depth + 1, max_depth))

    max_depth = 2 * math.log2(len(arr))
    return _hybrid_sort_helper(arr, 0, max_depth)


def generate_nearly_sorted_array(size: int) -> list:
    '''
    Generate a nearly sorted array of a cetain size.
    '''
    arr = sorted([random.randint(0, size) for _ in range(size)])
    # Randomly swap five elements in the array.
    for _ in range(5):
        i, j = random.sample(range(size), 2)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def generate_arrays(sizes: list) -> list:
    '''
    Generate random, nearly sorted, and reverse sorted arrays of size n.
    '''
    arrays = {}
    for size in sizes:
        arrays[size] = {
            'random': [random.randint(0, size) for _ in range(size)],
            'nearly_sorted': generate_nearly_sorted_array(size),
            'reverse_sorted': sorted(
                [random.randint(0, size) for _ in range(size)], reverse=True)
        }
    return arrays


def measure_performance(arrays: dict) -> list:
    results = []
    for size, array_types in arrays.items():
        for array_type, array in array_types.items():
            for sort_name, sort_func in [('Hybrid Sort', hybrid_sort),
                                         ('Quicksort', quick_sort),
                                         ('Heapsort', heapsort),
                                         ('Insertion Sort', insertion_sort)
                                         ]:
                arr_copy = array.copy()
                start_time = time.time()
                sort_func(arr_copy)
                end_time = time.time()
                results.append({
                    'Size': size,
                    'Array Type': array_type,
                    'Sort Algorithm': sort_name,
                    'Time (s)': end_time - start_time
                })
    return results


# Main function
def main():
    sizes = [100, 1000, 10000, 100000]
    arrays = generate_arrays(sizes)
    results = measure_performance(arrays)
    df = pd.DataFrame(results)

    # Specify the order of the columns
    sort_order = ['Hybrid Sort', 'Quicksort', 'Heapsort', 'Insertion Sort']

    # Create the pivot table with the specified column order
    pivot_table = df.pivot_table(index=['Size', 'Array Type'],
                                 columns='Sort Algorithm', values='Time (s)')
    pivot_table = pivot_table[sort_order]

    print(pivot_table)


if __name__ == "__main__":
    main()
