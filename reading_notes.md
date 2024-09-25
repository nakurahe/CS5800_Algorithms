## Notes for of 5800 Algorithms

### Contents
- Running Time
- Sort
- Search
- DAC
---
### Running Time
> Referred from CLRS 4th pp. 50-51, pp. 54-56  

E.g. f(n) = 7 * n ^ 3 + 100 * n ^ 2 - 20 * n + 6  
O
: 
- Defination in an easier ver:
    1. characterizes the **upper bound** on the asymptotic behavior of a function.
    2. a function grows **no faster** than a certain rate.
    3. O(f(n)) = O(n ^ 3) = O(n ^ 4) = ... = O(n ^ c) for any constant c >= 3.
- The formal definition of O-notation:  
O(g(n)) = {f(n): there exist positive constants c and n0 such that 0 <= f(n) <= cg(n) for all n >= n0}

Ω
: 
- Defination in an easier ver:
    1. characterizes the **lower bound** on the asymptotic behavior of a function.
    2. a function grows **at least** as fast as a certain rate.
    3. Ω(f(n)) = Ω(n ^ 3) = Ω(n ^ 2) = ... = Ω(n ^ c) for any constant c <= 3.
- The formal definition of O-notation:  
Ω(g(n)) = {f(n): there exist positive constants c and n0 such that 0 <= cg(n) <= f(n) for all n >= n0}

Θ
: 
- Defination in an easier ver:
    1. characterizes a **tight** bound on the asymptotic behavior of a function.
    2. a function grows **precisely** at a certain rate.
    3. Since O(f(n)) = O(n ^ 3) and Ω(f(n)) = Ω(n ^ 3), Θ(f(n)) = Θ(n ^ 3)
- The formal definition of O-notation:  
Θ(g(n)) = {f(n): there exist positive constants c1, c2 and n0 such that 0 <= c1g(n) <= f(n) <= c2g(n) for all n >= n0}

**Theorem 3.1:** if the function f(n) is both O(f(n)) and Ω(f(n)), then f(n) is Θ(f(n)).

> Referred from CLRS 4th pp. 102-104  

**The Master Theorem**:  
From T(n) = aT(n/b) + f(n) where a >= 1 and b > 1 (??), calculate the **watershed function n ^ log_b(a)**.
1. If n ^ log_b(a) grows *asymptotically* and *polynomially* faster than f(n), aka must be faster than f(n) by at least a factor of Θ(n ^ ε) for ε > 0, then **case 1 -- T(n) = Θ(n ^ log_b(a))** -- applies.
2. If n ^ log_b(a) grows at *nearly the same* asymptotic rate as f(n), then **case 2 -- T(n) = Θ(n ^ log_b(a) * lg(n))** -- applies. Note that the most common situation for case 2 occurs when **k = 0**.
3. If f(n) grows *asymptotically* and *polynomially* faster than n ^ log_b(a), aka must be asymptotically larger than n ^ log_b(a) by at least a factor of Θ(n ^ ε) for ε > 0, AND a * f(n / b) <= c * f(n) for c < 1, then **case 3 -- T(n) = Θ(f(n))** -- applies.
---
### Sort
#### Comparison sort
1. **Insertion sort**
: Insert a number into a **sorted array**.
- Iterative solution:  
```
for i in range(1, len(arr)):
    key = arr[i]
    j = i - 1
    while j >= 0 and key < arr[j]:
        arr[j + 1] = arr[j]
        j -= 1
    arr[j + 1] = key
```
- Recursive solution:
```
sort n - 1 recursively, then insert.
insert is basically the same as above.
```

2. **Merge sort**
: divide into sublists and combine them.
- Iterative solution:  
[placeholder]

#### Non-comparison sort
1. **Counting sort**
- Complexity:  
Time complexity: Θ(n + k)  
Space complexity: Θ(n + k)

- Stability
    - When placing elements in the output array, the algorithm iterates through the input array in reverse order.
    - For each element, it places the element at the position indicated by the count array and then decrements the count.
    - Since the algorithm processes elements from the end of the input array to the beginning, it ensures that elements with the same key are placed in the output array in the same relative order as they appear in the input array.

In conclusion, counting sort is stable because it preserves the relative order of elements with equal keys by processing the input array in reverse order and using the count array to determine the positions of elements in the output array.

2. **Bucket sort**
- Time Comeplexity: Θ(n)

### Search
**Binary Search**
- [placeholder]
- **Recurrence**:  
T(n) = Θ(1) if n = 1,  
T(n) = T((n - 1) / 2) + Θ(1) if n > 1

### Divide-and-conquer
**[QUICK TIP](https://atekihcan.github.io/CLRS/02/E02.03-07/)**:
Every time we see **lg(n)**, we should think of divide-and-conquer algorithms. It inherently means how many times n can be divided by 2, i.e. repeated division of n elements in two groups.

- LC question alteration: find the **subarray** that has the largest sum.  
```
if not nums:
    return []

max_sum = float('-inf')
current_sum = 0
start = 0
end = 0
temp_start = 0

for i in range(len(nums)):
    if current_sum <= 0:
        current_sum = nums[i]
        temp_start = i
    else:
        current_sum += nums[i]

    if current_sum > max_sum:
        max_sum = current_sum
        start = temp_start
        end = i

return nums[start:end+1]
```

> Referred from CLRS 4th pp. 85-89

**Strassen’s algorithm**
- Partition the matrix to **n/2 submatrices** and apply Strassen's algorithm. If the length is not a power of 2, then add m-n 0s to the matrix to make it to the power of 2.
- Pseudocode:  
```
strassen_algorithm(A, B):
    n = A.rows
    let C be a new n * n matrix
    if n == 1
        c11 = a11 * b11
    else partition A, B and C
        let S1, S2, ..., S10 be 10 new n/2 * n/2 matrices
        let P1, P2, ..., P7 be 7 new n/2 * n/2 matrices

        S1 = B12 - B22
        S2 = A11 + A12
        S3 = A21 + A22
        S4 = B21 - B11
        S5 = A11 + A22
        S6 = B11 + B22
        S7 = A12 - A22
        S8 = B21 + B22
        S9 = A11 - A21
        S10 = B11 + B12

        P1 = strassen_algorithm(A11, S1)
        P2 = strassen_algorithm(S2, B22)
        P3 = strassen_algorithm(S3, B11)
        P4 = strassen_algorithm(A22, S4)
        P5 = strassen_algorithm(S5, S6)
        P6 = strassen_algorithm(S7, S8)
        P7 = strassen_algorithm(S9, S10)

        C11 = P4 + P5 + P6 - P2
        C12 = P1 + P2
        C21 = P3 + P4
        C22 = P1 + P5 - P3 - P7
    
    return C
```
- Alteration
:  the time complexity of m sub-problems with k multiplications, would be T(n) = Θ(n ^ log_m(k))

**Substitution Method**  
1. Assume T(n) <= c * G(n) where G(n) stands for some function, c and n0 are positive and this inequation is true for all n >= n0.
2. Substitute T(n + k) in the already-given T(n) = T(n + k) + b with above inequation, as T(n) = c * G(n + k) + b <= c * G(n), then solve c, and n0 finally.

Lecture notes:
build-in sort: almost introsort
heapsort: space complexity

if it's of high frenquency, then constant factor matters.

How many leaves on the tree:
min: n!
max: 2 ^ n  <-- just one branch.
Could use Stirling's approximation to prove.
