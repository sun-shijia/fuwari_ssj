---
title: 排序算法
published: 2025-2-21
description: 各种类型的排序算法(python)
tags: [Blogging]
category: Code
draft: false
---

![sort](sort.png)

# 冒泡排序

冒泡排序（Bubble Sort）是一种简单直观的排序算法。它重复地走访过要排序的数列，一次比较两个元素，如果他们的顺序错误就把他们交换过来。
走访数列的工作是重复地进行直到没有再需要交换，也就是说该数列已经排序完成。这个算法的名字由来是因为越小的元素会经由交换慢慢“浮”到数列的顶端。

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(1, n):
        for j in range(0, n-i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

# 选择排序

选择排序也是一种简单直观的排序算法，它首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，
再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾，重复该过程。

```python
def selectinon_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        minindex = i
        for j in range(i + 1, n):
            if arr[j] < arr[minindex]:
                minindex = j
        if i != minindex:
            arr[i], arr[minindex] = arr[minindex], arr[i]
    return arr
```

# 插入排序

插入排序的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。类似于扑克牌插牌。

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```

# 希尔排序

希尔排序（Shell Sort），也称递减增量排序算法，是插入排序的一种更高效的改进版本。
先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，待整个序列中的记录“基本有序”时，再对全体记录进行依次直接插入排序。
因为插入排序在对几乎已经排好序的数据操作时，效率高（每次调整一到两位），即可以达到线性排序的效率，因此希尔排序利用分成小块的快速排序来提高效率。

```python
def shellSort(arr):
    import math
    gap = 1
    n = len(arr)
    while(gap < n / 3):
        gap = gap * 3 + 1
    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i - gap
            while j >= 0 and arr[j] > temp:
                arr[j + gap] = arr[j]
                j -= gap
            arr[j + gap] = temp
        gap = math.floor(gap / 3)
    return arr
```

# 归并排序

归并排序（Merge sort）是建立在归并操作上的一种有效的排序算法。是一种自上而下的递归（或迭代）方法，作为一种典型的分而治之思想的算法应用。

```python
def merge_sort(arr):
    n = len(arr)
    if n < 2:
        return arr
    
    left, right = arr[0: n // 2], arr[n // 2:]
    
    return merge(merge_sort(left), merge_sort(right))

def merge(left, right):
    result = []
    while left and right:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    
    while left:
        result.append(left.pop(0))

    while right:
        result.append(right.pop(0))
    
    return result
```

# 快速排序

快速排序也是一种分治法思想，算是在冒泡排序的基础上的递归分治法。效率高，是处理大数据最快的排序算法之一。
基本操作是从数列中选出一个元素，称为基准（pivot），然后重新排列数列，所有元素比基准小的摆放在基准前面，所有元素比基准大的摆放在基准的后面
这个分区退出之后，该基准就处于数列的中间位置。称为分区（partion）操作。然后进行递归

```python
def quickSort(arr, left=None, right=None):
    left = 0 if not isinstance(left,(int, float)) else left
    right = len(arr)-1 if not isinstance(right,(int, float)) else right
    if left < right:
        partitionIndex = partition(arr, left, right)
        quickSort(arr, left, partitionIndex-1)
        quickSort(arr, partitionIndex+1, right)
    return arr

def partition(arr, left, right):
    pivot = left
    index = pivot+1
    i = index
    while  i <= right:
        if arr[i] < arr[pivot]:
            swap(arr, i, index)
            index+=1
        i+=1
    swap(arr,pivot,index-1)
    return index-1

def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
```

# 堆排序

堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆是一个近似完全二叉树的结构，并同时满足堆的性质，
即子节点的键值或索引总是小于（或者大于）它的父节点，大顶堆用于升序排列，小顶堆用于降序排列。

```python
def heap_sort(arr):
    n = len(arr)
    build_heap(arr)
    for i in range(n - 1, 0, -1):
        swap(arr, i, 0)
        heapify(arr, 0, i)
    return arr

def build_heap(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1): # 堆排序建堆从一半开始排序
        heapify(arr, i, n)

def heapify(arr, i, n): # 保证堆的有序性
    left = 2 * i + 1
    right = 2 * i + 2
    largest = i
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        swap(arr, largest, i)
        heapify(arr, largest, n)
     
def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]
```

:::tip[拓展]
以下三种排序作为拓展
:::

# 计数排序

计数排序的核心在于将输入的数据值转化为键储存在额外开辟的数组空间中。要求输入的数据必须是有确定范围的整数。

```python
def counting_sort(arr, max_value):
    bucket_len = max_value + 1
    bucket = [0] * bucket_len
    sorted_index = 0
    arr_len = len(arr)
    for i in range(arr_len):
        if not bucket[arr[i]]:
            bucket[arr[i]] = 0
        bucket[arr[i]] += 1
    for j in range(bucket_len):
        while bucket[j] > 0:
            arr[sorted_index] = j
            sorted_index += 1
            bucket[j] -= 1
    return arr
```

# 桶排序

桶排序（Bucket Sort）是一种排序算法，它的基本思想是将数据分到有限数量的桶中，
每个桶再单独排序（桶内元素通常使用其他排序算法进行排序，如快速排序、归并排序等），然后再将所有桶中的数据合并起来。

```python
def bucket_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Step 1: 找到数组的最小值和最大值
    min_val, max_val = min(arr), max(arr)
    
    # Step 2: 根据数据的范围划分桶的数量
    bucket_count = len(arr)
    bucket_size = (max_val - min_val) / bucket_count
    
    # Step 3: 创建桶
    buckets = [[] for _ in range(bucket_count)]
    
    # Step 4: 将数据分配到不同的桶中
    for num in arr:
        # 计算元素属于哪个桶
        index = int((num - min_val) // bucket_size)
        if index == bucket_count:  # 防止 max_val 正好落在最大桶的边界上
            index -= 1
        buckets[index].append(num)
    
    # Step 5: 对每个桶内的元素进行排序（这里使用 Python 内置的 sort 方法）
    for bucket in buckets:
        bucket.sort()
    
    # Step 6: 合并所有桶中的元素
    sorted_arr = []
    for bucket in buckets:
        sorted_arr.extend(bucket)
    
    return sorted_arr
```

# 基数排序

基数排序是一种非比较型整数排序算法，其原理是将整数按位数切割成不同的数字，然后按每个位数分别比较。
由于整数也可以表达字符串（比如名字或日期）和特定格式的浮点数，所以基数排序也不是只能使用于整数。

```python
def radix(arr):
    
    digit = 0
    max_digit = 1
    max_value = max(arr)
    #找出列表中最大的位数
    while 10**max_digit < max_value:
        max_digit = max_digit + 1
    
    while digit < max_digit:
        temp = [[] for i in range(10)]
        for i in arr:
            #求出每一个元素的个、十、百位的值
            t = int((i/10**digit)%10)
            temp[t].append(i)
        
        coll = []
        for bucket in temp:
            for i in bucket:
                coll.append(i)
                
        arr = coll
        digit = digit + 1

    return arr
```

