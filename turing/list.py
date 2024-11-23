# Sample list of strings
sample_list = ["apple", "banana", "cherry"]

# NOTE: ADD MORE ITEMS TO A LIST
# 1. append
sample_list.append(10)
print(sample_list)

#2. extend
sample_list.extend(["a", 'b', 'c'])
print(sample_list)

#3. insert
sample_list.insert(3, 'mango')
print(sample_list)


# NOTE: REMOVE ITEMS FROM A LIST
# 1. pop
sample_list.pop()
print(sample_list)

#2. remove
sample_list.remove('apple')
print(sample_list)

#3. clear
sample_list.clear()
print(sample_list)

# NOTE: SEARCHING AND COUNTING
# 1. count
print(f"Apple Count: {sample_list.count('apples')}")

#2. index
print(f"Index of 1 is: {sample_list.index(1)}")

# NOTE: SORTING AND REVERSING
#1. sort
sample_list.sort()
print(sample_list)
sample_list.sort(reverse=True)
print(sample_list)