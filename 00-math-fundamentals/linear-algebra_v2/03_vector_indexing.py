import numpy as np

data = np.array([10, 20, 30, 40, 50])


print(f"Original vector: {data}")


print()


second_element = data[1]

print(f"1. The second element (at index 1) is: {second_element}")
print(f"   Explanation: data[1] gives us the element at position 1")
print()


print("2. Changing the last element (value 50) to 99...")



data[-1] = 99


print(f"   Used: data[-1] = 99  (-1 means 'last element')")

print()
print("Updated vector:", data)
print(f"   Last element is now: {data[-1]}")
