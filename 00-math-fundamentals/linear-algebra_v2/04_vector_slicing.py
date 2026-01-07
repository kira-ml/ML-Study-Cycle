import numpy as np


v = np.array([0, 5, 10, 15, 20, 25, 30])


print(f"Original vector: {v}")
print(f"Shape of v: {v.shape}")


first_three = v[0:3]

print("First 3 elements (v[0:3] or v[:3]):")
print(f" Result: {first_three}")
print(f"   Explanation: v[0:3] means start at index 0, go up to (but not including) index 3")
print(f"   So we get elements at positions: 0 → {v[0]}, 1 → {v[1]}, 2 → {v[2]}\n")



middle_section = v[2:5]
print("2. Elements from index 2 to 4 inclusive (v[2:5]):")
print(f"   Result: {middle_section}")
print(f"   Explanation: v[2:5] means start at index 2, go up to (but not including) index 5")
print(f"   So we get elements at positions: 2 → {v[2]}, 3 → {v[3]}, 4 → {v[4]}\n")




every_other = v[::2]

print("3. Every other element starting from first (v[::2]):")
print(f"   Result: {every_other}")
print(f"   Explanation: v[::2] means start at 0, step by 2")
print(f"   So we get elements at positions: 0 → {v[0]}, 2 → {v[2]}, 4 → {v[4]}, 6 → {v[6]}")



print("\n" + "="*50)
print("ADDITIONAL SLICING EXAMPLES FOR LEARNING:\n")

from_index_3 = v[3:]
print(f"v[3:] → All elements from index 3 to end: {from_index_3}")



last_three = v[-3:]
print(f"v[-3:] → Last 3 elements: {last_three}")

reversed_v = v[::-1]

print(f"v[::-1] → Reverse the array: {reversed_v}")

every_other_from_1 = v[1::2]
print(f"v[1::2] → Every other element starting from index 1: {every_other_from_1}")


print("\n" + "="*50)
print("MEMORY NOTE: Slicing creates VIEWS (not copies)")

# Demonstrate that slices are views
slice_view = v[2:5]
slice_view[0] = 999  # Modify the view
print(f"\nAfter modifying slice_view[0] = 999:")
print(f"Original v changed too: {v}")
print("This happens because slice_view is a VIEW into v's data")

# Reset v for clarity
v[2] = 10
print(f"\nReset v[2] back to 10: {v}")

# To create an independent copy:
slice_copy = v[2:5].copy()
slice_copy[0] = 888
print(f"\nUsing .copy() creates independent array:")
print(f"slice_copy after modification: {slice_copy}")
print(f"Original v unchanged: {v}")