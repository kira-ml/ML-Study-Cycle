import numpy as np

# 🎯 Welcome to Vector Academy! Let's break down numpy arrays like a pro.
# Today's lesson: Understanding .size vs .shape vs len() - no cap.

# First up, let's create a basic vector (a 1D array)
# Think of it as a shopping list, but for numbers.
v = np.array([1, 2, 3, 4, 5])

print("📦 Our Vector Setup:")
print(f"v = {v}")
print(f"Type of v: {type(v)} (it's a numpy.ndarray - the GOAT data structure)")
print()  # Breathing room for your brain


# 🔢 Part 1: .size - The Total Count
# -----------------------------------
# .size tells you: "How many total items are in this array?"
# Like counting all items in your shopping cart.
vector_size = v.size

print("🧮 Part 1: Vector Size (.size attribute)")
print(f"v.size = {vector_size}")
print(f"Translation: There are {vector_size} elements in vector v")
print(f"Quick check: v = {v} literally has {vector_size} numbers: {list(v)}")
print("(Like counting all items in your online cart 🛒)\n")


# 📐 Part 2: .shape - The Dimensional Blueprint
# ---------------------------------------------
# .shape tells you: "What's the structure/dimensions of this array?"
# Like knowing your closet has 3 shelves with 5 shirts each.
vector_shape = v.shape

print("📐 Part 2: Vector Shape (.shape attribute)")
print(f"v.shape = {vector_shape}")
print(f"Vibe check: This is a {len(vector_shape)}-dimensional array")
print(f"The lonely comma in {vector_shape} says: 'I'm a tuple, not just a number'")

print("\n🧠 Shape notation explained:")
print(f"v.shape = {vector_shape} means:")
print("  - Dimension 1️⃣ : It's a vector (1D array, no rows/columns drama)")
print(f"  - Size 🎯 : That single dimension has {vector_shape[0]} spots")
print("  - The comma 📍 : It's Python's way of saying 'tuple vibes only'")
print("  (Tuples are like lists but immutable - they don't change, period.)\n")


# 🔗 Part 3: The .size vs .shape Relationship
# --------------------------------------------
# For 1D arrays: .size = .shape[0] (they match!)
# For 2D arrays: .size = .shape[0] × .shape[1] (they multiply!)
print("🔗 Part 3: .size vs .shape - Bestie Relationship Status")
print(f"v = {v}")
print(f"v.size = {v.size}  ← Total element count")
print(f"v.shape = {v.shape}  ← Dimensional layout")

print(f"\nFor 1D vectors only: v.size = v.shape[0]")
print(f"Math check: {v.size} = {v.shape[0]}")
print(f"True? {v.size == v.shape[0]} ✅ (They're twins for 1D arrays!)\n")


# ⚖️ Part 4: NumPy vs Python - The Showdown
# ------------------------------------------
# Python's len() vs NumPy's .size/.shape - who wins?
print("⚖️ Part 4: NumPy Attributes vs Python's len() - Face Off")
print(f"v.size = {v.size}  ← NumPy's total count (includes everything)")
print(f"len(v) = {len(v)}  ← Python's first-dimension only")

print("\nFor 1D vectors, they give the same answer:")
print(f"len(v) = {len(v)}, v.shape[0] = {v.shape[0]}")
print(f"Equal? {len(v) == v.shape[0]} ✅")
print("(It's like asking 'How many songs in this playlist?' vs 'How many songs total?' - same answer for 1D!)\n")


# 🚨 Important Warning: 2D Arrays Change Everything
# -------------------------------------------------
# len() only looks at the FIRST dimension! This is crucial for matrices.
print("🚨 PART 5: CRITICAL WARNING - 2D Arrays Are Different!")
print("len() only checks the FIRST dimension. Don't get played by this!")

# Creating a matrix (2D array) - think of it as a spreadsheet
matrix = np.array([
    [1, 2, 3],   # Row 1
    [4, 5, 6]    # Row 2
])

print(f"\nmatrix = \n{matrix}")
print(f"matrix.shape = {matrix.shape}  ← (2 rows, 3 columns) ← The whole blueprint")
print(f"matrix.size = {matrix.size}  ← (2 × 3 = 6 total elements) ← Everything counted")
print(f"len(matrix) = {len(matrix)}  ← Only gives 2 (just the rows!) ← Python's limited view")

print("\n💡 Pro tip: Always use .size for total count, .shape for structure!")
print("len() is fine for lists, but for arrays? .size and .shape are your real ones.\n")


# 💪 Part 6: Practice Time - Flex Those Skills
# --------------------------------------------
print("💪 Part 6: Practice Exercises - Test Your Understanding")
print("=" * 60)

# Exercise 1: Create your own vector
print("\n🎯 Exercise 1: Make Your Own Vector")
print("Create a vector with 4 of your favorite numbers")
# Uncomment to try:
# my_vector = np.array([your_numbers_here])
# print(f"Your vector: {my_vector}")
# print(f"Size: {my_vector.size}, Shape: {my_vector.shape}")

# Exercise 2: Compare different vectors
print("\n🔍 Exercise 2: Vector Size/Shape Detective")
print("Let's analyze different vectors to see patterns:")

v1 = np.array([10])  # Single element vector
v2 = np.array([])    # Empty vector (ghost array 👻)
v3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Long boi vector

print(f"\nv1 = {v1} (The minimal vector)")
print(f"  v1.size = {v1.size} ← Just one element")
print(f"  v1.shape = {v1.shape} ← Still a tuple with a comma!")
print("  (Even one element gets the tuple treatment 💁‍♂️)")

print(f"\nv2 = {v2} (The empty vector 👻)")
print(f"  v2.size = {v2.size} ← Zero elements, empty vibes")
print(f"  v2.shape = {v2.shape} ← (0,) means '1D but empty'")
print("  Warning: Empty arrays can break your code if not handled!")

print(f"\nv3 = {v3} (The long vector 📏)")
print(f"  v3.size = {v3.size} ← {len(v3)} elements total")
print(f"  v3.shape = {v3.shape} ← 1D with {len(v3)} spots")
print("  (Size scales with length, shape just shows the structure)")

print("\n" + "=" * 60)
print("🎓 TL;DR Cheat Sheet:")
print("• .size → Total elements (like counting all items in cart)")
print("• .shape → Structure blueprint (rows × columns × depth...)")
print("• len() → Only first dimension (basic Python, not array-aware)")
print("• For 1D: size = shape[0] = len() ← They match!")
print("• For 2D+: size = product of all shape dimensions ← Math time!")
print("\nRemember: .size and .shape are your reliable array besties 💫")