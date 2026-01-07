import numpy as np

# 📊 Part 1: The Scalar Vibe - Single Values Only
# --------------------------------------------------
# A scalar is literally just one number. No dimensions, no arrays, no drama.
# Think: temperature, age, price - single vibes only.

temperature = 98.6  # Just one float, living its best single life

print("🔥 Part 1: Scalar Example - Single Value Energy")
print(f"🌡️  Temperature: {temperature}")
print(f"📁 Type of temperature: {type(temperature)}")
print(f"🤔 Is this a scalar? Yup! It's literally just: {temperature}")
print("   (No shape, no dimensions, just one lonely value doing its thing)\n")


# 📊 Part 2: Vector Flex - Multiple Values, One Array
# ----------------------------------------------------
# Vectors are arrays with attitude - multiple values in an ordered line.
# Think: feature sets, coordinates, or literally any list of numbers.

feature_vector = np.array([2.5, -1.0, 4.7, 0.3])  # Four features chillin' in one array

print("🧮 Part 2: Vector Example - Array Goals")
print(f"🎯 Feature Vector: {feature_vector}")
print(f"📁 Type: {type(feature_vector)} (It's a numpy array - the cool data structure)")
print(f"📐 Shape: {feature_vector.shape}")  
# ^ Shows dimensions as a tuple. (4,) means: "I'm 1D with 4 elements, no cap"
print(f"🔢 Number of elements: {len(feature_vector)} (four features, no less)")
print(f"🤔 Is this a vector? Absolutely! It's {feature_vector}")
print("   (An ordered collection - like a playlist but for numbers)\n")


# 🔍 Part 3: Scalar vs Vector - The Showdown
# -------------------------------------------
# This is where we flex how they're different. Spoiler: Vectors can do more.

print("⚡ Part 3: Scalar vs Vector - Side by Side Comparison")
print("=" * 50)

print("🧊 Scalar (temperature):")
print(f"  💎 Value: {temperature} (just this one number)")
print(f"  📏 Dimension: 0 (it's a point, not a line)")
print(f"  ✨ Can do: temperature + 5 = {temperature + 5}")
print("     (Adding 5 to a scalar? Basic math, but valid!)")

print("\n🎨 Vector (feature_vector):")
print(f"  💎 Value: {feature_vector} (multiple values, main character energy)")
print(f"  📏 Dimension: 1 (it's a line of points - think coordinate axis)")
print(f"  ✨ Can do: feature_vector + 5 = {feature_vector + 5}")
print("     (Broadcasting! Adds 5 to EVERY element - vector magic 🪄)")
print(f"  👆 Can access elements: feature_vector[0] = {feature_vector[0]} (first element)")
print(f"  👆 Can access elements: feature_vector[1] = {feature_vector[1]} (second element)")
print("     (Indexing! Like picking songs from your playlist 🎵)")

print("\n" + "=" * 50)
print("💡 TL;DR:")
print("• Scalar = Single value (like one notification)")
print("• Vector = Multiple values in order (like your entire notification history)")
print("• Both important, but vectors can do cooler math tricks ✨")