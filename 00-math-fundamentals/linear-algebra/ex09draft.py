import numpy as np
import matplotlib.pyplot as plt


flips = np.random.randint(0, 2, size=10000)

cumulative_heads = np.cumsum(flips)

trial_numbers = np.arange(1, 10001)
running_average = cumulative_heads / trial_numbers


plt.figure(figsize=(10, 6))
plt.plot(trial_numbers, running_average, label="Running Average")
plt.axhline(y=0.5, color="r", linestyle="--", label="True Mean: (0.5)")
plt.xlabel("Number of Trials")
plt.ylabel("Average")
plt.title("Law of Large Numbers: Convergence of Coin Flip Average to 0.5")
plt.legend()
plt.grid(True)
plt.show()