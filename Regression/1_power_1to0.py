
import numpy as np

ls = list(np.linspace(1, 0, num=20, dtype=float))

visu = [x**x for x in ls]
print(len(visu), len(ls))

import matplotlib.pyplot as plt 

plt.scatter([i for i in range(len(visu))], visu, color="red", label = "Power Values")
plt.plot([i for i in range(len(visu))], visu, color="blue")
plt.title("1^1 TO 0^0")
plt.xlabel("Total Number of Values")
plt.ylabel("Actual Values")
plt.legend(loc="center")
plt.savefig("1to0.png")
plt.show()
