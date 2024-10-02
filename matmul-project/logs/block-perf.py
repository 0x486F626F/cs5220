import re
import numpy as np
import matplotlib.pyplot as plt

# Read the log fileb
log_file = "03-block.log"
with open(log_file, "r") as file:
    lines = file.readlines()

# Initialize variables to store data
data = {}
current_bs = None

# Process each line in the log file
for line in lines:
    # Check for BS= lines
    if line.startswith("BS="):
        current_bs = int(re.search(r"BS=(\d+)", line).group(1))
        data[current_bs] = []
    else:
        # Parse the first column value
        first_col_value = float(line.split()[0])
        data[current_bs].append(first_col_value)

# Calculate averages and variances
bs_values = []
averages = []
variances = []

for bs, values in data.items():
    bs_values.append(bs)
    avg = np.mean(values)
    var = np.var(values)
    averages.append(avg)
    variances.append(var)

# Plotting
plt.errorbar(bs_values, averages, yerr=np.sqrt(variances), fmt='-o', capsize=5)
plt.xlabel("BS (Block Size)")
plt.ylabel("Average GFlop/s")
plt.title("Average GFlop/s vs Block Size with Error Bars")
plt.grid(True)
plt.savefig('03-block.jpg', dpi=300)
