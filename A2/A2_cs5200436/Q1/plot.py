import os
import matplotlib.pyplot as plt
import time
import sys
gspan_executable = "./executables/gSpan-64"
fsg_executable = "./executables/fsg"
gaston_executable = "./executables/gaston-1.1/gaston"

# Define the list of s values
s_values = [0.05, 0.1, 0.25, 0.50, 0.95]
# s_values = [0.70 , 0.95 ]

thresholds = [s * 100 for s in s_values]  

# Lists to store runtimes
runtimes_fsg = []
runtimes_gspan = [  ]
runtimes_gaston = [ ]


if len(sys.argv) < 2:
    print("python3 plot.py <output destination>")
else:
    filename = sys.argv[1]

# # Iterate over each s value
for s in s_values:
    # ---------------------------Gspan------------------------------------
    start_time_gspan = time.time()

    command_gspan = f"{gspan_executable} -f Yeast_Gspan.txt -s {s}"
    try:
        exit_code_gspan = os.system(command_gspan)
        if exit_code_gspan == 0:
            end_time = time.time()
            runtime = end_time - start_time_gspan
            runtimes_gspan.append(runtime)
            print(f"Command for s={s} executed successfully. Runtime: {runtime} seconds")
    except  Exception as e:
        print(f"Error executing command for s={s}: {e}")




    # ---------------------------FSG------------------------------------
    command_fsg = f"{fsg_executable} Yeast_FSG.txt -s {s*100}"

    start_time_gspan = time.time()
    try:
        exit_code_fsg = os.system(command_fsg)
        if exit_code_fsg == 0:
            end_time = time.time()
            runtime = end_time - start_time_gspan
            runtimes_fsg.append(runtime)
            print(f"Command for s={s} executed successfully. Runtime: {runtime} seconds")

    except  Exception as e:
        print(f"Error executing command for s={s}: {e}")


    # ---------------------------Gaston------------------------------------
    print("running gaston")
    command_gaston = f"{gaston_executable}  {64109*s} Yeast_Gaston.txt "
    start_time_gspan = time.time()
    try:
        exit_code_gaston = os.system(command_gaston)
        if exit_code_gaston == 0:
            end_time = time.time()
            runtime = end_time - start_time_gspan
            runtimes_gaston.append(runtime)
            print(f"Command for s={s} executed successfully. Runtime: {runtime} seconds")
        else:
            print(f"Error executing command for s={s}")
    except Exception as e:
        print(f"Error executing command for s={s}: {e}")

plt.plot(thresholds, runtimes_fsg, label='FSG', marker='o')  # Adding dots with 'o' marker
plt.plot(thresholds, runtimes_gspan, label='GSpan', marker='s')  # Adding squares with 's' marker
plt.plot(thresholds, runtimes_gaston, label='Gaston', marker='^')  # Adding triangles with '^' marker

plt.xlabel('Threshold (%)')
plt.ylabel('Runtime (seconds)')
plt.title('Runtime vs. Threshold')
plt.grid(True)
plt.legend()  # Adding legend

# Save the plot as an image file
plt.savefig(filename)

# Show the plot
plt.show()
