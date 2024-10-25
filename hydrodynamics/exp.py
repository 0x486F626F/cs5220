import os
import subprocess

def check_job_count(username):
    # Execute the squeue command
    try:
        result = subprocess.run(
            ['squeue', '-u', username, '--noheader', '--format=%t'],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing squeue: {e.stderr}")
        return False

    # Parse the output
    job_states = result.stdout.strip().split('\n')
    #print(f"Job states: {job_states}")

    # Count Pending and Running jobs
    count = sum(1 for state in job_states if state in ('PD', 'R'))

    # Check if the count is less than five
    if count < 5:
        return True
    else:
        return False


# List for 'i' loop
i_values = [8, 16, 32, 64]
k_values = [0.2, 0.4, 0.6, 0.8, 1.0]

# Loop through values of 'i' and 'j'
for k in k_values:
    for i in i_values:
        for j in range(1, 3):  # 'j' ranges from 1 to 5
            # Create the filename for each file
            filename = f"submit_par_{i}_{i}_{j}.sub"
        
            # Define the content of the file
            content = f"""#!/bin/bash
#SBATCH -J sph
#SBATCH -o ./results/sph_par_{k}_{i}_{i}_{j}.out
#SBATCH -e ./results/sph_par_{k}_{i}_{i}_{j}.err
#SBATCH -A m4776
#SBATCH -C cpu
#SBATCH -c {i}
#SBATCH --qos=debug
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --hint=nomultithread

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS={i}
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
make exe
srun ./sph.x -s {k}
"""
            # Write the content to the file
            with open(filename, 'w') as file:
                file.write(content)
        
            # Submit the generated file using sbatch
            while not check_job_count('ktai'):
                pass
            
            os.system(f"sbatch {filename}")
            os.remove(filename)

print("Files generated and submitted successfully!")


