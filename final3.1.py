import numpy as np
import random
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit
import time 

# Constants and settings
NO_SOLUTION = 1000
CONVERGENCE_TOL = 1e-1
MAX_ITERS = 1000

# Define minimum and maximum allowed number of plies
MIN_N = 1
MAX_N = 100

thickness = 0.000125 #thickness of single ply
Mat_E1 = 38610.64084e6
Mat_E2 = 13789.51459e6
Mat_G12 = 4826.33011e6
Mat_v12 = 0.2
Mat_Xt = 206.84272e6
Mat_Xc = 27.57903e6
Mat_Yt = 241.31651e6
Mat_Yc = 62.05282e6
Mat_S = 103.42136e6

Target_FI = 1
Target_Ex = 30e9
Target_Ey = 20e9
Target_Gxy = 6e9

df = pd.read_csv("Forces1.csv")
Flux_Nx = df['Nx'].values
Flux_Ny = df['Ny'].values
Flux_Nxy = df['Nxy'].values
Flux_Mx = df['Mx'].values
Flux_My = df['My'].values
Flux_Mxy = df['Mxy'].values
l = len(Flux_Mx)

class CompositeLaminate:
    def __init__(self, ply_props, ply_angles, thicknesses):
        self.ply_props = ply_props
        self.ply_angles = np.radians(ply_angles)
        self.thicknesses = np.array(thicknesses)
        self.num_plies = len(ply_props)
        self.h_total = sum(thicknesses)
        self.z = self._calculate_z_coordinates()
        self.A, self.B, self.D = self._calculate_abd_matrices()

    def _calculate_z_coordinates(self):
        z = [-self.h_total / 2]
        for t in self.thicknesses:
            z.append(z[-1] + t)
        return np.array(z)

    def _calculate_q_bar(self, E1, E2, G12, v12, theta):
        v21 = (E2 * v12) / E1
        Q11 = E1 / (1 - v12 * v21)
        Q22 = E2 / (1 - v12 * v21)
        Q12 = (v12 * E2) / (1 - v12 * v21)
        Q66 = G12
        Q_matrix = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]])

        c, s = np.cos(theta), np.sin(theta)
        T = np.array([[c**2, s**2, 2*c*s],
                      [s**2, c**2, -2*c*s],
                      [-c*s, c*s, c**2 - s**2]])
        tr_en_strain = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
        T_inv = np.linalg.inv(T)
        tren_strain_inv = np.linalg.inv(tr_en_strain)
        return T_inv @ Q_matrix @ tr_en_strain @ T @ tren_strain_inv

    def _calculate_abd_matrices(self):
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for i, (props, theta) in enumerate(zip(self.ply_props, self.ply_angles)):
            Q_bar = self._calculate_q_bar(*props, theta)
            z_upper = self.z[i + 1]
            z_lower = self.z[i]
            A += Q_bar * (z_upper - z_lower)
            B += 0.5 * Q_bar * (z_upper**2 - z_lower**2)
            D += (1/3) * Q_bar * (z_upper**3 - z_lower**3)

        return A, B, D

    def get_abd_matrix(self):
        return np.block([[self.A, self.B], [self.B, self.D]])
    
    def get_Modulus(self):
        A11 = self.A[0,0]
        A12 = self.A[0,1]
        A22 = self.A[1,1]
        A16 = self.A[0,2]
        A26 = self.A[1,2]
        A66 = self.A[2,2]

        Exh = A11 + (A12*((A26*A16)-(A12*A66))/((A22*A66)-A26**2)) + (A16*((-A16/A66)+((A26*A12*A66)-(A26*A16**2)/((A22*A66**2)-(A66*A26**2)))))
        Eyh = A22 + (A12*((A26*A16)-(A12*A66))/((A11*A66)-A16**2)) + (A16*((-A26/A66)+((A16*A12*A66)-(A16*A26**2)/((A11*A66**2)-(A66*A16**2)))))
        Gxyh = A66 - (A26**2/A22) + (((2*A12*A16*A26*A22)-(A12**2*A16**2)-(A16**2*A22**2))/((A11*A22**2)-(A22*A12**2)))

        return Exh, Eyh, Gxyh

    def compute_stresses(self, Nx, Ny, Nxy, Mx, My, Mxy):
        force_resultant = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
        midplane_strains_curvatures = np.linalg.inv(self.get_abd_matrix()) @ force_resultant
        strains = midplane_strains_curvatures[:3]
        curvatures = midplane_strains_curvatures[3:]

        stresses = []
        for i, (props, theta) in enumerate(zip(self.ply_props, self.ply_angles)):
            Q_bar = self._calculate_q_bar(*props, theta)
            z_upper = self.z[i + 1]
            z_lower = self.z[i]
            z_range = np.linspace(z_lower, z_upper, 2)
            for z in z_range:
                strain = strains + curvatures * z
                stress = Q_bar @ strain
                stresses.append(stress)
        return np.array(stresses)

@njit
def max_stress_criteria(stress_array):

    max_positive1 = max(stress_array[:,0])
    max_positive2 = max(stress_array[:,1])
    
    max_negative1 = max(-stress_array[:,0])
    max_negative2 = max(-stress_array[:,1])
    shear_stresses = stress_array[:,2]  # assuming this is an array
    
    # Max of absolute values for negative shear stresses
    max_negative = np.max(-shear_stresses[shear_stresses < 0]) if np.any(shear_stresses < 0) else 0
    
    # Max positive shear stress
    max_positive = np.max(shear_stresses[shear_stresses > 0]) if np.any(shear_stresses > 0) else 0
    
    # Final max12 is the bigger of the two
    max12 = max(max_negative, max_positive)

    NI = max(max_positive1 / Mat_Xt, max_positive2 / Mat_Yt, max12 / Mat_S, max_negative1/Mat_Xc, max_negative2/Mat_Yc)
    return NI

@njit
def TsaiWu(stress_array, Xt, Xc, Yt, Yc, S):

    F1 = (1/Xt) - (1/Xc)
    F2 = (1/Yt) - (1/Yc)
    F11 = 1/(Xt*Xc)
    F22 = 1/(Yt*Yc)
    F66 = 1/(S**2)
    F12 = -0.5 * (F11 * F22)**0.5  # often assumed
    sigma1 = stress_array[:, 0]  # all sigma1 values
    sigma2 = stress_array[:, 1]  # all sigma2 values
    tau12 = stress_array[:, 2]   # all tau12 values
    FI = (F1 * sigma1) + (F2 * sigma2) + (F11 * sigma1**2) + (F22 * sigma2**2) + (F66 * tau12**2) + (2 * F12 * sigma1 * sigma2)
    return max(FI)

@njit
def TsaiHill(stress_array, Xt, Xc, Yt, Yc, S):

    X = min(Xc, Xt)
    Y = min(Yc, Yt)
    sigma1 = stress_array[:, 0]  # all sigma1 values
    sigma2 = stress_array[:, 1]  # all sigma2 values
    tau12 = stress_array[:, 2]   # all tau12 values

    FI = (sigma1/X)**2 - (sigma1*sigma2)/(X**2) + (sigma2/Y)**2 + (tau12/S)**2
    return max(FI)

@njit
def Hoffman(stress_array, Xt, Xc, Yt, Yc, S):

    sigma1 = stress_array[:, 0]  # all sigma1 values
    sigma2 = stress_array[:, 1]  # all sigma2 values
    tau12 = stress_array[:, 2]   # all tau12 values

    FI = (sigma1 / Xt) - (sigma1 / Xc) + (sigma2 / Yt) - (sigma2 / Yc) + (sigma1 / Xt)**2 + (sigma2 / Yt)**2 + (tau12 / S)**2 - (sigma1 * sigma2) / (Xt * Yt)
    return max(FI)

def is_balanced(sequence):
    # Check if laminate is balanced:
    # For every non-zero, non-90 angle, count + and - plies
    counts = Counter()
    for angle in sequence:
        if abs(angle) not in [0, 90]:
            counts[abs(angle)] += angle // abs(angle)  # +1 or -1

    # Balanced if sum for each angle is zero
    return all(count == 0 for count in counts.values())

def generate_symmetric_balanced_laminate(n):
    base_angles = [0, 45, 90]

    if n % 2 == 0:
        half_n = n // 2

        # Keep generating until balanced laminate is found
        while True:
            half_sequence = []
            for _ in range(half_n):
                base_angle = random.choice(base_angles)
                if base_angle in [0, 90]:
                    half_sequence.append(base_angle)
                else:
                    sign = random.choice([1, -1])
                    half_sequence.append(sign * base_angle)

            # symmetric same sign
            full_sequence = half_sequence + list(reversed(half_sequence))

            if is_balanced(full_sequence):
                return full_sequence

    else:
        half_n = (n - 1) // 2

        # Middle ply must be 0 or 90 to keep symmetry and balance
        middle_ply = random.choice([0, 90])

        while True:
            half_sequence = []
            for _ in range(half_n):
                base_angle = random.choice(base_angles)
                if base_angle in [0, 90]:
                    half_sequence.append(base_angle)
                else:
                    sign = random.choice([1, -1])
                    half_sequence.append(sign * base_angle)

            full_sequence = half_sequence + [middle_ply] + list(reversed(half_sequence))

            if is_balanced(full_sequence):
                return full_sequence
                       
def generate_random_solution():
    n = random.randint(MIN_N, MAX_N)
    ply_props = [(Mat_E1, Mat_E2, Mat_G12, Mat_v12)] * n
    ply_angles = generate_symmetric_balanced_laminate(n)
    thicknesses = [thickness]*n
    return n, ply_props, ply_angles, thicknesses

def fitness_function(n, Ex, Ey, Gxy, FI,
                     Target_Ex, Target_Ey, Target_Gxy, Target_FI):
    dist1 = np.sqrt((1-(Ex/Target_Ex))**2 + (Target_FI-FI)**2)
    dist2 = np.sqrt((1-(Ey/Target_Ey))**2 + (Target_FI-FI)**2)
    dist3 = np.sqrt((1-(Gxy/Target_Gxy))**2 + (Target_FI-FI)**2)

    sum_dist1 = np.sum(dist1)
    sum_dist2 = np.sum(dist2)
    sum_dist3 = np.sum(dist3) 
    
    if 0.9*Target_Ex<=Ex<=1.1*Target_Ex and 0.9*Target_Ey<=Ey<=1.1*Target_Ey and 0.9*Target_Gxy<=Gxy<=1.1*Target_Gxy and 0.8*Target_FI<=FI<=Target_FI:
        w1, w2, w3 = 0.1,0.1,0.1
    else:
        w1, w2, w3 = 100.0,100.0,100.0

    dist = w1*sum_dist1 + w2*sum_dist2 + w3*sum_dist3
    fitness = -n**2*(1+dist)

    return fitness


def evaluate_solution_for_all_loads(n, ply_props, ply_angles, thicknesses, Nx_arr, Ny_arr, Nxy_arr, Mx_arr, My_arr, Mxy_arr):
    laminate = CompositeLaminate(ply_props, ply_angles, thicknesses)
    for Nx, Ny, Nxy, Mx, My, Mxy in zip(Nx_arr, Ny_arr, Nxy_arr, Mx_arr, My_arr, Mxy_arr):
        stresses = laminate.compute_stresses(Nx, Ny, Nxy, Mx, My, Mxy)
        FI1 = max_stress_criteria(stresses)
        FI2 = TsaiWu(stresses, Mat_Xt, Mat_Xc, Mat_Yt, Mat_Yc, Mat_S)
        FI3 = TsaiHill(stresses, Mat_Xt, Mat_Xc, Mat_Yt, Mat_Yc, Mat_S)
        FI4 = Hoffman(stresses, Mat_Xt, Mat_Xc, Mat_Yt, Mat_Yc, Mat_S)
        FI_values = {
        "Max Stress Criteria": FI1,
        "Tsai-Wu": FI2,
        "Tsai-Hill": FI3,
        "Hoffman": FI4
        }

        # Find max FI and its name
        max_criterion = max(FI_values, key=FI_values.get)
        max_value = FI_values[max_criterion]
        FI = max_value


    Exh, Eyh, Gxyh = laminate.get_Modulus()
    Ex = Exh/(n*thickness)
    Ey = Eyh/(n*thickness)
    Gxy = Gxyh/(n*thickness)

    fitness = fitness_function(n, Ex, Ey, Gxy, FI, Target_Ex, Target_Ey, Target_Gxy, Target_FI)

    return Ex, Ey, Gxy, FI, max_criterion, fitness

def evaluate_solution(n, ply_props, ply_angles, thicknesses):
    return evaluate_solution_for_all_loads(n, ply_props, ply_angles, thicknesses,
                                           Flux_Nx, Flux_Ny, Flux_Nxy, Flux_Mx, Flux_My, Flux_Mxy)

def initial_population(no_solution):
    results = []
    for _ in range(no_solution):
        n, ply_props, ply_angles, thicknesses = generate_random_solution()
        Ex, Ey, Gxy, FI, Criteria, fitness = evaluate_solution(n, ply_props, ply_angles, thicknesses)
        results.append((n, ply_angles, thicknesses, Ex, Ey, Gxy, FI, Criteria, fitness))
    # Sort by fitness descending (best first)
    results.sort(key=lambda x: x[-1], reverse=True)
    return results

def mutate_solution(base_sol, delta_n=5):
    n, ply_angles, thicknesses, _, _, _, _, _, _ = base_sol
    
    delta_n = max(1, delta_n)  # ensure delta_n >= 1
    
    delta = random.randint(-delta_n, delta_n)
    new_n = max(MIN_N, min(MAX_N, n + delta))

    mutated_thicknesses = [thickness] * new_n

    mutated_angles = generate_symmetric_balanced_laminate(new_n)
    ply_props = [(Mat_E1, Mat_E2, Mat_G12, Mat_v12)] * new_n

    Ex, Ey, Gxy, FI, Criteria, fitness = evaluate_solution(new_n, ply_props, mutated_angles, mutated_thicknesses)
    return (new_n, mutated_angles, mutated_thicknesses, Ex, Ey, Gxy, FI, Criteria, fitness)
 

def run_evolution():
    results = initial_population(NO_SOLUTION)

    # Save initial results
    with open("ni_results_sorted4.4_best.txt", "w") as f:
        f.write("Plies\tOrientations (deg)\tThicknesses (m)\tEx\tEy\tGxy\tFI\tCriteria\tFitness\n")
        for r in results:
            total_thickness = sum(r[2])
            f.write(f"{r[0]}\t{r[1]}\t{total_thickness:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]}\t{r[8]:.6f}\n")
    print("Sorted results saved to ni_results_sorted4.4_best.txt")

    # Take top 10%
    top_10_percent = results[:max(1, len(results)//10)]
    with open("ni_results_top_10_percent4.4_best.txt", "w") as f:
        f.write("Plies\tOrientations (deg)\tThicknesses (m)\tEx\tEy\tGxy\tFI\tCriteria\tFitness\n")
        for r in top_10_percent:
            total_thickness = sum(r[2])
            f.write(f"{r[0]}\t{r[1]}\t{total_thickness:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]}\t{r[8]:.6f}\n")
    print("Top 10% results saved to ni_results_top_10_percent4.4_best.txt")

    current_population = top_10_percent
    iteration = 0

    best_fitness_history = []
    no_improvement_counter = 0
    mut_n_delta = 5  # start with +-5 delta n for mutation

    # Setup live plot
    plt.ion()
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    max_fi = 1  # max FI value for normalization

    while iteration < MAX_ITERS:
        mutated_results = []

        best_solution = current_population[0]
        best_fitness = best_solution[-1]

        # Check if best fitness is stagnant (same as last iteration)
        if best_fitness_history and abs(best_fitness - best_fitness_history[-1]) < 1e-8:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        best_fitness_history.append(best_fitness)
        reduction_schedule = {
            50: 5,
            110: 4,
            180: 3,
            260: 2,
            360: 1
        }

        for threshold in sorted(reduction_schedule.keys()):
            if no_improvement_counter == threshold:
                new_delta = reduction_schedule[threshold]
                if mut_n_delta > new_delta:
                    mut_n_delta = new_delta
                    print(f"Mutation delta_n reduced to {mut_n_delta} after {no_improvement_counter} stagnant iterations")
                break
        start = time.perf_counter()
        for base_sol in current_population:
            # Generate 9 mutated solutions per base solution, using current mut_n_delta
            for _ in range(9):
                mutated_results.append(mutate_solution(base_sol, delta_n=mut_n_delta))
        end = time.perf_counter()
        print("time = ", end - start)
        combined_population = current_population + mutated_results
        combined_population.sort(key=lambda x: x[-1], reverse=True)

        top_10_count = max(1, len(combined_population) // 10)
        current_population = combined_population[:top_10_count]

        # Normalize values for plotting and distance calculation
        fi_vals = [r[6] for r in current_population]
        ex_vals = [r[3] for r in current_population]
        ey_vals = [r[4] for r in current_population]
        gxy_vals = [r[5] for r in current_population]

        fi_norm = [fi / max_fi for fi in fi_vals]
        ex_norm = [ex / Target_Ex for ex in ex_vals]
        ey_norm = [ey / Target_Ey for ey in ey_vals]
        gxy_norm = [gxy / Target_Gxy for gxy in gxy_vals]

        # Clear previous plots
        for ax in axs:
            ax.cla()

        # Labels and titles
        axs[0].set_xlabel('Normalized FI')
        axs[0].set_ylabel('Normalized Ex')
        axs[0].set_title('Normalized FI vs Ex')

        axs[1].set_xlabel('Normalized FI')
        axs[1].set_ylabel('Normalized Ey')
        axs[1].set_title('Normalized FI vs Ey')

        axs[2].set_xlabel('Normalized FI')
        axs[2].set_ylabel('Normalized Gxy')
        axs[2].set_title('Normalized FI vs Gxy')

        # Scatter points
        axs[0].scatter(fi_norm, ex_norm, color='b', s=10)
        axs[1].scatter(fi_norm, ey_norm, color='g', s=10)
        axs[2].scatter(fi_norm, gxy_norm, color='r', s=10)

        # Vertical lines at FI = 1/max_fi and 0.8/max_fi normalized
        fi_1_norm = 1 / max_fi
        fi_08_norm = 0.8 / max_fi
        for ax in axs:
            ax.axvline(x=fi_1_norm, color='k', linestyle='--', linewidth=1)
            ax.axvline(x=fi_08_norm, color='k', linestyle='--', linewidth=1)

        # Horizontal lines at target and ±10% (normalized to 1, so 0.9 and 1.1)
        targets_norm = [1, 1, 1]  # All target moduli normalized to 1
        colors = ['b', 'g', 'r']
        for i, ax in enumerate(axs):
            ax.axhline(y=targets_norm[i], color=colors[i], linestyle='-', linewidth=1.5, label='Target')
            ax.axhline(y=0.9 * targets_norm[i], color=colors[i], linestyle=':', linewidth=1, label='90% Target')
            ax.axhline(y=1.1 * targets_norm[i], color=colors[i], linestyle=':', linewidth=1, label='110% Target')

            # Line joining origin (0,0) and (FI=1 normalized, target modulus=1)
            ax.plot([0, fi_1_norm], [0, targets_norm[i]], color='k', linestyle='-', linewidth=1)

            ax.legend(loc='upper left', fontsize='small')

        plt.tight_layout()

        # Calculate distance from 45-degree line for each modulus and write to file
        slope = max_fi  # slope of the line y = slope * x in original scale, same in normalized scale here

       #with open(f"distance_iteration_{iteration+1}.txt", "w") as dist_file:
       #    dist_file.write("Plies\tOrientations\tFI\tEx_dist\tEy_dist\tGxy_dist\n")
       #    for r in current_population:
       #        fi_n = r[6] / max_fi
       #        ex_n = r[3] / Target_Ex
       #        ey_n = r[4] / Target_Ey
       #        gxy_n = r[5] / Target_Gxy

       #        dist_ex = abs(slope * fi_n - ex_n) / (slope**2 + 1)**0.5
       #        dist_ey = abs(slope * fi_n - ey_n) / (slope**2 + 1)**0.5
       #        dist_gxy = abs(slope * fi_n - gxy_n) / (slope**2 + 1)**0.5

       #        dist_file.write(f"{r[0]}\t{r[1]}\t{r[6]:.6f}\t{dist_ex:.6e}\t{dist_ey:.6e}\t{dist_gxy:.6e}\n")

        plt.pause(0.01)  # non-blocking plot update

        if (iteration + 1) % 10 == 0:
            # Save top 100 solutions every 50 iterations
            top_100 = current_population[:100]
            filename = f"top_100_iter_{iteration+1}.txt"
            with open(filename, "w") as f:
                f.write("Plies\tOrientations (deg)\tThicknesses (m)\tEx\tEy\tGxy\tFI\tCriteria\tFitness\n")
                for r in top_100:
                    total_thickness = sum(r[2])
                    f.write(f"{r[0]}\t{r[1]}\t{total_thickness:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]}\t{r[8]:.6f}\n")
            print(f"Saved top 100 solutions to {filename} at iteration {iteration+1}")

        print(f"Iteration {iteration+1}: Best fitness = {best_fitness:.8f}, FI = {best_solution[6]:.6f}, Criteria = {best_solution[7]}")
        print(f"              Best Ex = {best_solution[3]:.6e}, Ey = {best_solution[4]:.6e}, Gxy = {best_solution[5]:.6e}")

        if abs(best_fitness) < CONVERGENCE_TOL:
            print("Convergence achieved.")
            break

        iteration += 1

    plt.ioff()
    plt.show()  # Keep the plot open after finishing

    # Save final results
    with open("final_fi_results4.4.7_best.txt", "w") as f:
        f.write("Plies\tOrientations (deg)\tThicknesses (m)\tEx\tEy\tGxy\tFI\tCriteria\tFitness\n")
        for r in current_population:
            total_thickness = sum(r[2])
            f.write(f"{r[0]}\t{r[1]}\t{total_thickness:.6f}\t{r[3]:.6f}\t{r[4]:.6f}\t{r[5]:.6f}\t{r[6]:.6f}\t{r[7]}\t{r[8]:.6f}\n")

    print("Final results saved to final_fi_results4.4_best.txt")

if __name__ == "__main__":
    run_evolution()
