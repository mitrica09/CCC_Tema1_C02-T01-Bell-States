
"""
C02-T01 | Generarea si testarea starilor Bell

Obiective:
1. Generarea celor patru stari Bell.
2. Simularea folosind Qiskit Aer.
3. Analiza distributiilor de masurare.
4. Calcularea matricelor de corelatie Pauli 3x3.
5. Compararea rezultatelor experimentale cu cele teoretice.
6. Salvarea rezultatelor si graficelor.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram


# ==================================================
# 1. CONFIGURARE
# ==================================================

simulator = AerSimulator()

SHOTS = 4096

BELL_STATES = ["Phi+", "Phi-", "Psi+", "Psi-"]
BASES = ["X", "Y", "Z"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ==================================================
# 2. GENERAREA STARILOR BELL
# ==================================================

def prepare_bell_state(qc, state_name):
    """
    Pregateste una dintre cele patru stari Bell
    folosind portile Hadamard, CNOT, Z si X.
    """

    # Circuitul de baza
    qc.h(0)
    qc.cx(0, 1)

    # Modificarea starii
    if state_name == "Phi-":
        qc.z(0)

    elif state_name == "Psi+":
        qc.x(1)

    elif state_name == "Psi-":
        qc.z(0)
        qc.x(1)


def create_bell_circuit(state_name):

    qc = QuantumCircuit(2, 2)

    prepare_bell_state(qc, state_name)

    qc.measure([0, 1], [0, 1])

    return qc


# ==================================================
# 3. SIMULAREA CELOR PATRU STARI BELL
# ==================================================

def simulate_bell_states():

    all_results = {}

    print("\nGENERAREA SI SIMULAREA STARILOR BELL")

    for state in BELL_STATES:

        print("\n" + "=" * 45)
        print(f"STAREA BELL: {state}")
        print("=" * 45)

        qc = create_bell_circuit(state)

        print(qc.draw())

        compiled = transpile(qc, simulator)

        result = simulator.run(
            compiled,
            shots=SHOTS
        ).result()

        counts = result.get_counts()

        all_results[state] = counts

        print("Rezultatele masuratorilor:")
        print(counts)

    return all_results


# ==================================================
# 4. HISTOGRAMELE MASURATORILOR
# ==================================================

def plot_bell_histograms(all_results):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8)
    )

    for ax, state in zip(axes.flatten(), BELL_STATES):

        plot_histogram(
            all_results[state],
            ax=ax
        )

        # Setarea explicita a titlului fiecarui grafic
        ax.set_title(
            f"Starea Bell {state}",
            fontsize=12,
            fontweight="bold",
            pad=12
        )

        ax.set_xlabel("Rezultatul masuratorii")

        ax.set_ylabel("Numar de masuratori")

    fig.suptitle(
        "Distributia masuratorilor pentru cele patru stari Bell",
        fontsize=15,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        RESULTS_DIR / "bell_histograms.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ==================================================
# 5. SELECTAREA BAZELOR DE MASURARE PAULI
# ==================================================

def apply_basis_rotation(qc, qubit, basis):
    """
    Transforma masurarea in baza X sau Y
    intr-o masurare standard in baza Z.
    """

    if basis == "X":
        qc.h(qubit)

    elif basis == "Y":
        qc.sdg(qubit)
        qc.h(qubit)

    # Pentru Z nu este necesara nicio rotatie.


# ==================================================
# 6. CALCULAREA CORELATIEI
# ==================================================

def measure_correlation(state_name, basis0, basis1):
    """
    Calculeaza corelatia dintre doi qubiti
    pentru bazele de masurare selectate.
    """

    qc = QuantumCircuit(2, 2)

    # Pregatirea starii Bell
    prepare_bell_state(qc, state_name)

    # Selectarea bazelor de masurare
    apply_basis_rotation(qc, 0, basis0)
    apply_basis_rotation(qc, 1, basis1)

    # Masurarea qubitilor
    qc.measure([0, 1], [0, 1])

    compiled = transpile(qc, simulator)

    result = simulator.run(
        compiled,
        shots=SHOTS
    ).result()

    counts = result.get_counts()

    # Calculul corelatiei
    correlation = 0

    for bitstring, count in counts.items():

        bits = bitstring.replace(" ", "")

        bit0 = int(bits[-1])
        bit1 = int(bits[-2])

        # Conversia bitilor in valori +1 si -1
        value0 = 1 - 2 * bit0
        value1 = 1 - 2 * bit1

        correlation += value0 * value1 * count

    return correlation / SHOTS


# ==================================================
# 7. CALCULAREA MATRICELOR DE CORELATIE 3x3
# ==================================================

def calculate_correlation_matrices():

    matrices = {}

    print("\n\nMATRICELE EXPERIMENTALE DE CORELATIE PAULI")

    for state in BELL_STATES:

        matrix = np.zeros((3, 3))

        for i, basis0 in enumerate(BASES):

            for j, basis1 in enumerate(BASES):

                matrix[i, j] = measure_correlation(
                    state,
                    basis0,
                    basis1
                )

        matrices[state] = matrix

        print("\n" + "=" * 50)
        print(f"MATRICEA DE CORELATIE - {state}")
        print("=" * 50)

        print("       X       Y       Z")

        for i, basis in enumerate(BASES):

            print(
                f"{basis}   " +
                "   ".join(
                    f"{value:7.3f}"
                    for value in matrix[i]
                )
            )

    return matrices


# ==================================================
# 8. MATRICILE TEORETICE
# ==================================================

THEORETICAL_MATRICES = {

    "Phi+": np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]),

    "Phi-": np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]),

    "Psi+": np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]),

    "Psi-": np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ])
}


# ==================================================
# 9. COMPARAREA REZULTATELOR
# ==================================================

def compare_matrices(experimental_matrices):

    output_lines = []

    output_lines.append(
        "COMPARAREA REZULTATELOR EXPERIMENTALE CU CELE TEORETICE"
    )

    print("\n\nCOMPARAREA REZULTATELOR")

    for state in BELL_STATES:

        experimental = experimental_matrices[state]
        theoretical = THEORETICAL_MATRICES[state]

        error = np.abs(experimental - theoretical)

        max_error = np.max(error)
        mean_error = np.mean(error)

        print("\n" + "=" * 50)
        print(f"STAREA BELL: {state}")
        print("=" * 50)

        print("Matrice teoretica:")
        print(theoretical)

        print("\nMatrice experimentala:")
        print(np.round(experimental, 3))

        print(f"\nEroare absoluta maxima: {max_error:.5f}")
        print(f"Eroare absoluta medie: {mean_error:.5f}")

        output_lines.append(f"\nStarea Bell: {state}")
        output_lines.append("Matrice teoretica:")
        output_lines.append(str(theoretical))
        output_lines.append("Matrice experimentala:")
        output_lines.append(str(np.round(experimental, 4)))
        output_lines.append(f"Eroare maxima: {max_error:.5f}")
        output_lines.append(f"Eroare medie: {mean_error:.5f}")

    return output_lines


# ==================================================
# 10. GRAFICE HEATMAP
# ==================================================

def plot_correlation_matrices(matrices):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 9)
    )

    for ax, state in zip(axes.flatten(), BELL_STATES):

        matrix = matrices[state]

        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        ax.set_title(
            f"Starea Bell {state}",
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))

        ax.set_xticklabels(BASES)
        ax.set_yticklabels(BASES)

        ax.set_xlabel("Baza qubitului 1")
        ax.set_ylabel("Baza qubitului 0")

        for i in range(3):

            for j in range(3):

                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black"
                )

    fig.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="Valoarea corelatiei"
    )

    fig.suptitle(
        "Matricele de corelatie Pauli pentru starile Bell",
        fontsize=14,
        fontweight="bold"
    )

    plt.savefig(
        RESULTS_DIR / "correlation_matrices.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ==================================================
# 11. EXPORTUL MATRICELOR IN CSV
# ==================================================

def export_matrices_to_csv(matrices):

    file_path = RESULTS_DIR / "correlation_results.csv"

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Bell State",
            "Qubit 0 Basis",
            "X",
            "Y",
            "Z"
        ])

        for state in BELL_STATES:

            matrix = matrices[state]

            for i, basis in enumerate(BASES):

                writer.writerow([
                    state,
                    basis,
                    *matrix[i].tolist()
                ])

    print(f"\nMatricele au fost salvate in: {file_path}")


# ==================================================
# 12. SALVAREA RAPORTULUI TEXT
# ==================================================

def save_results(all_results, matrices, comparison_lines):

    file_path = RESULTS_DIR / "results.txt"

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "C02-T01 - GENERAREA SI TESTAREA STARILOR BELL\n"
        )

        file.write(f"\nNumar de shots: {SHOTS}\n")

        file.write("\nREZULTATELE MASURATORILOR\n")

        for state in BELL_STATES:

            file.write(f"\nStarea {state}:\n")
            file.write(str(all_results[state]))
            file.write("\n")

        file.write("\nMATRICELE EXPERIMENTALE\n")

        for state in BELL_STATES:

            file.write(f"\nStarea {state}:\n")
            file.write(str(np.round(matrices[state], 5)))
            file.write("\n")

        file.write("\n")
        file.write("\n".join(comparison_lines))

    print(f"Raportul text a fost salvat in: {file_path}")


# ==================================================
# 13. PROGRAMUL PRINCIPAL
# ==================================================

def main():

    # Generarea si simularea starilor
    all_results = simulate_bell_states()

    # Histograme
    plot_bell_histograms(all_results)

    # Calcularea matricelor
    matrices = calculate_correlation_matrices()

    # Comparatia experimentala cu cea teoretica
    comparison_lines = compare_matrices(matrices)

    # Grafice heatmap
    plot_correlation_matrices(matrices)

    # Export rezultate
    export_matrices_to_csv(matrices)

    save_results(
        all_results,
        matrices,
        comparison_lines
    )

    print("\n" + "=" * 55)
    print("SIMULAREA S-A INCHEIAT CU SUCCES!")
    print("=" * 55)

    print("Stari Bell generate: 4")
    print("Baze de masurare: X, Y, Z")
    print("Combinatii de masurare per stare: 9")
    print("Shots per experiment:", SHOTS)

    print("\nRezultatele sunt disponibile in folderul results.")


if __name__ == "__main__":
    main()
