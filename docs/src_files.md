## Modules
- `src/monitored_clifford.py`: Module for simulating dynamics and computing correlations in Clifford circuits.
- `src/monitored_haar.py`: Module for simulating dynamics and computing correlations in Haar-random circuits.
- `src/measurement_only.py`: Module for simulating entanglement dynamics and computing the QFI density in measurement-only circuits (XX and Z).
- `src/simulated_annealing.py`:  Module to perform simulated annealing using the QFI density as the cost function.
- `src/monitored_structured`: Module to simulate monitored structured random quantum circuits.
- `src/ED_symmetries/*_circuits`: Module to calculate Z2 and U1 symmetric circuits using exact diagonalization (ED).

## Scripts
- `src/annealing_clifford.py`: Script to perform simulated annealing on correlation functions generated in monitored Clifford circuits using the QFI density as the cost function.
- `src/annealing_haar.py`: Script to perform simulated annealing on correlation functions generated in monitored Haar circuits using the QFI density as the cost function.
- `src/correlations_clifford.py`: Script for simulating monitored Clifford circuits, which are quantum circuits composed of Clifford gates and measurements, and saving the connected correlations of the final state in a single random realization.
- `src/correlations_haar.py`: Script for simulating monitored Haar circuits, which are quantum circuits composed of Haar gates and measurements, and saving the connected correlations of the final state in a single random realization.
- `src/qfi_meas-only.py`: Script to calculate the QFI density in measurement-only circuits (XX vs Z). 