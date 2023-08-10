from qiskit import QuantumCircuit
from qiskit.test.mock import FakeQasmSimulator
from qiskit.compiler import transpile

from compute_empirical_complexity import get_data
from data_prep import DataPrep


def get_qiskit_circuit_stats(dense_statevector, qubits_count, basis_gates=['u1', 'u2', 'u3', 'cx'],
                             optimization_level=3):
    """
    :return: circuit depth, count of individual gate types
    """
    backend_architecture = FakeQasmSimulator()
    cfg = backend_architecture.configuration()
    qc = QuantumCircuit(qubits_count)  # Create a quantum circuit with one qubit
    qc.initialize(dense_statevector, qc.qubits)  # Initialize the circuit
    optimized_circuit = transpile(qc, coupling_map=cfg.coupling_map, basis_gates=basis_gates,
                                  optimization_level=optimization_level)
    return optimized_circuit.depth(), optimized_circuit.count_ops()


if __name__ == "__main__":
    files = [
        {"file_name": "../datasets/balance_scale.csv", "label_location": "first", 'labels': ['R'],
         'is_laborious': False},
        {"file_name": "../datasets/tictactoe.csv", "label_location": "last", 'labels': ['positive'],
         'is_laborious': True},
        {"file_name": "../datasets/breast_cancer.csv", "label_location": "last", "remove_columns": [0], 'labels': [2],
         'is_laborious': True},
        {"file_name": "../datasets/zoo.csv", "label_location": "last", "remove_columns": [0], 'labels': [1],
         'is_laborious': True},
        {"file_name": "../datasets/SPECTrain.csv", "label_location": "first", 'labels': [1], 'is_laborious': False}
    ]

    encoding = 'label'

    for file in files:
        if "remove_columns" in file:
            remove_columns = file["remove_columns"]
        else:
            remove_columns = None

        classes, max_attr_count, features_count = get_data(file["file_name"],
                                                           label_location=file["label_location"],
                                                           encoding=encoding,
                                                           columns_to_remove=remove_columns)
        for label in classes:
            if 'labels' in file:  # process only a subset of labels present in file['labels']
                if label not in file['labels']:
                    continue
            # FIXME: right now we will remove duplicate observations.
            #        in the future we can change the normalization factors according to the number of occurrences
            database = classes[label]

            # TODO: add suffix with u or c register as well as h register

            dp = DataPrep(db=database, is_binary=False, symbol_count=max_attr_count)

            # save mathematics sparse array to a file
            with open(f"{file['file_name']}.label.{label}.mathematica_sparse_array", 'w') as f:
                f.write(dp.to_mathematica())

            # TODO: uncomment to try to get the circuit from the statevector
            # circuit_depth, gate_count = get_qiskit_circuit_stats(dp.to_qiskit_dense_statevector(), dp.qubits_count,
            #                                                      basis_gates=['rx', 'ry', 'rz', 'cx'],
            #                                                      optimization_level=3)
            # print(f"Depth {circuit_depth}")
            # print(f"Gate count: {gate_count}")

