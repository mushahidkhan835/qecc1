from qiskit import QuantumCircuit, Aer
from qiskit import execute

from create_statevector import CreateStatevector


def main():
    """
    Main driver
    """
    db = ['011', '111']
    print(f"Create db with {len(db)} strings: {db}")
    non_zero_statevector_indexes = CreateStatevector.convert_db_to_statevector_indexes(db)
    qubits_count = len(db[0])
    print(f"We will need {qubits_count} qubits: {db}")

    # Let's set the state using dense vector
    dense_statevector = CreateStatevector.create_dense_statevector(non_zero_statevector_indexes, qubits_count)
    print("Raw dense Statevector looks like this:")
    print(dense_statevector)
    qc = QuantumCircuit(qubits_count)  # Create a quantum circuit
    qc.initialize(dense_statevector, qc.qubits)  # Initialize the circuit
    print("And the circuit for dense Statevector look like this:")
    print(qc)

    sim = Aer.get_backend('aer_simulator')
    qc.save_statevector()  # Tell simulator to save statevector
    result = execute(qc, sim).result()  # Do the simulation and return the result
    out_state = result.get_statevector()
    print(f"The output state vector is {out_state}")
    out_counts = result.get_counts()
    print(f"The output p-values is {out_counts}")

    # Let's repeat the exercise for the sparse vector
    sparse_statevector = CreateStatevector.create_sparse_statevector(non_zero_statevector_indexes, qubits_count)
    print(sparse_statevector)
    print("Direct circuit creation will result in failure. Uncomment the commands below to see the error.")
    # qc = QuantumCircuit(qubits_count)  # Create a quantum circuit
    # qc.initialize(sparse_statevector)  # Apply initialisation operation
    # print(qc)


if __name__ == "__main__":
    main()
