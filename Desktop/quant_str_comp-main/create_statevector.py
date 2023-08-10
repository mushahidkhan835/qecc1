from collections.abc import Iterable
from math import sqrt

from qiskit.quantum_info import Statevector
from qiskit.opflow import SparseVectorStateFn
import numpy as np
from scipy.sparse import bsr_matrix


class CreateStatevector:

    @staticmethod
    def map_bitstring_to_statevector_index(bitstring: str) -> int:
        """
        We map bitstring to the index of the statevector by simply converting bitstring to an integer.
        For example `001` will map to `1`, `011` will map to `3`, and `111` will map to `8`.

        Note that Python 3 has no limit on the maximum size of an integer. Under the hood, it will use the library
        for representing arbitrary large integers.

        :param bitstring: the string of bits
        :return: index of the element in the statevector
        """
        return int(bitstring, 2)

    @staticmethod
    def convert_db_to_statevector_indexes(db: Iterable[str]) -> list[int]:
        """
        Convert a list of objects (a.k.a. strings, a.k.a. patterns) in the database into the list of statevector indexes.

        :param db: a list of strings containing only `0` and `1` characters
        :return: a list of non-zero statevector indexes
        """
        return list(map(CreateStatevector.map_bitstring_to_statevector_index, db))

    @staticmethod
    def get_normalization_factor(count_of_objects_in_the_database: int) -> float:
        """
        Compute normalization factor for every object (a.k.a. string, a.k.a. pattern) in the database.

        For now, we will assume that each object appears in the database exactly once. Thus, each object has the same
        probability value.

        Later, we can extend it as follows. Suppose a database looks as follows: `['01', '01', '11']`.
        Then, we can assign p-value of `2/3` to object `01` and `1/3` to object `11`.

        :param count_of_objects_in_the_database: count of objects in the database
        :return: normalization factor
        """
        return 1 / sqrt(count_of_objects_in_the_database)

    @staticmethod
    def create_dense_statevector(non_zero_indexes: Iterable[int], qubits_count: int) -> Statevector:
        """
        Create dense (i.e., regular) statevector.

        :param non_zero_indexes: a list of indexes in the statevector that correspond to the objects in the database
        :param qubits_count: the number of qubits equivalent to the number of bits in an object residing in the database
        :return: statevector
        """
        # create an array representing all possible states
        total_number_of_states = 2 ** qubits_count
        my_states = np.zeros(total_number_of_states)

        # fill positions of objects with the value of `normalization_factor * 1`
        normalization_factor = CreateStatevector.get_normalization_factor(len(non_zero_indexes))
        for ind in non_zero_indexes:
            my_states[ind] = normalization_factor

        # now let us convert my states to statevector
        return Statevector(my_states)

    @staticmethod
    def create_sparse_statevector(non_zero_indexes: Iterable[int], qubits_count: int) -> SparseVectorStateFn:
        """
        Create sparse statevector.

        :param non_zero_indexes: a list of indexes in the statevector that correspond to the objects in the database
        :param qubits_count: the number of qubits equivalent to the number of bits in an object residing in the database
        :return: statevector
        """
        # create a sparse array representing all possible states
        non_zero_indexes_count = len(non_zero_indexes)
        normalization_factor = CreateStatevector.get_normalization_factor(non_zero_indexes_count)
        total_number_of_states = 2 ** qubits_count

        row = np.array(np.zeros(non_zero_indexes_count), dtype=np.int8)
        col = np.array(non_zero_indexes)
        data = np.full((non_zero_indexes_count, 1), normalization_factor).flatten()
        my_states = bsr_matrix((data, (row, col)), shape=(1, total_number_of_states))

        # now let us convert my states to statevector
        return SparseVectorStateFn(my_states)
