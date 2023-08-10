import unittest
from data_prep import DataPrep


class MyTestCase(unittest.TestCase):
    def test_to_mathematica(self):
        dp = DataPrep(db=[['ab', 'bc'], ['bc', 'cd']])
        self.assertEqual('SparseArray[{{2,1} -> 1, {7,1} -> 1}, {16,1}] / 2^(1/2)', dp.to_mathematica())

        dp = DataPrep(db=[['ab', 'bc'], ['bc', 'cd'], ['bc', 'ef']])
        self.assertEqual('SparseArray[{{2,1} -> 1, {7,1} -> 1, {8,1} -> 1}, {16,1}] / 3^(1/2)', dp.to_mathematica())

    def test_to_statevector_indexes(self):
        nz_ind, qubit_count = DataPrep.to_statevector_indexes(['000', '001', '011', '111'])
        self.assertEqual([0, 1, 3, 7], nz_ind)
        self.assertEqual(3, qubit_count)


if __name__ == '__main__':
    unittest.main()
