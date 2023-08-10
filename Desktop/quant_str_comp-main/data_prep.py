import logging
import math
from collections.abc import Iterable

from qiskit.opflow import SparseVectorStateFn
from qiskit.quantum_info import Statevector

from create_statevector import CreateStatevector

logger = logging.getLogger(__name__)


class DataPrep:

    def __init__(self, db: Iterable[str], target: str = None, is_binary: bool = False,
                 symbol_length: int = 1, symbol_count: int = None):
        """
        :param db: a set of strings (passed in a list) against which we compare the target string
        :param target: target string
        :param symbol_length: the number of characters that codify a symbol; used only when is_binary == True
        :param symbol_count: the number of characters in the alphabet; used only when is_binary == False.
               the default value is None -- in this case the number of symbols is determined automatically based on the
               number of distinct characters in the `db`. However, we may need to override this number for machine
               learning tasks as a particular dataset may not have all the characters present in all the classes
        :param is_binary: are we dealing with binary strings?
        """

        # create binary representation of strings in the database
        is_target_present = True
        if target is None:
            # TODO: for now we will create dummy target string db[0], but this is not elegant,
            #       rework the design of underlying functions
            target = db[0]
            is_target_present = False

        if is_binary:
            target_bin, string_db = DataPrep.massage_binary_strings(target, db, symbol_length)
        else:
            if symbol_count is not None:
                target_bin, string_db, symbol_length, _ = DataPrep.massage_symbol_strings(target, db, symbol_count)
            else:
                target_bin, string_db, symbol_length, _ = DataPrep.massage_symbol_strings(target, db)

        # FIXME: right now we will remove duplicate observations.
        #        in the future we can change the normalization factors according to the number of occurrences
        self.db_bin = list(set(string_db))
        self.symbol_length = symbol_length

        if is_target_present:
            self.target_bin = target_bin
        else:
            self.target_bin = None

        self.qubits_count = len(string_db[0])

    def to_mathematica(self) -> str:
        """
        Convert `db` (a database of strings) to sparse statevector in mathematica format
        """
        non_zero_statevector_indexes, qubits_count = DataPrep.to_statevector_indexes(self.db_bin)

        # TODO: specify total number of elements
        out = []
        for ind in non_zero_statevector_indexes:
            out.append(f"{{{ind+1},1}} -> 1")
        out = f"SparseArray[{{{', '.join(out)}}}, {{{2**qubits_count},1}}] / {len(non_zero_statevector_indexes)}^(1/2)"
        return out

    def to_qiskit_dense_statevector(self) -> Statevector:
        """
        Convert `db` (a database of strings) to dense (regular) statevector in QisKit format

        :db: Database of binary strings
        :return: dense statevector
        """
        return self.to_qiskit_dense_statevector_static(self.db_bin)

    @staticmethod
    def to_qiskit_dense_statevector_static(db_bin: Iterable[str]) -> Statevector:
        """
        Convert `db` (a database of strings) to dense (regular) statevector in QisKit format

        :db: Database of binary strings
        :return: dense statevector
        """
        non_zero_statevector_indexes, qubits_count = DataPrep.to_statevector_indexes(db_bin)
        return CreateStatevector.create_dense_statevector(non_zero_statevector_indexes, qubits_count)

    def to_qiskit_sparse_statevector(self) -> SparseVectorStateFn:
        """
        Convert `db` (a database of strings) to sparse statevector in QisKit format
        """
        non_zero_statevector_indexes, qubits_count = DataPrep.to_statevector_indexes(self.db_bin)
        return CreateStatevector.create_sparse_statevector(non_zero_statevector_indexes, qubits_count)

    @staticmethod
    def to_statevector_indexes(db: Iterable[str]) -> [list[int], int]:
        """
        :db: Database of binary strings
        :return: a list of non-zero statevector indexes, qubit_count
        """
        non_zero_statevector_indexes = CreateStatevector.convert_db_to_statevector_indexes(db)
        qubits_count = len(db[0])
        return non_zero_statevector_indexes, qubits_count

    @staticmethod
    def massage_binary_strings(target: str, db: Iterable[str], symbol_length: int) -> [str, Iterable[str]]:
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :param symbol_length: length of a symbol
        :return: massaged target and database strings
        """
        # sanity checks
        if not isinstance(target, str):
            raise TypeError("Target string should be of type str")
        for my_str in db:
            if not isinstance(my_str, str):
                raise TypeError(f"Database string {my_str} should be of type str")

        bits_in_str_cnt = len(target)
        symbols_in_str_cnt = bits_in_str_cnt / symbol_length
        if bits_in_str_cnt % symbols_in_str_cnt != 0:
            raise TypeError(f"Possible data corruption: bit_count MOD symbol_length should be 0, but got "
                            f"{bits_in_str_cnt % symbols_in_str_cnt}")

        for my_str in db:
            if len(my_str) != bits_in_str_cnt:
                raise TypeError(
                    f"Target string size is {bits_in_str_cnt}, but db string {my_str} size is {len(my_str)}")

        if not DataPrep.is_str_binary(target):
            raise TypeError(
                f"Target string should be binary, but the string {target} has these characters {set(target)}")

        for my_str in db:
            if not DataPrep.is_str_binary(my_str):
                raise TypeError(f"Strings in the database should be binary, but the string {my_str} "
                                f"has these characters {set(my_str)}")

        return target, db

    @staticmethod
    def massage_symbol_strings(target: str, db: Iterable[Iterable[str]], override_symbol_count: int = None) \
            -> [str, Iterable[str]]:
        """
        Massage binary strings and perform sanity checks

        :param target: target string
        :param db: database of strings
        :param override_symbol_count: number of symbols in the alphabet, if None -- determined automatically
        :return: target string converted to binary format,
                 database strings converted to binary format,
                 length of symbol in binary format,
                 map of textual symbols to their binary representation (used only for debugging)
        """

        # sanity checks
        if not isinstance(target, list):
            raise TypeError("Target string should be of type list")
        for my_str in db:
            if not isinstance(my_str, list):
                raise TypeError(f"Database string {my_str} should be of type list")

        # compute  strings' length
        symbols_in_str_cnt = len(target)
        for my_str in db:
            if len(my_str) != symbols_in_str_cnt:
                raise TypeError(
                    f"Target string has {symbols_in_str_cnt} symbols, but db string {my_str} has {len(my_str)}")

        # get distinct symbols
        symbols = {}
        id_cnt = 0
        for symbol in target:
            if symbol not in symbols:
                symbols[symbol] = id_cnt
                id_cnt += 1
        for my_str in db:
            for symbol in my_str:
                if symbol not in symbols:
                    symbols[symbol] = id_cnt
                    id_cnt += 1

        # override symbol length if symbol count was specified by the user
        dic_symbol_count = len(symbols)
        if override_symbol_count is not None:
            if dic_symbol_count > override_symbol_count:
                raise ValueError(f"Alphabet has at least {dic_symbol_count}, "
                                 f"but the user asked only for {override_symbol_count} symbols")
            dic_symbol_count = override_symbol_count

        # figure out how many bits a symbol needs
        symbol_length = math.ceil(math.log2(dic_symbol_count))
        logger.debug(f"We got {dic_symbol_count} distinct symbols requiring {symbol_length} bits per symbol")

        # convert ids for the symbols to binary strings
        bin_format = f"0{symbol_length}b"
        for symbol in symbols:
            symbols[symbol] = format(symbols[symbol], bin_format)

        # now let's produce binary strings
        # TODO: += is not the most efficient way to concatenate strings, think of a better way
        target_bin = ""
        for symbol in target:
            target_bin += symbols[symbol]

        db_bin = []
        for my_str in db:
            db_str_bin = ""
            for symbol in my_str:
                db_str_bin += symbols[symbol]
            db_bin.append(db_str_bin)

        return target_bin, db_bin, symbol_length, symbols

    @staticmethod
    def is_str_binary(my_str: str) -> bool:
        """
        Check if a string contains only 0s and 1s

        :param my_str: string to check
        :return: True if binary, False -- otherwise
        """
        my_chars = set(my_str)
        if my_chars.issubset({'0', '1'}):
            return True
        else:
            return False
