from itertools import chain, combinations
import logging
from typing import Iterable

import pandas as pd
from qiskit.test.mock import FakeQasmSimulator
from qiskit.compiler import transpile

from string_comparison import StringComparator


def int_db_to_bin_str_db(int_db: Iterable[int], qubit_cnt: int) -> list[str]:
    """
    Convert integer db to binary string db
    :param int_db:
    :param qubit_cnt:
    :return:
    """
    return [f'{record:0{qubit_cnt}b}' for record in int_db]

def get_gates_cnt(circuit):
    gates = circuit.count_ops()
    out = {}
    if 'cx' in gates:
        out['cx'] = gates['cx']
    else:
        out['cx'] = 0

    u_gates_cnt = 0
    other_gates_cnt = 0
    for gate_name in gates:
        if gate_name in ('u1', 'u2', 'u3'):
            u_gates_cnt += 1
        else:
            other_gates_cnt += 1
    out['u*'] = u_gates_cnt
    out['other'] = other_gates_cnt

    return out


# optimization_levels= [0]
# optimization_tries = 1

optimization_levels = [2, 3]
optimization_tries = 50

# withdrawal is not available in the UI on June 28, 2022, check at a later date
max_string_length = 3
basis_gates = ['u1', 'u2', 'u3', 'cx']
backend_architecture = FakeQasmSimulator()
cfg = backend_architecture.configuration()

stats = []
for string_length in range(1, max_string_length + 1):  # for each string length
    print(f"Working on strings of length {string_length}")
    all_strings = list(range(0, 2 ** string_length))
    # create all possible databases using iterator for powerset
    db_powerset = chain.from_iterable(combinations(all_strings, r) for r in range(len(all_strings) + 1))
    db_ind_cnt = 0
    for db in db_powerset:
        db_ind_cnt += 1
        if db_ind_cnt % 10 == 0:
            print(f"  Processed {db_ind_cnt} databases. Current db_size is {len(db)}")
        if not len(db) > 0:  # skip empty sets
            continue
        db = int_db_to_bin_str_db(db, string_length)
        stats_db = {}
        for storage_method in ('ep-pqm', 'jplf', 'avm'):
            if storage_method == 'ep-pqm':
                storage_saving = False
                storage_saving_method = None
            elif storage_method == 'jplf':
                storage_saving = True
                storage_saving_method = 'jplf'
            elif storage_method == 'avm':
                storage_saving = True
                storage_saving_method = 'avm'
            else:
                raise NotImplementedError(f"Storage method {storage_method} is unknown")

            x = StringComparator(db=db, target=db[0], is_binary=True, symbol_length=1, storage_profiling_on=True,
                                 storage_saving=storage_saving, storage_saving_method=storage_saving_method)

            # TODO play with optimization level
            best_circuit = None
            for optimization_level in optimization_levels:
                for optimization_try in range(optimization_tries):
                    optimized_circuit = transpile(x.circuit, coupling_map=cfg.coupling_map, basis_gates=basis_gates,
                                                  optimization_level=optimization_level)
                    if best_circuit is None:  # first run
                        best_circuit = optimized_circuit
                        best_circuit_stats = get_gates_cnt(optimized_circuit)
                    else:  # minimize gates count, priority: cx, u*, others
                        optimized_circuit_stats = get_gates_cnt(optimized_circuit)
                        is_better = False
                        if best_circuit_stats['cx'] > optimized_circuit_stats['cx']:
                            is_better = True
                        elif best_circuit_stats['cx'] == optimized_circuit_stats['cx'] and \
                                best_circuit_stats['u*'] > optimized_circuit_stats['u*']:
                            is_better = True
                        elif best_circuit_stats['cx'] == optimized_circuit_stats['cx'] and \
                                best_circuit_stats['u*'] == optimized_circuit_stats['u*'] and \
                                best_circuit_stats['other'] > optimized_circuit_stats['other']:
                            is_better = True

                        if is_better:
                            best_circuit = optimized_circuit
                            best_circuit_stats = get_gates_cnt(optimized_circuit)

            # for gate_name, gate_ctn in optimized_circuit.count_ops():
            stats_db[storage_method] = best_circuit.count_ops()
            # print(f"Gates for {storage_method}: {x}")
        stats.append({
            'string_length': string_length,
            'db_size': len(db),
            'gates': stats_db
        })

pd.json_normalize(stats).to_csv('stats_gates_storage.csv')
