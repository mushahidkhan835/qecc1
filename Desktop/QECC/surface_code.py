# -*- coding: utf-8 -*-

"""Generates circuits for quantum error correction with surface code patches."""


import networkx as nx

from catalyst import qjit, measure
import pennylane as qml
from jax import numpy as jnp

class SurfaceCode:
    """
    Implementation of a distance d surface code, implemented over
    T syndrome measurement rounds.
    """

    def __init__(self, d, T):
        """
        Creates the circuits corresponding to a logical 0 encoded
        using a surface code with X and Z stabilizers.
        Args:
            d (int): Number of physical "data" qubits. Only odd d's allowed
            T (int): Number of rounds of ancilla-assisted syndrome measurement. Normally T=d
        Additional information:
            No measurements are added to the circuit if `T=0`. Otherwise
            `T` rounds are added, followed by measurement of the code
            qubits (corresponding to a logical measurement and final
            syndrome measurement round)
            This circuit is for "rotated lattices" i.e. it requires
            d**2 data qubits and d**2-1 syndrome qubits only. Hence,
            d=odd allows equal number of Z and X stabilizer mesaurments.
        """
        self.d = d
        self.T = 0
        self.data =  list(range(d ** 2 - 1, d ** 2 - 1 + d ** 2))
        self.ancilla =  list(range(d ** 2 - 1))
        self.output = []



    def get_circuit_list(self):
        """
        Returns:
            circuit_list: self.circuit as a list, with
            circuit_list[0] = circuit['0']
            circuit_list[1] = circuit['1']
        """
        circuit_list = [self.circuit[log] for log in ["0", "1"]]
        return circuit_list

    """It assigns vertices to qubits on a 2D graph, where,
    data qubits are on the x lines and, syndrome qubits are
    on the 0.5+x lines, in the cartesian coordinate system, where x is an integer."""

    def get_data_string_arr(self, data_ind):
        data_string_arr = []
        i = 0
        while i < len(data_ind):
            j = i
            temp = []
            while j < len(data_ind):
                if data_ind[j][0] == data_ind[i][0]:
                    temp.append(data_ind[j])
                    j += 1
                else:
                    break
            data_string_arr.append(temp)
            i = j
            
        return data_string_arr

    def arrange_lattice_info_correct_order(self, top, bottom, left, right, middle, distance):
        final = top
        i = 1
        addRight = True
        j = 0
        while j < len(middle):
            if i == distance and addRight and right:
                final.append(right.pop(0))
                addRight = False
                i = 1
            elif i == 1 and not addRight and left:
                final.append(left.pop(0))
                addRight = True
                
            else:    
                final.append(middle[j])
                j += 1
                i += 1
            
        for b in bottom:
            final.append(b)
            
        return final

    def lattice(self):
        top = []
        right = []
        bottom = []
        left = []
        middle = []
        
        data_string_arr = []
        i = 0
        
        d = self.d
        data_string = nx.Graph()
        syndrome_string = nx.Graph()
        
        for i in range(0, d):
            for j in range(0, d):
                data_string.add_node((i, j))
        data_ind = list(data_string.nodes)
        data_string_arr = self.get_data_string_arr(data_ind)
        
        for i in range(len(data_string_arr)):
            for j in range(len(data_string_arr[0])):
                if j == len(data_string_arr[0]) - 1:
                    if i < len(data_string_arr) - 1 and i % 2 == 0:
                        right.append([data_string_arr[i][j][0] + 0.5, data_string_arr[i][j][1] + 0.5])
                        
                if i == 0:
                    if j < len(data_string_arr[0]) - 1 and j % 2 == 0:
                        top.append([data_string_arr[i][j][0]- 0.5, data_string_arr[i][j][1] + 0.5])
                                    
                if i == len(data_string_arr) - 1:
                    if j < len(data_string_arr[0]) - 1 and j % 2 != 0:
                        bottom.append([data_string_arr[i][j][0] + 0.5, data_string_arr[i][j][1] + 0.5])
                if j == 0:
                    if i < len(data_string_arr) - 1  and i % 2 != 0:
                        left.append([data_string_arr[i][j][0] + 0.5, data_string_arr[i][j][1] - 0.5])
                        
        for i in range(1, len(data_string_arr)):
            for j in range(len(data_string_arr[0]) - 1):
                middle.append([data_string_arr[i][j][0] - 0.5, data_string_arr[i][j + 1][1] - 0.5])

        lattice_arr = self.arrange_lattice_info_correct_order(top, bottom, left, right, middle, d)
        
        for l in lattice_arr:
            syndrome_string.add_node(tuple(l))
            
        return (list(syndrome_string), data_ind)


    def connection(self):
        syn_index, data_index = self.lattice()
        order_list = []
        for i in range(self.d ** 2 - 1):
            r = syn_index[i][0]
            c = syn_index[i][1]

            order = []
            order.append((r, c))
            if r == -0.5:  # top semicircile
                order.append([-1, data_index.index((r + 0.5, c - 0.5)), -1, data_index.index((r + 0.5, c + 0.5))])

            elif c == -0.5:  # left semicircle
                order.append([-1, data_index.index((r - 0.5, c + 0.5)), -1, data_index.index((r + 0.5, c + 0.5))])
            elif r == self.d - 0.5:  # bottom semicircle
                order.append([data_index.index((r - 0.5, c - 0.5)), -1, data_index.index((r - 0.5, c + 0.5)), -1])

            elif c == self.d - 0.5:  # right semicircle
                order.append([data_index.index((r - 0.5, c - 0.5)), -1, data_index.index((r + 0.5, c - 0.5)), -1])
            else:
                if (r + c) % 2 == 0:  # square patches
                    order.append([data_index.index((r - 0.5, c - 0.5)), 
                               data_index.index((r + 0.5, c - 0.5)),
                               data_index.index((r - 0.5, c + 0.5)),
                               data_index.index((r + 0.5, c + 0.5))])
                else:
                    order.append([data_index.index((r - 0.5, c - 0.5)), 
                        data_index.index((r - 0.5, c + 0.5)),
                        data_index.index((r + 0.5, c - 0.5)),
                        data_index.index((r + 0.5, c + 0.5))])
                    
            order_list.append(order)
        return order_list
        
    
    def syndrome_measurement(self):
            order = self.connection()
            results_x = []
            results_y = []
            @qjit
            @qml.qnode(qml.device("lightning.qubit", wires=len(self.data)+len(self.ancilla)))
            def circuit():
                qml.PauliX(13)
                for _ in range(self.T - 1):
                    for i in range(len(order)):
                        for j in range(0, 4):
                            if (order[i][0][0] + order[i][0][1]) % 2 == 0:  # Xstabilizer
                                if j == 1:
                                    qml.Hadamard(self.ancilla[i])       
                                if order[i][1][j] != -1:
                                    qml.CNOT([self.ancilla[i], self.data[order[i][1][j]]])

                                if j == 4:
                                    qml.Hadamard(self.ancilla[i])    
                            else:
                                if order[i][1][j] != -1:
                                    qml.CNOT([self.data[order[i][1][j]], self.ancilla[i]])
                                   
                    qml.Barrier(len(self.data)+len(self.ancilla))
                    for q in self.ancilla:
                        if (order[i][0][0] + order[i][0][1]) % 2 == 0:  # Xstabilizer
                            results_x.append(measure(q))
                    qml.Barrier(len(self.data)+len(self.ancilla))
                    
                    for q in self.ancilla:
                        if (order[i][0][0] + order[i][0][1]) % 2 != 0:
                            results_y.append(measure(q))
                return None
            
            circuit()
            return results_x,  results_y


    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        for log in ["0", "1"]:
            self.circuit[log].add_register(self.c_output)
            for i in range(self.d ** 2):
                self.circuit[log].measure(self.data[i], self.c_output[i])

    def process_results(self, raw_results):
        """
        Args:
            raw_results (dict): A dictionary whose keys are logical values,
                and whose values are standard counts dictionaries, (as
                obtained from the `get_counts` method of a ``qiskit.Result``
                object).
        Returns:
            syn: d+1 dimensional array where 0th array stores qubit readouts
            while the subsequesnt rows store the results from measurement rounds
            as required for extraction of nodes with errors to be sent to the decoder
        Additional information:
            The circuits must be executed outside of this class, so that
            their is full freedom to compile, choose a backend, use a
            noise model, etc. The results from these executions should then
            be used to create the input for this method.
        """
        results = []
        results = list(max(raw_results, key=raw_results.get))

        syn = []
        new = []
        for i in results:
            for j in range(len(i)):
                if i[j] != " ":
                    new.append(int(i[j]))
                else:
                    syn.append(new)
                    new = []
        syn.append(new)

        return syn

    def extract_nodes(self, syn_meas_results):
        """Extracts node locations of qubits which flipped in
        consecutive rounds (stored as (k,i,j)) and the data qubits which were flipped
        during readout (stored as (-2,i,j)). Here k spans range(0,d-1,1)
        Z syndrome nodes and Z logical data qubit nodes (see figure) in error_nodesZ
        and we do the same for X stabilizers and X logical qubits in error_nodesX.
        Note that arrays are reversed in terms of syndrome rounds, when compared to
        syn_meas_results
        """
        processed_results = []
        new = []
        for j in syn_meas_results[0]:
            new.append(j)
        processed_results.append(new)
        new = []
        for j in syn_meas_results[len(syn_meas_results) - 1]:
            new.append(j)
        processed_results.append(new)

        for i in range(len(syn_meas_results) - 2, 0, -1):
            new = []
            for j in range(0, len(syn_meas_results[i])):
                new.append((syn_meas_results[i][j] + syn_meas_results[i + 1][j]) % 2)
            processed_results.append(new)

        syn, dat = self.lattice()
        error_nodesX = []
        error_nodesZ = []


        for i in range(1, len(processed_results)):
            for j in range(len(processed_results[i])):

                if processed_results[i][j] == 1:

                    if (syn[j][0] + syn[j][1]) % 2 == 0:
                        error_nodesX.append((i - 1, syn[j][0], syn[j][1]))
                    else:
                        error_nodesZ.append((i - 1, syn[j][0], syn[j][1]))
        return error_nodesX, error_nodesZ