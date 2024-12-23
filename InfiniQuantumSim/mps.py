import numpy as np

import InfiniQuantumSim.sqlEinSum as ses

from InfiniQuantumSim.utils import INDICES

class MPS:
    def __init__(self, num_qubits, bond_dim=1, contr_method = 'np'):
        """
        Initialize an MPS for a given number of qubits.
        By default, the MPS represents the |0...0> state.
        
        Parameters:
        - num_qubits: Number of qubits in the system.
        - bond_dim: Initial bond dimension (default is 1).
        """
        self.num_qubits = num_qubits
        self.bond_dim = bond_dim
        self.tensors = []
        self.contr_method = contr_method
        self.con = None
        self.cur = None
        if "sql" in contr_method:
            self.con, self.cur = ses.connect_and_setup_db(contr_method)
        
        # Initialize tensors representing the |0...0> state
        for _ in range(num_qubits):
            # tensors in zero state have shape (1, 2, 1)
            tensor = np.zeros((1, 2, 1), dtype=complex)

            # Set the state to |0>
            tensor[0, 0, 0] = 1.0 + .0j
            self.tensors.append(tensor)

        match contr_method:
            case "np": self.contraction = self.np_contraction
            case _:
                raise ValueError(f"Contraction Method {contr_method} not implemented")
        

    def np_contraction(self, einsum, tensor1, tensor2):
        return np.einsum(einsum, tensor1, tensor2)

    def apply_single_qubit_gate(self, gate, qubit):
        """
        Apply a single-qubit gate to the MPS.
        
        Parameters:
        - gate: A 2x2 numpy array representing the gate.
        - qubit: The index of the qubit (0-based).
        """
        # Get the tensor at the specified qubit‚
        tensor = self.tensors[qubit]
        # Contract the gate with the physical index
        # Tensor shape: (left_dim, physical_dim, right_dim)
        # Gate shape: (physical_dim_out, physical_dim_in)
        # Resulting tensor shape: (left_dim, physical_dim_out, right_dim)
        left_dim, physical_dim, right_dim = tensor.shape


        # tensor = np.tensordot(tensor, gate, axes=([1], [0]))
        tensor = self.contraction('ijk,jl->ilk', tensor, gate)

        self.tensors[qubit] = tensor
        
    
    def apply_two_qubit_gate(self, gate_tensor, qubit1, qubit2, max_bond_dim=None):
        """
        Apply a two-qubit gate to the MPS.

        Parameters:
        - gate_tensor: A numpy array of shape (2,2,2,2) representing the two-qubit gate.
        - qubit1, qubit2: Indices of the qubits (0-based). qubit2 = qubit1 + 1
        - max_bond_dim: Maximum bond dimension after truncation.
        """
        if qubit2 != qubit1 + 1:
            raise ValueError("Only nearest-neighbor gates are supported.")

        # Get the tensors at qubit1 and qubit2
        tensor1 = self.tensors[qubit1]
        tensor2 = self.tensors[qubit2]

        # Merge the two tensors at qubit1 and qubit2
        # Tensor1 shape: (l1, p1, r1)
        # Tensor2 shape: (l2, p2, r2)
        l1, p1, r1 = tensor1.shape
        l2, p2, r2 = tensor2.shape
        assert r1 == l2, "Bond dimensions do not match."

        # Gate shape: (p1', p2', p1, p2)
        gate_tensor = gate_tensor.reshape(2, 2, 2, 2)
        
        # merged_tensor = np.tensordot(tensor1, tensor2, axes=([2], [0]))  # Contract over r1/l2
        # merged_tensor shape: (l1, p1, p2, r2)

        merged_tensor = self.contraction('ijk,klm->ijlm', tensor1, tensor2)
        merged_tensor = self.contraction('ijkl,hklm->hijm', gate_tensor, merged_tensor)

        # Reshape for SVD
        l1, p1p, p2p, r2 = merged_tensor.shape
        merged_tensor = merged_tensor.reshape(l1 * p1p, p2p * r2)
        # Perform SVD and truncate
        U, Vh, new_bond_dim = self._svd_truncate(merged_tensor, max_bond_dim)
        U = U.reshape(l1, p1p, new_bond_dim)
        Vh = Vh.reshape(new_bond_dim, p2p, r2)
        self.tensors[qubit1] = U
        self.tensors[qubit2] = Vh.transpose(0, 1, 2)

    def _svd_truncate(self, merged_tensor, max_bond_dim=None, epsilon=1e-10):
        # Perform SVD
        U, S, Vh = np.linalg.svd(merged_tensor, full_matrices=False)
        # Determine which singular values to keep based on threshold
        above_threshold = S > epsilon
        num_keep = np.count_nonzero(above_threshold)
        if max_bond_dim is not None:
            num_keep = min(num_keep, max_bond_dim)
        # Truncate U, S, Vh
        U = U[:, :num_keep]
        S = S[:num_keep]
        Vh = Vh[:num_keep, :]
        # Rescale U and Vh with the square root of singular values
        S_sqrt = np.sqrt(S)
        U = U * S_sqrt[np.newaxis, :]
        Vh = S_sqrt[:, np.newaxis] * Vh
        return U, Vh, num_keep
    
    def apply_swap_gate(self, qubit1, qubit2, max_bond_dim=None):
        """
        Apply a SWAP gate between two adjacent qubits.
        
        Parameters:
        - qubit1, qubit2: Indices of the qubits (0-based). Must be adjacent.
        - max_bond_dim: Maximum bond dimension after truncation.
        """
        if abs(qubit1 - qubit2) != 1:
            raise ValueError("SWAP gate can only be applied to adjacent qubits.")
        min_qubit = min(qubit1, qubit2)
        # SWAP gate tensor
        SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]).reshape(2,2,2,2)
        self.apply_two_qubit_gate(SWAP, min_qubit, min_qubit+1, max_bond_dim)
    
    def get_swap_path(self, qubit1, qubit2):
        """
        Generate a sequence of SWAP operations to bring qubit2 next to qubit1.
        
        Parameters:
        - qubit1, qubit2: Indices of the qubits (0-based).
        
        Returns:
        - path: List of tuples representing SWAP operations.
        """
        path = []
        if qubit1 < qubit2:
            # Move qubit2 towards qubit1
            for q in range(qubit2, qubit1, -1):
                path.append((q - 1, q))
        elif qubit1 > qubit2:
            # Move qubit2 towards qubit1
            for q in range(qubit2, qubit1):
                path.append((q, q + 1))
        return path
    
    def apply_gate(self, gate, max_bond_dim=None):
        """
        Apply a general gate to the MPS.
        
        Parameters:
        - gate: An instance of the Gate class.
        - max_bond_dim: Maximum bond dimension after truncation.
        """
        if len(gate.qubits) == 1:
            # Single-qubit gate
            qubit = gate.qubits[0]
            self.apply_single_qubit_gate(gate.tensor, qubit)
        elif len(gate.qubits) == 2:
            qubit1, qubit2 = gate.qubits
            if abs(qubit1 - qubit2) == 1:
                # Qubits are adjacent
                min_qubit = min(qubit1, qubit2)
                self.apply_two_qubit_gate(gate.tensor, min_qubit, min_qubit + 1, max_bond_dim)
            else:
                # Qubits are not adjacent; swap them together
                path = self.get_swap_path(qubit1, qubit2)
                # Bring qubits together
                for q1, q2 in path:
                    self.apply_swap_gate(q1, q2, max_bond_dim)
                # Apply the gate
                min_qubit = min(qubit1, qubit2)
                self.apply_two_qubit_gate(gate.tensor, min_qubit, min_qubit + 1, max_bond_dim)
                # Swap qubits back to original positions
                for q1, q2 in reversed(path):
                    self.apply_swap_gate(q1, q2, max_bond_dim)
        else:
            raise NotImplementedError("Gates with more than two qubits are not supported.")
    
    def get_state_vector(self):
        """
        Reconstruct the full state vector from the MPS.
        Warning: Exponential in the number of qubits.
        
        Returns:
        - state_vector: A numpy array of size 2**num_qubits containing the state amplitudes.
        """
        # create einsum for contracting all tensors in MPS
        einsum_notation = ""
        last_index = 0
        for tensor in self.tensors:
            n_i = len(tensor.shape)
            einsum_notation += INDICES[last_index:last_index+n_i] + ","
            last_index += n_i-1

        einsum_notation = einsum_notation[:-1] + "->" + einsum_notation[0] + einsum_notation[-2]

        #return oe.contract(einsum_notation, *self.tensors)
    
        tensor = self.tensors[0]
        for i in range(1, self.num_qubits):
            tensor = np.tensordot(tensor, self.tensors[i], axes=([2], [0]))
            # Move physical indices together
            tensor = np.transpose(tensor, (0, 2, 1, 3))
            # Merge bond dimensions
            left_dim = tensor.shape[0]
            right_dim = tensor.shape[3]
            tensor = tensor.reshape(left_dim, -1, right_dim)
        # The final tensor should have shape (1, 2**num_qubits, 1)
        state_vector = tensor.reshape(-1)
        return state_vector
    
    def __del__(self):
        if self.con is not None:
            self.con.close()