# Classical Simulation of Quantum Circuits using RDBMS

This repository contains the works associated with the Paper ["Quantum Data Managemen in the NISQ Era"](https://arxiv.org/pdf/2409.14111).

In the [technical report](technical_report.pdf), we provide detailed experimental settings and comprehensive results in Appendices B3 and B4.


# Reproducing the shown results

The results shown in the paper can either be reproduced using the [benchmark notebook](benchmark.ipynb) or by running the [simulation script](simulation.py).
Required packages for the Python installation can be installed using the command 

`pip install -r requirements.txt`

Here is a list of the included dependencies

```
duckdb==1.1.3
jupyter_client==8.6.3
jupyter_core==5.7.2
numpy==2.2.1
opt-einsum==3.3.0
psycopg2-binary==2.9.10
```

To generate an SQL query from a quantum circuit provided as json you can use the [circuit_sqlgen file](circuit_sqlgen.py). A circuit should be of the following format:

```
{
  "number_of_qubits": 3,
  "gates": [
    {
        "qubits": [0],
        "gate": "H"
    },
    {
        "qubits": [0, 1],
        "gate": "CNOT"
    },
    {
        "qubits": [1, 2],
        "gate": "CNOT"
    }
  ]
}
```

This json quantum circuit would create the GHZ state for three qubits.

When running the simulation.py, make sure there is PostgreSQL running on your machine with the following settings:

Using PostgreSQL is optional and if not needed should be explicitly excluded. PostgreSQL should have the following configuaration:

- POSTGRES_DB: 'postgres'
- POSTGRES_USER: 'postgres'
- POSTGRES_PASSWORD: 'password'
- HostPort: "5432"

If you want to run a customized version of the simulations please refer to the API Documentation below.

To quickly do a test run without PostgreSQL and DuckDB run simulation.py with the argument `sqlite` for an extensive test or `test` for a very quick test run.

# Simulation API
1. [Functions](#functions)
    - [sql_to_np](#sql_to_np)
    - [generate_ghz_circuit](#generate_ghz_circuit)
    - [generate_w_circuit](#generate_w_circuit)
    - [generate_qft_circuit](#generate_qft_circuit)
    - [generate_qpe_circuit](#generate_qpe_circuit)
    - [generate_ghz_qft](#generate_ghz_qft)
    - [generate_w_qft](#generate_w_qft)

2. [Classes](#classes)
    - [QuantumCircuit](#quantumcircuit)
    - [Gate](#gate)


## Functions

### sql_to_np
#### Parameters:
- `db_result` (list): Database query result rows.
- `expected_shape` (tuple): Shape of the resulting tensor.
- `complex_flag` (bool, optional): If `True`, assumes complex values in the result.

#### Returns:
- `np.ndarray`: The constructed NumPy tensor.

### generate_ghz_circuit
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.
- `reverse` (bool, optional): If `True`, the circuit will be reversed

#### Returns:
- `dict`: The constructed circuit as dict.

### generate_w_circuit
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.
- `reverse` (bool, optional): If `True`, the circuit will be reversed

#### Returns:
- `dict`: The constructed circuit as dict.

### generate_qft_circuit
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.
- `reverse` (bool, optional): If `True`, the circuit will be reversed

#### Returns:
- `dict`: The constructed circuit as dict.

### generate_qpe_circuit
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.

#### Returns:
- `dict`: The constructed circuit as dict.

### generate_ghz_qft
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.

#### Returns:
- `dict`: The constructed circuit as dict.

### generate_w_qft
#### Parameters:
- `num_qubits` (int): Number of Qubits in the GHZ state preparation.

#### Returns:
- `dict`: The constructed circuit as dict.


## Classes

### QuantumCircuit
Class for constructing and simulating quantum circuits.

#### Constructor:
```python
QuantumCircuit(num_qubits: int = None, circuit_dict: dict = None)
```

#### Attributes:
- `num_qubits` (int): Number of Qubits in the circuit
- `gates` (list): List of gates in the circuit
- `tensor_uniques` (dict): All unique tensors of the initial state and the gates in the circuit
- `einsum`  (str): Einstein Summation Notation as str
- `mps` (MPS | None): If simulation method "MPS" is used this contains the corresponding MPS Object
- `con` (Connection | None): DB Connection if exists
- `cur` (Cursor | None): DB Cursor if exists
- `dispatcher` (dict): Gate dispatcher to map str gate names to corresponding gate method

#### Methods:

- `to_query(complex=True)`: Converts the circuit to a SQL query.
- `convert_to_einsum()`: Converts the circuit to Einstein summation notation.
- `run(contr_method='np')`: Runs the circuit using the specified contraction method.
- `benchmark_circuit_performance(n_runs)`: Benchmarks the circuit's performance.
- `export_circuit_query()`: Exports the circuit query.
- `One-Qubit_Gate(qubit)`: possible gates: H, X, Y, Z
- `Controlled-Gate(control_qubit, target_qubit)`: possible gates: CNOT, CY, CZ
- `R(qubit, k)`: applies Phase Shift gate with parameter k
- `RY(qubit, theta)`: applies Rotation around Y-axis of Bloch-Sphere with angle theta
- `G(qubit, p)`: G(p) gate as described in [Cruz et al.](https://arxiv.org/pdf/1807.05572)


### Gate
Base class for quantum gates.

#### Constructor:
```python
Gate(qubits: list, tensor: numpy.ndarray, name: str = None, two_qubit_gate: bool = False)
```

#### Attributes:
- `qubits` (list): Qubits the gate acts on.
- `tensor` (np.ndarray): Matrix representation of the gate.
- `two_qubit_gate` (bool): Indicates if the gate is a two-qubit gate.
- `gate_name` (str): Name of the gate.
