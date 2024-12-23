# Classical Simulation of Quantum Circuits using RDBMS

This repository contains the works associated with the Paper ["Quantum Data Managemen in the NISQ Era"](https://arxiv.org/pdf/2409.14111).


# Reproducing the shown results

The results shown in the paper can either be reproduced using the [benchmark notebook](benchmark.ipynb) or by running the [simulation script](simulation.py).
To generate an SQL query from a quantum circuit provided as json you can use the [circuit_sqlgen file](circuit_sqlgen.py). A circuit should be of the following format:

```
{
  "n_qubits": 3,
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

# API Usage