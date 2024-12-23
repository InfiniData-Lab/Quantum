import sys, json

from InfiniQuantumSim.TLtensor import QuantumCircuit


def query_from_circuit(json_circuit):
    if isinstance(json_circuit, str):
        with open(json_circuit, 'r') as file:
            json_circuit = json.load(file)
    elif isinstance(json_circuit, dict):
        ...
    else:
        raise ValueError("circuit should be provided either as dict object or str of the filename")
    
    qc = QuantumCircuit(circuit_dict=json_circuit)

    return qc.to_query()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = query_from_circuit(sys.argv[1])
    else:
        raise ValueError("Please provide filename to your json-circuit as argument to convert to SQL Query!")
