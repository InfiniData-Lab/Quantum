import sys, json, datetime

import InfiniQuantumSim.TLtensor as tlt

def init_simulation_params():
    circuit_n_qubits = {
        'W': range(5,35),
        'H': range(5,36),
        'QFT': range(5,25),
        'GHZ': range(5,35),
        'QPE': range(5,19)
    } 

    circuits = {
        'GHZ': tlt.generate_ghz_circuit, 
        'W': tlt.generate_w_circuit,
        'QFT': tlt.generate_qft_circuit,
        'QPE': tlt.generate_qpe_circuit,
    }

    return circuit_n_qubits, circuits


def save_results_to_json(results, fname = None):
    if fname is None:
        fname = datetime.datetime.now().strftime("results/%m%d%y_MBP_QFTonly") + ".json"

    with open(fname, "w") as file:
        json.dump(results, file)


def simulation_benchmark(n_runs, circuit_n_qubits, circuits, mem_limit_bytes=2**34, time_limit_seconds=2**4):
    results = {}
    for cname, circuit in circuits.items():
        oom = ["psql", "ducksql"]
        results[cname] = {}
        progress = 0
        l_qbits = len(circuit_n_qubits[cname])-1
        for n_qubits in circuit_n_qubits[cname]:
            sys.stdout.write('\r')
            circuit_dict = circuit(n_qubits)
            qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)

            results[cname][n_qubits] = qc.benchmark_ciruit_performance(n_runs, oom = oom)
            for method in results[cname][n_qubits].keys():
                if method in oom or method == "eqc":
                        continue
                mem_avg = sum(results[cname][n_qubits][method]["memory"])/n_runs
                tim_avg = sum(results[cname][n_qubits][method]["time"])/n_runs
                if "sql" in method:
                        mem_avg += sum(results[cname][n_qubits]["eqc"]["tensor"]["memory"])/n_runs
                        tim_avg += sum(results[cname][n_qubits]["eqc"]["tensor"]["time"])/n_runs
                        mem_avg += sum(results[cname][n_qubits]["eqc"]["contraction"]["memory"])/n_runs
                        tim_avg += sum(results[cname][n_qubits]["eqc"]["contraction"]["time"])/n_runs

                if  mem_avg >= mem_limit_bytes:
                        print(f'{method} kicked out due to memory limitations!\n')
                        oom.append(method)
                elif tim_avg >= time_limit_seconds:
                        print(f'{method} kicked out due to time limitations!\n')
                        oom.append(method)

            sys.stdout.write("[%-20s] %d%%" % ('='*int((20/l_qbits)*progress), progress*(100/l_qbits)))
            sys.stdout.flush()
            progress += 1

        print("\n" + cname + " done!")
    
    return results



if __name__ == "__main__":
    
    n_runs = 20
    
    circuit_n_qubits, circuits = init_simulation_params()

    results = simulation_benchmark(n_runs, circuit_n_qubits, circuits)

    save_results_to_json(results)