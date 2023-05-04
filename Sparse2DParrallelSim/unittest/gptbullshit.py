from joblib import Parallel, delayed
import numpy as np
import threading

def parallel_computation(x):
    # Define your computation here
    return x ** 2

def thread_function(results, i, sub_results):
    # Add the sub-results to the main results array
    results[i] = np.sum(sub_results)

if __name__ == '__main__':
    data = np.random.randn(1000)
    num_threads = 4
    sub_data = np.array_split(data, num_threads)

    # Initialize the results array
    results = np.zeros(num_threads)

    # Define the parallel computation function
    def parallel_computation_threaded(sub_data, i):
        sub_results = Parallel(n_jobs=-1, backend='multiprocessing')(
            delayed(parallel_computation)(x) for x in sub_data)
        thread_function(results, i, sub_results)

    # Create the threads and start them
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=parallel_computation_threaded, args=(sub_data[i], i))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Compute the final result
    final_result = np.sum(results)
    print(final_result)
