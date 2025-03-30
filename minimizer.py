def find_minimizers(sequence, k, w):
    """
    Compute k-mer minimizers for the given sequence.
    
    Args:
        sequence (str): The DNA (or any character) sequence.
        k (int): Length of the k-mer.
        w (int): Window size over which to compute minimizers.
                 Must be >= k.
    
    Returns:
        list of tuples: Each tuple contains (window_start_index, minimizer).
                        The minimizer is the lexicographically smallest k-mer in the window.
    """
    if w < k:
        raise ValueError("Window size w must be greater than or equal to k")

    minimizers = []
    n = len(sequence)
    
    # Loop over every window starting index where a full window can be taken
    for i in range(n - w + 1):
        window = sequence[i:i+w]
        # Extract all k-mers within the current window
        kmers = [window[j:j+k] for j in range(w - k + 1)]
        # Choose the lexicographically smallest k-mer as the minimizer
        min_kmer = min(kmers)
        minimizers.append((i, min_kmer))
    
    return minimizers


# Example usage
if __name__ == '__main__':
    seq = "ATCGGTACTGCTATACTAGCTAGCTAGATCGGTACTGCTATACTA"
    k = 6  # k-mer length
    w = 8  # window size
    result = find_minimizers(seq, k, w)
    
    print("Window Start\tMinimizer")
    for index, minimizer in result:
        print(f"{index}\t\t{minimizer}")
