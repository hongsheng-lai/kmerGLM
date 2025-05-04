import hashlib
import random

#############################
# Utility Functions
#############################
def generate_random_sequence(length):
    """Generate a random DNA sequence (only A, C, G, T) of given length."""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(random.choice(bases) for _ in range(length))

def stable_hash(kmer):
    """
    Returns a stable integer hash for the k-mer using MD5.
    MD5 produces a reproducible hash for the same input.
    """
    return int(hashlib.md5(kmer.encode('utf-8')).hexdigest(), 16)

def canonical_kmer(kmer):
    """
    Return the canonical form of a k-mer, i.e. the lexicographically smaller string 
    between the k-mer and its reverse complement.
    """
    complement = str.maketrans("ACGT", "TGCA")
    rev_comp = kmer[::-1].translate(complement)
    return min(kmer, rev_comp)

#############################
# 1. Lexicographical Minimizers
#############################
def find_minimizers(sequence, k, w):
    """
    Compute k-mer minimizers for the given sequence using lexicographical order.
    Removes consecutive duplicates.
    
    Args:
        sequence (str): The DNA (or any character) sequence.
        k (int): Length of the k-mer.
        w (int): Window size over which to compute minimizers (must be >= k).

    Returns:
        list: List of minimizers (lexicographically smallest k-mer in each window),
              with consecutive duplicates removed.
    """
    if w < k:
        raise ValueError("Window size w must be greater than or equal to k")
    
    minimizers = []
    prev_min_kmer = None
    n = len(sequence)
    
    for i in range(n - w + 1):
        window = sequence[i:i + w]
        kmers = [window[j:j + k] for j in range(w - k + 1)]
        min_kmer = min(kmers)  # Lexicographical order
        if min_kmer != prev_min_kmer:
            minimizers.append(min_kmer)
            prev_min_kmer = min_kmer
            
    return minimizers

#############################
# 2. Hash-Based Minimizers
#############################
def find_minimizers_hash(sequence, k, w):
    """
    Compute k-mer minimizers for the sequence using a stable hash (MD5) based ordering.
    In each window, the k-mer with the smallest hash value is chosen.
    Consecutive duplicates are removed.
    
    Args:
        sequence (str): The DNA sequence.
        k (int): Length of the k-mer.
        w (int): Window size (>= k).
        
    Returns:
        list: List of minimizers based on hash scores.
    """
    if w < k:
        raise ValueError("Window size w must be greater than or equal to k")
    
    minimizers = []
    prev_min_kmer = None
    n = len(sequence)
    
    for i in range(n - w + 1):
        window = sequence[i:i + w]
        kmers = [window[j:j+k] for j in range(w - k + 1)]
        scores = [stable_hash(kmer) for kmer in kmers]
        min_index = scores.index(min(scores))
        min_kmer = kmers[min_index]
        if min_kmer != prev_min_kmer:
            minimizers.append(min_kmer)
            prev_min_kmer = min_kmer
            
    return minimizers

#############################
# 3. Custom-Scoring Minimizers
#############################
def default_scoring(kmer):
    """Default scoring using a stable MD5 hash."""
    return stable_hash(kmer)

def find_minimizers_custom(sequence, k, w, scoring_function=None, canonical=False):
    """
    Compute k-mer minimizers for the sequence using a customizable scoring function.
    
    Args:
        sequence (str): The DNA sequence.
        k (int): Length of the k-mer.
        w (int): Window size (>= k).
        scoring_function (function): Function that takes a k-mer and returns a numeric score.
                                     If None, uses default_scoring (stable MD5 hash).
        canonical (bool): If True, converts each k-mer to its canonical form before scoring.
        
    Returns:
        list: List of minimizers determined by the scoring function.
    """
    if w < k:
        raise ValueError("Window size w must be greater than or equal to k")
    if scoring_function is None:
        scoring_function = default_scoring
    
    minimizers = []
    prev_min_kmer = None
    n = len(sequence)
    
    for i in range(n - w + 1):
        window = sequence[i:i+w]
        kmers = [window[j:j+k] for j in range(w - k + 1)]
        if canonical:
            kmers = [canonical_kmer(kmer) for kmer in kmers]
        scores = [scoring_function(kmer) for kmer in kmers]
        min_index = scores.index(min(scores))
        min_kmer = kmers[min_index]
        if min_kmer != prev_min_kmer:
            minimizers.append(min_kmer)
            prev_min_kmer = min_kmer
            
    return minimizers

# Optional: A sample custom scoring function that penalizes GC content deviation from 50%
def gc_penalty_scoring(kmer):
    gc_fraction = (kmer.count('G') + kmer.count('C')) / len(kmer)
    # Lower penalty (score) for GC near 0.5, higher otherwise.
    # Multiplying by a large constant to make differences more distinct if needed.
    return abs(gc_fraction - 0.5) * 100

#############################
# 4. Syncmers
#############################
def find_syncmers(sequence, k, s, position=0):
    """
    Compute syncmers for a sequence.
    
    A k-mer is a syncmer if the smallest s-mer (by lexicographical order) within it 
    appears at the specified position.
    
    Args:
        sequence (str): The DNA sequence.
        k (int): Length of the k-mer.
        s (int): Length of the substring used for syncmer condition (s < k).
        position (int): The required position for the smallest s-mer.
        
    Returns:
        list: List of k-mers that satisfy the syncmer condition.
    """
    if s >= k:
        raise ValueError("Parameter s must be less than k")
    
    syncmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        s_mers = [kmer[j:j+s] for j in range(k - s + 1)]
        min_s_mer = min(s_mers)
        # Check if the s-mer at the given fixed position equals the minimum s-mer
        if kmer[position:position+s] == min_s_mer:
            syncmers.append(kmer)
    return syncmers

#############################
# Main Script - Example Usage
#############################
if __name__ == '__main__':
    # Generate a random DNA sequence
    seq = generate_random_sequence(1000)
    k = 6
    w = 20

    # Lexicographical minimizers
    lex_minimizers = find_minimizers(seq, k, w)
    print("Lexicographical Minimizers:")
    print("  Count:", len(lex_minimizers))
    # print(lex_minimizers)
    
    # Hash-based minimizers
    hash_minimizers = find_minimizers_hash(seq, k, w)
    print("\nHash-based Minimizers:")
    print("  Count:", len(hash_minimizers))
    # print(hash_minimizers)
    
    # Custom scoring minimizers (using default MD5 scoring)
    custom_minimizers = find_minimizers_custom(seq, k, w, canonical=True)
    print("\nCustom Minimizers (with canonical conversion):")
    print("  Count:", len(custom_minimizers))
    # print(custom_minimizers)
    
    # Custom scoring minimizers with GC penalty scoring
    gc_minimizers = find_minimizers_custom(seq, k, w, scoring_function=gc_penalty_scoring)
    print("\nCustom Minimizers (using GC penalty scoring):")
    print("  Count:", len(gc_minimizers))
    # print(gc_minimizers)
    
    # Syncmers
    # For syncmers we need s to be less than k; here, s is set to 3 and we choose syncmers where the smallest 3-mer appears at position 1.
    s = 3
    syncmers = find_syncmers(seq, k, s, position=1)
    print("\nSyncmers:")
    print("  Count:", len(syncmers))
    # print(syncmers)
