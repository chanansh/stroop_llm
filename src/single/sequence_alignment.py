import numpy as np
from typing import List, Tuple, Dict
from Levenshtein import distance as levenshtein_distance, ratio, opcodes

def align_sequences(sequence1: List[str], sequence2: List[str]) -> Tuple[List[str], List[str]]:
    """
    Align two sequences of words using Levenshtein's opcodes for word-level alignment.
    
    Args:
        sequence1: First sequence of words
        sequence2: Second sequence of words
        
    Returns:
        Tuple of aligned sequences
    """
    # Get word-level edit operations
    ops = opcodes(sequence1, sequence2)
    
    # Initialize aligned sequences
    aligned_seq1 = []
    aligned_seq2 = []
    
    # Apply operations to create aligned sequences
    for op, i1, i2, j1, j2 in ops:
        if op == 'equal':
            # Words match, add them to both sequences
            aligned_seq1.extend(sequence1[i1:i2])
            aligned_seq2.extend(sequence2[j1:j2])
        elif op == 'replace':
            # Words differ, add both with their original content
            aligned_seq1.extend(sequence1[i1:i2])
            aligned_seq2.extend(sequence2[j1:j2])
        elif op == 'delete':
            # Words deleted from seq1, add gaps to seq2
            aligned_seq1.extend(sequence1[i1:i2])
            aligned_seq2.extend(['-'] * (i2 - i1))
        elif op == 'insert':
            # Words inserted in seq2, add gaps to seq1
            aligned_seq1.extend(['-'] * (j2 - j1))
            aligned_seq2.extend(sequence2[j1:j2])
    
    return aligned_seq1, aligned_seq2

def calculate_edit_metrics(aligned_seq1: List[str], aligned_seq2: List[str]) -> Dict[str, float]:
    """
    Calculate various edit distance metrics between aligned sequences.
    
    Args:
        aligned_seq1: First aligned sequence
        aligned_seq2: Second aligned sequence
        
    Returns:
        Dictionary containing various edit distance metrics
    """
    total_length = len(aligned_seq1)
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a.lower() == b.lower())
    insertions = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if b == "-")
    deletions = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == "-")
    substitutions = total_length - matches - insertions - deletions
    
    # Calculate additional similarity metrics
    similarity_scores = []
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a != "-" and b != "-":
            similarity_scores.append(ratio(a.lower(), b.lower()))
    
    metrics = {
        "match_rate": matches / total_length,
        "insertion_rate": insertions / total_length,
        "deletion_rate": deletions / total_length,
        "substitution_rate": substitutions / total_length,
        "total_edit_distance": insertions + deletions + substitutions,
        "mean_similarity": np.mean(similarity_scores) if similarity_scores else 0,
        "std_similarity": np.std(similarity_scores) if similarity_scores else 0
    }
    
    return metrics

def analyze_response_alignment(ai_responses: List[str], correct_responses: List[str]) -> Dict:
    """
    Analyze the alignment between AI responses and correct responses.
    
    Args:
        ai_responses: List of AI responses
        correct_responses: List of correct responses
        
    Returns:
        Dictionary containing alignment analysis results
    """
    # Align sequences
    aligned_ai, aligned_correct = align_sequences(ai_responses, correct_responses)
    
    # Calculate edit metrics
    metrics = calculate_edit_metrics(aligned_ai, aligned_correct)
    
    # Add aligned sequences to results
    metrics["aligned_ai"] = aligned_ai
    metrics["aligned_correct"] = aligned_correct
    
    return metrics 