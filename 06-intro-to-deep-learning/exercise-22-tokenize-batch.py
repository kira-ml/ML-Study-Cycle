"""
Exercise 22 — Data Preprocessing: Tokenization & Batching
Filename: exercise-22-tokenize-batch.py

Core Concept: Real-data preprocessing pipeline including tokenization, 
vocabulary creation, padding, masking, and efficient batching strategies.

This implementation demonstrates:
1. Simple tokenizer with whitespace splitting and subword heuristics
2. Vocabulary creation with special tokens and frequency-based filtering
3. Dynamic padding with attention masks
4. Length-based bucketing to minimize padding waste
5. Throughput benchmarking and optimization analysis

Key Insights:
- Tokenization converts raw text to numerical indices
- Padding enables batch processing of variable-length sequences
- Attention masks prevent processing of padding tokens
- Bucketing groups similar-length sequences to reduce computational waste
- Efficient batching significantly impacts training throughput
"""

import re
import time
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class Batch:
    """Container for a batch of tokenized sequences with masks."""
    input_ids: np.ndarray
    attention_mask: np.ndarray
    sequence_lengths: List[int]
    
    def __post_init__(self):
        self.batch_size = len(self.sequence_lengths)
        self.max_seq_len = self.input_ids.shape[1]
    
    def __repr__(self) -> str:
        return f"Batch(batch_size={self.batch_size}, max_seq_len={self.max_seq_len}, " \
               f"padding_ratio={self.padding_ratio():.3f})"
    
    def padding_ratio(self) -> float:
        """Calculate the ratio of padding tokens in the batch."""
        total_tokens = self.batch_size * self.max_seq_len
        actual_tokens = sum(self.sequence_lengths)
        return 1.0 - (actual_tokens / total_tokens)


class SimpleTokenizer:
    """
    Simple Tokenizer with Whitespace Splitting and Subword Heuristics.
    
    Implements a basic tokenization pipeline that demonstrates:
    - Whitespace-based tokenization
    - Simple subword splitting for unknown words
    - Vocabulary management with special tokens
    - Frequency-based vocabulary pruning
    
    Why tokenization matters:
    - Converts text to numerical representations models can process
    - Balances vocabulary size with sequence length
    - Handles out-of-vocabulary words via subword splitting
    - Special tokens enable model control ([PAD], [UNK], etc.)
    """
    
    def __init__(self, vocab_size: int = 2000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '[PAD]': 0,    # Padding token for batch processing
            '[UNK]': 1,    # Unknown words not in vocabulary
            '[BOS]': 2,    # Beginning of sequence
            '[EOS]': 3,    # End of sequence
        }
        self.vocab = {}
        self.inverse_vocab = {}
        
    def train(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Process:
        1. Tokenize all texts and count token frequencies
        2. Keep most frequent tokens up to vocab_size
        3. Add special tokens to vocabulary
        4. Build mapping between tokens and indices
        """
        print("Training tokenizer...")
        
        # Step 1: Tokenize and count frequencies
        token_counter = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            token_counter.update(tokens)
        
        # Step 2: Select most frequent tokens
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common = token_counter.most_common(available_slots)
        
        # Step 3: Build vocabulary starting with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add most frequent tokens
        for token, _ in most_common:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Step 4: Build inverse mapping
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
        print(f"Coverage: {self._calculate_coverage(token_counter, texts):.2%} of training tokens")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize a single text using whitespace splitting and simple subword heuristics.
        
        Process:
        1. Lowercase and basic cleaning
        2. Whitespace splitting
        3. Simple subword splitting for long words
        4. Basic punctuation handling
        """
        # Basic text cleaning
        text = text.lower().strip()
        
        # Simple punctuation handling - separate punctuation from words
        text = re.sub(r'([.!?,;:])', r' \1 ', text)
        
        # Whitespace tokenization
        tokens = text.split()
        
        # Apply subword splitting to long words
        processed_tokens = []
        for token in tokens:
            if len(token) > 10:  # Long word heuristic
                # Simple subword splitting: split every 3-4 characters
                subwords = [token[i:i+4] for i in range(0, len(token), 4)]
                processed_tokens.extend(subwords)
            else:
                processed_tokens.append(token)
        
        return processed_tokens
    
    def _calculate_coverage(self, token_counter: Counter, texts: List[str]) -> float:
        """Calculate what percentage of tokens are in vocabulary."""
        total_tokens = 0
        covered_tokens = 0
        
        for text in texts:
            tokens = self._tokenize_text(text)
            total_tokens += len(tokens)
            covered_tokens += sum(1 for token in tokens if token in self.vocab)
        
        return covered_tokens / total_tokens if total_tokens > 0 else 0.0
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token indices.
        
        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add [BOS] and [EOS] tokens
            
        Returns:
            List of token indices
        """
        tokens = self._tokenize_text(text)
        
        # Convert tokens to indices, using [UNK] for unknown tokens
        indices = []
        
        if add_special_tokens:
            indices.append(self.vocab['[BOS]'])
        
        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab['[UNK]']))
        
        if add_special_tokens:
            indices.append(self.vocab['[EOS]'])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        tokens = []
        for idx in indices:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                # Skip special tokens in output (except for demonstration)
                if token not in ['[PAD]', '[BOS]', '[EOS]']:
                    tokens.append(token)
        
        return ' '.join(tokens)


class DataBatcher:
    """
    Efficient Batching with Padding, Masking, and Length Bucketing.
    
    Implements strategies to optimize training throughput:
    - Dynamic padding: Pad sequences only to batch maximum, not global maximum
    - Attention masks: Prevent processing of padding tokens
    - Length bucketing: Group similar-length sequences to minimize padding
    - Batch size constraints: Control memory usage
    
    Why efficient batching matters:
    - Reduces computational waste from padding tokens
    - Improves GPU utilization and training speed
    - Enables processing of variable-length sequences
    - Maintains correct attention semantics
    """
    
    def __init__(self, max_seq_len: int = 256, batch_size: int = 32):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        
    def create_batches_naive(self, encoded_sequences: List[List[int]]) -> List[Batch]:
        """
        Create batches with naive sequential grouping.
        
        Simple approach: take sequences in order they appear
        Pros: Simple to implement
        Cons: High padding waste due to length variance
        """
        batches = []
        
        for i in range(0, len(encoded_sequences), self.batch_size):
            batch_sequences = encoded_sequences[i:i + self.batch_size]
            batch = self._create_batch(batch_sequences)
            batches.append(batch)
        
        return batches
    
    def create_batches_bucketed(self, encoded_sequences: List[List[int]], 
                              num_buckets: int = 10) -> List[Batch]:
        """
        Create batches with length-based bucketing.
        
        Process:
        1. Sort sequences by length
        2. Divide into buckets by length percentiles
        3. Create batches within each bucket
        4. Shuffle batches for training
        
        Why bucketing helps:
        - Groups similar-length sequences together
        - Drastically reduces padding within batches
        - Can improve throughput by 2-5x
        """
        # Step 1: Sort sequences by length
        sequences_with_lengths = [(seq, len(seq)) for seq in encoded_sequences]
        sequences_with_lengths.sort(key=lambda x: x[1])
        
        # Step 2: Create length buckets
        bucket_boundaries = self._create_bucket_boundaries(
            [length for _, length in sequences_with_lengths], 
            num_buckets
        )
        
        # Step 3: Assign sequences to buckets
        buckets = [[] for _ in range(num_buckets)]
        for seq, length in sequences_with_lengths:
            bucket_idx = self._find_bucket(length, bucket_boundaries)
            if bucket_idx < num_buckets:  # Skip sequences longer than max bucket
                buckets[bucket_idx].append(seq)
        
        # Step 4: Create batches within each bucket
        all_batches = []
        for bucket_sequences in buckets:
            # Shuffle sequences within bucket to maintain randomness
            random.shuffle(bucket_sequences)
            
            # Create batches from this bucket
            for i in range(0, len(bucket_sequences), self.batch_size):
                batch_sequences = bucket_sequences[i:i + self.batch_size]
                if batch_sequences:  # Skip empty batches
                    batch = self._create_batch(batch_sequences)
                    all_batches.append(batch)
        
        # Final shuffle of all batches
        random.shuffle(all_batches)
        return all_batches
    
    def _create_bucket_boundaries(self, lengths: List[int], num_buckets: int) -> List[int]:
        """Create bucket boundaries based on length percentiles."""
        if not lengths:
            return [self.max_seq_len]
        
        # Use percentiles to create balanced buckets
        percentiles = np.linspace(0, 100, num_buckets + 1)
        boundaries = np.percentile(lengths, percentiles).astype(int).tolist()
        
        # Ensure boundaries don't exceed max sequence length
        boundaries = [min(b, self.max_seq_len) for b in boundaries]
        
        return boundaries
    
    def _find_bucket(self, length: int, boundaries: List[int]) -> int:
        """Find which bucket a sequence of given length belongs to."""
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= length <= boundaries[i + 1]:
                return i
        return len(boundaries) - 1  # Default to last bucket
    
    def _create_batch(self, sequences: List[List[int]]) -> Batch:
        """
        Create a single batch with padding and attention masks.
        
        Process:
        1. Find maximum sequence length in this batch (up to max_seq_len)
        2. Pad all sequences to this length
        3. Create attention mask (1 for real tokens, 0 for padding)
        4. Track original sequence lengths
        """
        if not sequences:
            raise ValueError("Cannot create batch from empty sequence list")
        
        # Step 1: Find batch maximum length
        sequence_lengths = [min(len(seq), self.max_seq_len) for seq in sequences]
        batch_max_len = min(max(sequence_lengths), self.max_seq_len)
        
        # Step 2: Pad sequences and create attention masks
        batch_input_ids = []
        batch_attention_mask = []
        
        for seq, orig_length in zip(sequences, sequence_lengths):
            # Truncate if necessary
            truncated_seq = seq[:batch_max_len]
            current_length = len(truncated_seq)
            
            # Pad sequence
            padding_needed = batch_max_len - current_length
            padded_seq = truncated_seq + [self._get_pad_token()] * padding_needed
            batch_input_ids.append(padded_seq)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = [1] * current_length + [0] * padding_needed
            batch_attention_mask.append(attention_mask)
        
        return Batch(
            input_ids=np.array(batch_input_ids, dtype=np.int32),
            attention_mask=np.array(batch_attention_mask, dtype=np.bool_),
            sequence_lengths=sequence_lengths
        )
    
    def _get_pad_token(self) -> int:
        """Return padding token index. In practice, this would come from tokenizer."""
        return 0  # Assuming [PAD] token is at index 0


class ThroughputBenchmark:
    """Benchmark batching strategies and measure throughput."""
    
    def __init__(self, batcher: DataBatcher, tokenizer: SimpleTokenizer):
        self.batcher = batcher
        self.tokenizer = tokenizer
    
    def benchmark(self, texts: List[str], num_runs: int = 5) -> Dict[str, float]:
        """
        Benchmark different batching strategies.
        
        Measures:
        - Preprocessing throughput (examples/second)
        - Padding efficiency (percentage of useful tokens)
        - Batch creation speed
        """
        # Encode all texts first
        print("Encoding texts for benchmarking...")
        encoded_sequences = [self.tokenizer.encode(text) for text in texts]
        
        results = {}
        
        # Benchmark naive batching
        print("\nBenchmarking naive batching...")
        naive_metrics = self._benchmark_strategy(
            lambda: self.batcher.create_batches_naive(encoded_sequences),
            num_runs, "naive"
        )
        results.update(naive_metrics)
        
        # Benchmark bucketed batching
        print("Benchmarking bucketed batching...")
        bucketed_metrics = self._benchmark_strategy(
            lambda: self.batcher.create_batches_bucketed(encoded_sequences),
            num_runs, "bucketed"
        )
        results.update(bucketed_metrics)
        
        return results
    
    def _benchmark_strategy(self, batch_fn, num_runs: int, strategy_name: str) -> Dict[str, float]:
        """Benchmark a single batching strategy."""
        times = []
        padding_ratios = []
        batch_counts = []
        
        for run in range(num_runs):
            start_time = time.time()
            batches = batch_fn()
            end_time = time.time()
            
            times.append(end_time - start_time)
            padding_ratios.append(np.mean([batch.padding_ratio() for batch in batches]))
            batch_counts.append(len(batches))
        
        avg_time = np.mean(times)
        avg_padding = np.mean(padding_ratios)
        avg_batches = np.mean(batch_counts)
        
        throughput = len(batches) * self.batcher.batch_size / avg_time if avg_time > 0 else 0
        
        print(f"  {strategy_name}: {throughput:.1f} examples/sec, "
              f"padding: {avg_padding:.3f}, batches: {avg_batches:.1f}")
        
        return {
            f"{strategy_name}_throughput": throughput,
            f"{strategy_name}_padding_ratio": avg_padding,
            f"{strategy_name}_batch_count": avg_batches,
            f"{strategy_name}_time": avg_time
        }


def load_tiny_corpus() -> List[str]:
    """
    Load a small text corpus for demonstration.
    
    Using a mix of short sentences and longer passages to
    demonstrate length variability and bucketing benefits.
    """
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! This is a simple example.",
        "Machine learning transforms how we process data.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require substantial computational resources.",
        "Attention mechanisms have revolutionized sequence modeling.",
        "Transformers use self-attention to process sequences in parallel.",
        "Tokenization is the first step in text preprocessing.",
        "Batching improves training efficiency on GPU hardware.",
        "Padding enables processing of variable-length sequences.",
        "The cat sat on the mat.",
        "She read the book carefully.",
        "They went to the market yesterday.",
        "Programming requires logical thinking and problem-solving skills.",
        "Data preprocessing is crucial for machine learning pipelines.",
        "Neural networks learn hierarchical representations of data.",
        "Backpropagation enables gradient-based optimization of deep models.",
        "Regularization techniques prevent overfitting in machine learning.",
        "The weather today is quite pleasant and suitable for outdoor activities.",
        "Researchers continue to develop more efficient algorithms for training large models.",
    ]
    
    # Add some longer sequences to demonstrate bucketing benefits
    longer_texts = [
        "In the field of artificial intelligence, recent advances in large language models "
        "have demonstrated remarkable capabilities in understanding and generating human-like text. "
        "These models are typically trained on massive datasets containing billions of tokens.",
        
        "The process of training neural networks involves multiple iterations over the training data, "
        "with each iteration consisting of forward propagation, loss computation, backward propagation, "
        "and parameter updates using optimization algorithms like stochastic gradient descent.",
        
        "Effective data preprocessing pipelines include tokenization, vocabulary construction, "
        "sequence encoding, dynamic padding, attention mask creation, and efficient batching strategies "
        "that minimize computational waste while maintaining model performance and training stability."
    ]
    
    return corpus + longer_texts


def analyze_tokenization(tokenizer: SimpleTokenizer, texts: List[str]):
    """Analyze tokenization results and statistics."""
    print("\n" + "=" * 70)
    print("TOKENIZATION ANALYSIS")
    print("=" * 70)
    
    # Tokenize all texts and collect statistics
    sequence_lengths = []
    all_tokens = []
    
    for text in texts:
        tokens = tokenizer._tokenize_text(text)
        sequence_lengths.append(len(tokens))
        all_tokens.extend(tokens)
    
    # Print statistics
    print(f"Corpus size: {len(texts)} texts")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Unique tokens: {len(set(all_tokens))}")
    print(f"Sequence length - Min: {min(sequence_lengths)}, "
          f"Max: {max(sequence_lengths)}, Mean: {np.mean(sequence_lengths):.1f}")
    
    # Show tokenization examples
    print("\nTokenization examples:")
    for i, text in enumerate(texts[:3]):
        tokens = tokenizer._tokenize_text(text)
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Text {i+1}: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Encoded: {encoded}")
        print(f"  Decoded: {decoded}\n")


def plot_benchmark_results(results: Dict[str, float]):
    """Visualize benchmarking results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Throughput comparison
    strategies = ['naive', 'bucketed']
    throughputs = [results[f'{s}_throughput'] for s in strategies]
    
    ax1.bar(strategies, throughputs, color=['skyblue', 'lightcoral'])
    ax1.set_title('Batching Throughput (examples/sec)')
    ax1.set_ylabel('Examples per Second')
    
    # Padding ratio comparison
    padding_ratios = [results[f'{s}_padding_ratio'] for s in strategies]
    ax2.bar(strategies, padding_ratios, color=['skyblue', 'lightcoral'])
    ax2.set_title('Padding Ratio (lower is better)')
    ax2.set_ylabel('Padding Ratio')
    
    # Speedup calculation
    speedup = results['bucketed_throughput'] / results['naive_throughput']
    ax3.bar(['Speedup'], [speedup], color='lightgreen')
    ax3.set_title(f'Bucketed vs Naive Speedup: {speedup:.2f}x')
    ax3.set_ylabel('Speedup Factor')
    
    # Batch count comparison
    batch_counts = [results[f'{s}_batch_count'] for s in strategies]
    ax4.bar(strategies, batch_counts, color=['skyblue', 'lightcoral'])
    ax4.set_title('Number of Batches')
    ax4.set_ylabel('Batch Count')
    
    plt.tight_layout()
    plt.savefig('exercise-22-batching-benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main demonstration function."""
    print("Exercise 22: Data Preprocessing - Tokenization & Batching")
    print("Core Concept: Tokenization, padding, masking, and efficient batching")
    print("=" * 70)
    
    # Load corpus
    texts = load_tiny_corpus()
    print(f"Loaded corpus: {len(texts)} texts")
    
    # Initialize and train tokenizer
    tokenizer = SimpleTokenizer(vocab_size=500)
    tokenizer.train(texts)
    
    # Analyze tokenization
    analyze_tokenization(tokenizer, texts)
    
    # Initialize batcher
    batcher = DataBatcher(max_seq_len=32, batch_size=8)
    
    # Benchmark different strategies
    benchmark = ThroughputBenchmark(batcher, tokenizer)
    results = benchmark.benchmark(texts)
    
    # Display results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    speedup = results['bucketed_throughput'] / results['naive_throughput']
    padding_reduction = results['naive_padding_ratio'] - results['bucketed_padding_ratio']
    
    print(f"Throughput improvement: {speedup:.2f}x faster")
    print(f"Padding reduction: {padding_reduction:.3f} less padding")
    print(f"Naive → {results['naive_throughput']:.1f} examples/sec, "
          f"{results['naive_padding_ratio']:.3f} padding")
    print(f"Bucketed → {results['bucketed_throughput']:.1f} examples/sec, "
          f"{results['bucketed_padding_ratio']:.3f} padding")
    
    # Visualize results
    plot_benchmark_results(results)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("1. TOKENIZATION: Converts text to numerical indices with vocabulary")
    print("2. PADDING: Enables batch processing but creates computational waste") 
    print("3. ATTENTION MASKS: Prevent processing of padding tokens")
    print("4. BUCKETING: Groups similar-length sequences to minimize padding")
    print("5. THROUGHPUT: Efficient batching can significantly speed up training")
    print("6. TRADE-OFFS: Balance between implementation complexity and efficiency")


if __name__ == "__main__":
    main()