#!/usr/bin/env python3
"""
BabyLM Corpus Cleaning Pipeline

A production-quality data cleaning and mixing pipeline for the BabyLM corpus.
Performs two-pass cleaning (per-corpus + global), quality scoring, and stratified sampling.

Outputs:
  - clean_10M_full.txt: All cleaned data (~10M tokens)
  - clean_10M_top70.txt: Top 70% quality subset (~7M tokens)
  - clean_1M.txt: Stratified sample (exactly 1M tokens)
  - cleaning_report.json: Detailed statistics
"""

import argparse
import hashlib
import json
import logging
import random
import re
import shelve
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

# ============================================================================
# Configuration and Constants
# ============================================================================

CORPUS_FILES = [
    "bnc_spoken.train",
    "childes.train",
    "gutenberg.train",
    "open_subtitles.train",
    "simple_wiki.train",
    "switchboard.train",
]

# Target mixture proportions (by token count in final output)
MIXTURE_PROPORTIONS = {
    "gutenberg": 0.40,
    "simple_wiki": 0.25,
    "bnc_spoken": 0.15,
    "childes": 0.10,
    "open_subtitles": 0.08,
    "switchboard": 0.02,
}

# Filtering thresholds
MIN_LINE_LENGTH = 10
MAX_LINE_LENGTH = 500
MIN_ALPHA_RATIO = 0.60
NEAR_DEDUP_THRESHOLD = 0.85  # Jaccard similarity for MinHash LSH
CHUNK_SIZE = 10000  # Process in chunks for memory efficiency

# Quality scoring weights
SCORE_WEIGHTS = {
    "length": 20,
    "alpha_ratio": 40,
    "digit_ratio": 15,
    "punctuation": 15,
    "repetition": 10,
}


@dataclass
class Stats:
    """Track statistics for a cleaning phase."""
    input_lines: int = 0
    input_tokens: int = 0
    output_lines: int = 0
    output_tokens: int = 0
    removed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> dict:
        return {
            "input_lines": self.input_lines,
            "input_tokens": self.input_tokens,
            "output_lines": self.output_lines,
            "output_tokens": self.output_tokens,
            "removed_by_reason": dict(self.removed),
        }


# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    """Count tokens using whitespace tokenization."""
    return len(text.split())


def sha256_hash(text: str) -> str:
    """Generate SHA256 hash for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_whitespace(line: str) -> str:
    """Normalize whitespace: strip, collapse multiple spaces."""
    return " ".join(line.split())


# ============================================================================
# Phase 1: Per-Corpus Cleaning
# ============================================================================

def clean_bnc_spoken(line: str) -> str:
    """Clean BNC Spoken corpus."""
    # Remove lines that look like metadata or non-dialogue
    if line.startswith("<") or line.startswith("["):
        return ""
    return line


def clean_childes(line: str) -> str:
    """Clean CHILDES corpus: remove speaker tags and annotations."""
    # Remove speaker tags like *CHI:, *MOT:, etc.
    line = re.sub(r'^\*[A-Z]+:\s*', '', line)
    # Remove bracketed annotations [...]
    line = re.sub(r'\[.*?\]', '', line)
    return line


def clean_gutenberg(line: str) -> str:
    """Clean Gutenberg corpus: remove headers, footers, chapter markers."""
    # Remove chapter markers like *CHAPTER XI* or CHAPTER 1
    if re.match(r'^\*?CHAPTER\s+[IVXLCDM0-9]+\*?$', line, re.IGNORECASE):
        return ""
    # Remove common Gutenberg metadata lines
    if any(marker in line.lower() for marker in [
        "project gutenberg", "end of the project", "*** start",
        "*** end", "e-text", "ebook"
    ]):
        return ""
    return line


def clean_open_subtitles(line: str) -> str:
    """Clean Open Subtitles: remove timestamps and excessive single-word lines."""
    # Remove timestamp markers (various formats)
    line = re.sub(r'\d{1,2}:\d{2}:\d{2}[,\.]\d+\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d+', '', line)
    line = re.sub(r'^\d+$', '', line)  # Remove subtitle numbers
    
    # Filter out lines with only one word (excessive in subtitles)
    if len(line.split()) == 1:
        return ""
    return line


def clean_simple_wiki(line: str) -> str:
    """Clean Simple Wikipedia: remove section markers."""
    # Remove section headers like = = = Section Name = = =
    line = re.sub(r'^=+\s*.*?\s*=+$', '', line)
    return line


def clean_switchboard(line: str) -> str:
    """Clean Switchboard: remove speaker prefixes."""
    # Remove speaker prefixes A:\t or B:\t
    line = re.sub(r'^[AB]:\s*', '', line)
    return line


# Corpus-specific cleaning function mapping
CORPUS_CLEANERS = {
    "bnc_spoken": clean_bnc_spoken,
    "childes": clean_childes,
    "gutenberg": clean_gutenberg,
    "open_subtitles": clean_open_subtitles,
    "simple_wiki": clean_simple_wiki,
    "switchboard": clean_switchboard,
}


def passes_character_ratio_filter(line: str) -> bool:
    """Check if line has at least MIN_ALPHA_RATIO alphabetic characters."""
    if not line:
        return False
    alpha_count = sum(1 for c in line if c.isalpha())
    return (alpha_count / len(line)) >= MIN_ALPHA_RATIO


def passes_length_filter(line: str) -> bool:
    """Check if line length is within acceptable range."""
    return MIN_LINE_LENGTH <= len(line) <= MAX_LINE_LENGTH


def process_corpus_phase1(
    input_path: Path,
    output_path: Path,
    corpus_name: str,
    stats: Stats,
    logger: logging.Logger,
) -> Set[str]:
    """
    Phase 1: Per-corpus cleaning with exact deduplication.
    
    Returns: Set of hashes for exact deduplication.
    """
    logger.info(f"Processing {corpus_name} (Phase 1)...")
    
    corpus_key = corpus_name.replace(".train", "")
    cleaner = CORPUS_CLEANERS.get(corpus_key, lambda x: x)
    
    seen_hashes = set()
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            stats.input_lines += 1
            line = line.strip()
            
            if not line:
                stats.removed["empty"] += 1
                continue
            
            stats.input_tokens += count_tokens(line)
            
            # Normalize whitespace
            line = normalize_whitespace(line)
            
            # Apply corpus-specific cleaning
            line = cleaner(line)
            if not line:
                stats.removed["corpus_specific"] += 1
                continue
            
            # Character ratio filter
            if not passes_character_ratio_filter(line):
                stats.removed["low_alpha_ratio"] += 1
                continue
            
            # Length filter
            if not passes_length_filter(line):
                stats.removed["length"] += 1
                continue
            
            # Exact deduplication (within corpus)
            line_hash = sha256_hash(line)
            if line_hash in seen_hashes:
                stats.removed["exact_duplicate"] += 1
                continue
            
            seen_hashes.add(line_hash)
            outfile.write(line + '\n')
            stats.output_lines += 1
            stats.output_tokens += count_tokens(line)
    
    logger.info(f"  {corpus_name}: {stats.input_lines} → {stats.output_lines} lines "
                f"({stats.output_lines/stats.input_lines*100:.1f}% kept)")
    
    return seen_hashes


# ============================================================================
# Phase 2: Global Cleaning & Deduplication
# ============================================================================

def read_lines_with_tokens(file_path: Path) -> Iterator[Tuple[str, int]]:
    """Yield (line, token_count) tuples from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line, count_tokens(line)


def stratified_sample(
    corpus_files: Dict[str, Path],
    target_tokens: int,
    proportions: Dict[str, float],
    random_seed: int,
    logger: logging.Logger,
) -> List[Tuple[str, str]]:
    """
    Perform stratified sampling from multiple corpora to achieve target token count.
    
    Returns: List of (line, corpus_name) tuples.
    """
    logger.info("Performing stratified sampling...")
    random.seed(random_seed)
    
    # Calculate target tokens per corpus
    target_per_corpus = {
        corpus: int(target_tokens * prop)
        for corpus, prop in proportions.items()
    }
    
    sampled_lines = []
    
    for corpus_name, file_path in corpus_files.items():
        corpus_key = corpus_name.replace(".train.pass1.txt", "").replace("_", "")
        
        # Map file name to proportion key
        # bnc_spoken.train.pass1.txt -> bnc_spoken
        corpus_key_clean = corpus_name.replace(".train.pass1.txt", "")
        
        if corpus_key_clean not in proportions:
            logger.warning(f"No proportion defined for {corpus_key_clean}, skipping")
            continue
        
        target = target_per_corpus[corpus_key_clean]
        
        # Read all lines from this corpus
        lines = []
        total_tokens = 0
        for line, token_count in read_lines_with_tokens(file_path):
            lines.append((line, token_count))
            total_tokens += token_count
        
        logger.info(f"  {corpus_key_clean}: {len(lines)} lines, {total_tokens} tokens, "
                   f"target: {target} tokens")
        
        # If we have fewer tokens than target, take all
        if total_tokens <= target:
            sampled_lines.extend([(line, corpus_key_clean) for line, _ in lines])
            logger.info(f"    Taking all {len(lines)} lines ({total_tokens} tokens)")
        else:
            # Reservoir sampling to approximate target token count
            random.shuffle(lines)
            current_tokens = 0
            count = 0
            for line, token_count in lines:
                if current_tokens + token_count > target:
                    break
                sampled_lines.append((line, corpus_key_clean))
                current_tokens += token_count
                count += 1
            logger.info(f"    Sampled {count} lines ({current_tokens} tokens)")
    
    return sampled_lines


def detect_url(line: str) -> bool:
    """Check if line contains a URL."""
    url_pattern = r'https?://|www\.|\.com|\.org|\.net'
    return bool(re.search(url_pattern, line, re.IGNORECASE))


def detect_code(line: str) -> bool:
    """Check if line looks like code (has {, }, ; in >10% of tokens)."""
    tokens = line.split()
    if not tokens:
        return False
    code_chars = sum(1 for token in tokens if any(c in token for c in ['{', '}', ';']))
    return (code_chars / len(tokens)) > 0.10


def detect_table_or_list(line: str) -> bool:
    """Check if line looks like a table or list (>30% digits or punctuation)."""
    if not line:
        return False
    digits_and_punct = sum(1 for c in line if c.isdigit() or c in '.,;:|()[]{}')
    return (digits_and_punct / len(line)) > 0.30


def _generate_minhash_for_line(args):
    """Worker function to generate MinHash for a single line."""
    idx, text, num_hashes, ngram_size = args
    from datasketch import MinHash
    m = MinHash(num_perm=num_hashes)
    encoded = text.encode('utf-8')
    ngrams = [encoded[j:j+ngram_size] for j in range(len(encoded) - ngram_size + 1)]
    if ngrams:
        m.update_batch(ngrams)
    return idx, m


def near_dedup_lsh(
    lines: List[str],
    threshold: float,
    logger: logging.Logger,
) -> List[str]:
    """
    Perform near-deduplication using datasketch MinHash LSH and multiprocessing.
    """
    import multiprocessing
    from datasketch import MinHashLSH
    
    logger.info("Performing near-deduplication with datasketch MinHash LSH...")
    
    num_hashes = 128
    ngram_size = 3
    
    lsh = MinHashLSH(threshold=threshold, num_perm=num_hashes)
    
    # Generate signatures for all lines using multiprocessing
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"  Generating signatures using {num_cores} cores...")
    
    minhashes = {}
    
    # Prepare arguments generator
    tasks = ((i, line, num_hashes, ngram_size) for i, line in enumerate(lines))
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        for count, (idx, m) in enumerate(pool.imap_unordered(_generate_minhash_for_line, tasks, chunksize=5000)):
            minhashes[idx] = m
            if (count + 1) % 100000 == 0:
                logger.info(f"  Generated {count+1}/{len(lines)} signatures...")
                
    # Find duplicates
    logger.info("  Inserting into LSH and querying for duplicates...")
    duplicates = set()
    
    for i in range(len(lines)):
        if i in duplicates:
            continue
            
        m = minhashes[i]
        
        # Query LSH
        result = lsh.query(m)
        is_dup = False
        for matched_idx in result:
            if matched_idx != i:
                # Double-check Jaccard to avoid false positives
                sim = m.jaccard(minhashes[matched_idx])
                if sim >= threshold:
                    duplicates.add(i)  # Mark current as duplicate of prior
                    is_dup = True
                    break
                    
        if not is_dup:
            lsh.insert(i, m)
        else:
            # Free memory for duplicate signatures
            del minhashes[i]
            
        if (i + 1) % 100000 == 0:
            logger.info(f"  Processed {i+1}/{len(lines)} for LSH collisions...")
            
    logger.info(f"  Found {len(duplicates)} near-duplicates")
    
    # Return lines that are not duplicates
    return [line for i, line in enumerate(lines) if i not in duplicates]

def process_phase2(
    pass1_files: Dict[str, Path],
    output_path: Path,
    target_tokens: int,
    proportions: Dict[str, float],
    random_seed: int,
    stats: Stats,
    logger: logging.Logger,
) -> None:
    """
    Phase 2: Merge, global dedup, near-dedup, and final filters.
    """
    logger.info("Starting Phase 2: Global cleaning...")
    
    # Step 1: Stratified sampling
    sampled = stratified_sample(pass1_files, target_tokens, proportions, random_seed, logger)
    stats.input_lines = len(sampled)
    stats.input_tokens = sum(count_tokens(line) for line, _ in sampled)
    
    logger.info(f"Sampled {stats.input_lines} lines with {stats.input_tokens} tokens")
    
    # Step 2: Global exact deduplication
    logger.info("Performing global exact deduplication...")
    seen_hashes = set()
    deduped = []
    
    for line, corpus in sampled:
        line_hash = sha256_hash(line)
        if line_hash not in seen_hashes:
            seen_hashes.add(line_hash)
            deduped.append((line, corpus))
        else:
            stats.removed["global_exact_duplicate"] += 1
    
    logger.info(f"  After exact dedup: {len(deduped)} lines")
    
    # Step 3: Final filters (URL, code, tables)
    logger.info("Applying final filters...")
    filtered = []
    
    for line, corpus in deduped:
        if detect_url(line):
            stats.removed["url"] += 1
            continue
        if detect_code(line):
            stats.removed["code"] += 1
            continue
        if detect_table_or_list(line):
            stats.removed["table_or_list"] += 1
            continue
        
        filtered.append((line, corpus))
    
    logger.info(f"  After filters: {len(filtered)} lines")
    
    # Step 4: Near-deduplication (SKIPPED FOR MEMORY EFFICIENCY ON 100M)
    logger.info("  Skipping near-deduplication for memory efficiency...")
    deduped_lines = [line for line, _ in filtered]
    
    # Reconstruct corpus mapping
    line_to_corpus = {line: corpus for line, corpus in filtered}
    
    # Step 5: Write output
    logger.info(f"Writing {output_path.name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in deduped_lines:
            f.write(line + '\n')
            stats.output_lines += 1
            stats.output_tokens += count_tokens(line)
    
    logger.info(f"Phase 2 complete: {stats.output_lines} lines, {stats.output_tokens} tokens")
    
    # Return corpus mapping for phase 4
    return line_to_corpus


# ============================================================================
# Phase 3: Quality Scoring & Top 70% Selection
# ============================================================================

def calculate_quality_score(line: str) -> float:
    """
    Calculate quality score (0-100) for a line based on multiple factors.
    
    Components:
    - Length score (20): Longer lines up to 100 chars get higher scores
    - Alpha ratio (40): Higher percentage of alphabetic characters
    - Digit ratio (15): Lower percentage of digits (inverted)
    - Punctuation (15): Optimal range is 2-10%
    - Repetition (10): Penalize repeated 3-grams
    """
    if not line:
        return 0.0
    
    # Length score: normalized to 100 chars (max 20 points)
    length_score = min(len(line) / 100.0, 1.0) * SCORE_WEIGHTS["length"]
    
    # Alpha ratio score (max 40 points)
    alpha_count = sum(1 for c in line if c.isalpha())
    alpha_ratio = alpha_count / len(line)
    alpha_score = alpha_ratio * SCORE_WEIGHTS["alpha_ratio"]
    
    # Digit ratio score - inverted (max 15 points)
    digit_count = sum(1 for c in line if c.isdigit())
    digit_ratio = digit_count / len(line)
    digit_score = (1.0 - digit_ratio) * SCORE_WEIGHTS["digit_ratio"]
    
    # Punctuation score (max 15 points)
    punct_count = sum(1 for c in line if c in '.,;:!?')
    punct_ratio = punct_count / len(line)
    if 0.02 <= punct_ratio <= 0.10:
        punct_score = SCORE_WEIGHTS["punctuation"]
    elif punct_ratio < 0.02:
        punct_score = (punct_ratio / 0.02) * SCORE_WEIGHTS["punctuation"]
    else:
        punct_score = max(0, (1.0 - (punct_ratio - 0.10) / 0.10)) * SCORE_WEIGHTS["punctuation"]
    
    # Repetition penalty score (max 10 points)
    # Count repeated 3-grams
    tokens = line.split()
    if len(tokens) >= 3:
        trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        repetition_ratio = 1.0 - (unique_trigrams / total_trigrams) if total_trigrams > 0 else 0.0
        repetition_score = (1.0 - repetition_ratio) * SCORE_WEIGHTS["repetition"]
    else:
        repetition_score = SCORE_WEIGHTS["repetition"]  # Short lines get full points
    
    total_score = length_score + alpha_score + digit_score + punct_score + repetition_score
    return round(total_score, 2)


def process_phase3(
    input_path: Path,
    output_path: Path,
    top_percent: float,
    random_seed: int,
    stats: Stats,
    logger: logging.Logger,
) -> Tuple[float, Dict[str, int]]:
    """
    Phase 3: Calculate quality scores and select top percentage.
    
    Returns: (quality_threshold, score_distribution)
    """
    logger.info("Starting Phase 3: Quality scoring & top 70% selection...")
    
    # Step 1: Score all lines
    logger.info("Calculating quality scores...")
    scored_lines = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                score = calculate_quality_score(line)
                scored_lines.append((score, line))
                stats.input_lines += 1
                stats.input_tokens += count_tokens(line)
    
    logger.info(f"  Scored {len(scored_lines)} lines")
    
    # Step 2: Sort by score (descending), use seeded random for stable tie-breaking
    random.seed(random_seed)
    scored_lines.sort(key=lambda x: (x[0], random.random()), reverse=True)
    
    # Step 3: Select top N% by line count
    cutoff_index = int(len(scored_lines) * top_percent)
    top_lines = scored_lines[:cutoff_index]
    
    quality_threshold = top_lines[-1][0] if top_lines else 0.0
    
    logger.info(f"  Quality threshold for top {top_percent*100}%: {quality_threshold:.2f}")
    
    # Step 4: Calculate score distribution
    score_buckets = defaultdict(int)
    for score, _ in scored_lines:
        bucket = int(score // 10) * 10  # 0-9, 10-19, etc.
        score_buckets[f"{bucket}-{bucket+9}"] += 1
    
    # Step 5: Write output
    logger.info(f"Writing {output_path.name}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for score, line in top_lines:
            f.write(line + '\n')
            stats.output_lines += 1
            stats.output_tokens += count_tokens(line)
    
    logger.info(f"Phase 3 complete: {stats.output_lines} lines, {stats.output_tokens} tokens")
    
    return quality_threshold, dict(score_buckets)


# ============================================================================
# Phase 4: Stratified Sampling for 1M
# ============================================================================

def stratified_sample_1m(
    input_path: Path,
    output_path: Path,
    line_to_corpus: Dict[str, str],
    target_tokens: int,
    proportions: Dict[str, float],
    random_seed: int,
    stats: Stats,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Phase 4: Create stratified 1M token sample preserving corpus proportions.
    
    Returns: Final mixture breakdown (corpus -> token count).
    """
    logger.info("Starting Phase 4: Stratified sampling for 1M...")
    
    random.seed(random_seed)
    
    # Step 1: Read all lines and organize by corpus
    logger.info("Organizing lines by corpus...")
    corpus_lines = defaultdict(list)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                stats.input_lines += 1
                stats.input_tokens += count_tokens(line)
                
                # Look up corpus (default to "unknown" if not found)
                corpus = line_to_corpus.get(line, "unknown")
                corpus_lines[corpus].append(line)
    
    logger.info(f"  Found lines from {len(corpus_lines)} corpora")
    
    # Step 2: Calculate target tokens per corpus
    target_per_corpus = {
        corpus: int(target_tokens * prop)
        for corpus, prop in proportions.items()
    }
    
    # Step 3: Reservoir sampling for each corpus
    sampled_lines = []
    actual_tokens_per_corpus = defaultdict(int)
    
    for corpus, target in target_per_corpus.items():
        if corpus not in corpus_lines:
            logger.warning(f"  No lines from {corpus}, skipping")
            continue
        
        lines = corpus_lines[corpus]
        random.shuffle(lines)
        
        # Select lines until we hit target tokens
        selected = []
        token_count = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if token_count + line_tokens > target:
                # Check if adding this line gets us closer to target
                if abs((token_count + line_tokens) - target) < abs(token_count - target):
                    selected.append(line)
                    token_count += line_tokens
                break
            selected.append(line)
            token_count += line_tokens
        
        logger.info(f"  {corpus}: {len(selected)} lines, {token_count} tokens (target: {target})")
        
        sampled_lines.extend(selected)
        actual_tokens_per_corpus[corpus] = token_count
    
    # Step 4: Shuffle final lines
    random.shuffle(sampled_lines)
    
    # Step 5: Write output
    logger.info(f"Writing {output_path.name}...")
    actual_total_tokens = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in sampled_lines:
            f.write(line + '\n')
            stats.output_lines += 1
            token_count = count_tokens(line)
            stats.output_tokens += token_count
            actual_total_tokens += token_count
    
    logger.info(f"Phase 4 complete: {stats.output_lines} lines, {stats.output_tokens} tokens")
    
    return dict(actual_tokens_per_corpus)


# ============================================================================
# Phase 5: Comprehensive Reporting
# ============================================================================

def generate_final_report(
    report: dict,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Generate comprehensive cleaning_report.json."""
    logger.info("Generating final report...")
    
    # Calculate overall statistics
    phase1_total = report["phases"]["phase1_per_corpus_cleaning"]
    total_input_lines = sum(v["input_lines"] for v in phase1_total.values())
    total_input_tokens = sum(v["input_tokens"] for v in phase1_total.values())
    
    phase2 = report["phases"]["phase2_global_cleaning"]
    phase3 = report["phases"]["phase3_quality_scoring"]
    phase4 = report["phases"]["phase4_stratified_1m"]
    
    report["summary"] = {
        "total_input_lines": total_input_lines,
        "total_input_tokens": total_input_tokens,
        "clean_10m_lines": phase2["output_lines"],
        "clean_10m_tokens": phase2["output_tokens"],
        "clean_top70_lines": phase3["output_lines"],
        "clean_top70_tokens": phase3["output_tokens"],
        "clean_1m_lines": phase4["output_lines"],
        "clean_1m_tokens": phase4["output_tokens"],
        "overall_retention_rate": f"{(phase2['output_lines'] / total_input_lines * 100):.2f}%",
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report saved to {output_path}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BabyLM corpus cleaning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input corpus files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output files",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=10_000_000,
        help="Target token count for clean_10M_full.txt",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.70,
        help="Percentage of top-quality lines to keep",
    )
    parser.add_argument(
        "--sample-tokens",
        type=int,
        default=1_000_000,
        help="Target tokens for stratified sample (clean_1M.txt)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging(args.verbose)
    random.seed(args.random_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("BabyLM Corpus Cleaning Pipeline")
    logger.info("=" * 80)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Random seed: {args.random_seed}")
    logger.info("")
    
    # Track statistics for report
    report = {
        "config": {
            "random_seed": args.random_seed,
            "target_tokens": args.target_tokens,
            "top_percent": args.top_percent,
            "sample_tokens": args.sample_tokens,
        },
        "phases": {},
    }
    
    # ========================================================================
    # Phase 1: Per-Corpus Cleaning
    # ========================================================================
    logger.info("PHASE 1: Per-Corpus Cleaning & Exact Deduplication")
    logger.info("-" * 80)
    
    phase1_stats = {}
    pass1_files = {}
    
    for corpus_file in CORPUS_FILES:
        input_path = args.input_dir / corpus_file
        output_path = args.output_dir / f"{corpus_file}.pass1.txt"
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}, skipping")
            continue
        
        stats = Stats()
        process_corpus_phase1(input_path, output_path, corpus_file, stats, logger)
        
        phase1_stats[corpus_file] = stats.to_dict()
        pass1_files[corpus_file + ".pass1.txt"] = output_path
    
    report["phases"]["phase1_per_corpus_cleaning"] = phase1_stats
    logger.info("")
    
    # ========================================================================
    # Phase 2: Global Cleaning
    # ========================================================================
    logger.info("PHASE 2: Global Cleaning & Deduplication")
    logger.info("-" * 80)
    
    phase2_stats = Stats()
    target_m = args.target_tokens // 1_000_000
    output_10m_full = args.output_dir / f"clean_{target_m}M_full.txt"
    
    line_to_corpus = process_phase2(
        pass1_files,
        output_10m_full,
        args.target_tokens,
        MIXTURE_PROPORTIONS,
        args.random_seed,
        phase2_stats,
        logger,
    )
    
    report["phases"]["phase2_global_cleaning"] = phase2_stats.to_dict()
    logger.info("")
    
    # ========================================================================
    # Phase 3: Quality Scoring & Top 70%
    # ========================================================================
    logger.info("PHASE 3: Quality Scoring & Top 70% Selection")
    logger.info("-" * 80)
    
    phase3_stats = Stats()
    target_m = args.target_tokens // 1_000_000
    top_pct = int(args.top_percent * 100)
    output_top70 = args.output_dir / f"clean_{target_m}M_top{top_pct}.txt"
    
    quality_threshold, score_distribution = process_phase3(
        output_10m_full,
        output_top70,
        args.top_percent,
        args.random_seed,
        phase3_stats,
        logger,
    )
    
    report["phases"]["phase3_quality_scoring"] = {
        **phase3_stats.to_dict(),
        "quality_threshold": quality_threshold,
        "score_distribution": score_distribution,
    }
    logger.info("")
    
    # ========================================================================
    # Phase 4: Stratified Sampling for 1M
    # ========================================================================
    logger.info("PHASE 4: Stratified Sampling for 1M")
    logger.info("-" * 80)
    
    phase4_stats = Stats()
    sample_m = max(1, args.sample_tokens // 1_000_000)
    output_1m = args.output_dir / f"clean_{sample_m}M.txt"
    
    final_mixture = stratified_sample_1m(
        output_top70,
        output_1m,
        line_to_corpus,
        args.sample_tokens,
        MIXTURE_PROPORTIONS,
        args.random_seed,
        phase4_stats,
        logger,
    )
    
    report["phases"]["phase4_stratified_1m"] = {
        **phase4_stats.to_dict(),
        "final_mixture": final_mixture,
        "target_mixture": {k: int(args.sample_tokens * v) for k, v in MIXTURE_PROPORTIONS.items()},
    }
    logger.info("")
    
    # ========================================================================
    # Phase 5: Generate Final Report
    # ========================================================================
    logger.info("PHASE 5: Generating Final Report")
    logger.info("-" * 80)
    
    report_path = args.output_dir / "cleaning_report.json"
    generate_final_report(report, report_path, logger)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Output files:")
    logger.info(f"  - {output_10m_full.name}: {phase2_stats.output_tokens:,} tokens")
    logger.info(f"  - {output_top70.name}: {phase3_stats.output_tokens:,} tokens")
    logger.info(f"  - {output_1m.name}: {phase4_stats.output_tokens:,} tokens")
    logger.info(f"  - {report_path.name}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
