"""Microbenchmark for ``compute_semantic_links_within_batch``."""

import random
import time
import tracemalloc
from array import array

import numpy as np
from hindsight_api.engine.retain.link_utils import compute_semantic_links_within_batch
from rich.console import Console
from rich.table import Table

console = Console()


def _legacy_baseline(unit_ids, embeddings, top_k=50, *, threshold=0.7):
    """Legacy pre-PR implementation with float64 and argsort."""
    if len(unit_ids) < 2:
        return []
    links = []
    new_embeddings_matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(new_embeddings_matrix, axis=1)
    valid_embeddings = np.isfinite(new_embeddings_matrix).all(axis=1) & np.isfinite(norms) & (norms > 0)
    normalized_embeddings = np.zeros_like(new_embeddings_matrix)
    normalized_embeddings[valid_embeddings] = (
        new_embeddings_matrix[valid_embeddings] / norms[valid_embeddings, np.newaxis]
    )
    block_rows = 256
    for start in range(0, len(unit_ids), block_rows):
        stop = min(start + block_rows, len(unit_ids))
        block_similarities = normalized_embeddings[start:stop] @ normalized_embeddings.T
        block_similarities[:, ~valid_embeddings] = -np.inf
        for i in range(start, stop):
            if not valid_embeddings[i]:
                continue
            similarities = block_similarities[i - start]
            similarities[i] = -np.inf
            above_threshold = np.where(similarities >= threshold)[0]
            if len(above_threshold) > 0:
                sorted_indices = above_threshold[np.argsort(-similarities[above_threshold])][:top_k]
                for other_idx in sorted_indices:
                    other_id = unit_ids[other_idx]
                    similarity = float(min(1.0, max(0.0, similarities[other_idx])))
                    links.append((unit_ids[i], other_id, "semantic", similarity, None))
    return links


def run_benchmark(
    n_values: list[int] = [50, 200, 500, 1700],
    dim: int = 1536,
    threshold: float = 0.7,
    top_k: int = 50,
    repeats: int = 5,
):
    table = Table(title="Within-Batch Semantic Links Benchmark (Wall Time, CPU Time & Peak RAM)")
    table.add_column("Scenario", justify="left")
    table.add_column("N", justify="right", style="cyan")
    table.add_column("Base Wall (ms)", justify="right")
    table.add_column("Opt Wall (ms)", justify="right", style="green")
    table.add_column("Base CPU (ms)", justify="right")
    table.add_column("Opt CPU (ms)", justify="right", style="green")
    table.add_column("Speedup", justify="right", style="bold green")
    table.add_column("Base RAM (MB)", justify="right")
    table.add_column("Opt RAM (MB)", justify="right", style="yellow")
    table.add_column("RAM Cut", justify="right", style="bold yellow")
    table.add_column("Links", justify="right")

    for scenario, thresh, is_uniform in [
        ("Sparse Production (Gaussian, th=0.7)", 0.7, False),
        ("Dense Matching Stress (Uniform, th=0.3)", 0.3, True),
    ]:
        for n in n_values:
            if is_uniform:
                embeddings = [array("f", [random.random() for _ in range(dim)]) for _ in range(n)]
            else:
                embeddings = [array("f", [random.gauss(0, 1) for _ in range(dim)]) for _ in range(n)]
            unit_ids = [f"unit_{i}" for i in range(n)]

            # --- Baseline profiling ---
            wall_base, cpu_base = [], []
            tracemalloc.start()
            for _ in range(repeats):
                tw0 = time.perf_counter()
                tc0 = time.process_time()
                _legacy_baseline(unit_ids, embeddings, top_k=top_k, threshold=thresh)
                tc1 = time.process_time()
                tw1 = time.perf_counter()
                wall_base.append((tw1 - tw0) * 1000)
                cpu_base.append((tc1 - tc0) * 1000)
            _, peak_base = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # --- Optimized profiling ---
            wall_opt, cpu_opt = [], []
            links_count = 0
            tracemalloc.start()
            for _ in range(repeats):
                tw0 = time.perf_counter()
                tc0 = time.process_time()
                links = compute_semantic_links_within_batch(unit_ids, embeddings, top_k=top_k, threshold=thresh)
                tc1 = time.process_time()
                tw1 = time.perf_counter()
                wall_opt.append((tw1 - tw0) * 1000)
                cpu_opt.append((tc1 - tc0) * 1000)
                links_count = len(links)
            _, peak_opt = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            avg_base_wall = sum(wall_base) / len(wall_base)
            avg_opt_wall = sum(wall_opt) / len(wall_opt)
            avg_base_cpu = sum(cpu_base) / len(cpu_base)
            avg_opt_cpu = sum(cpu_opt) / len(cpu_opt)

            speedup = avg_base_wall / avg_opt_wall if avg_opt_wall > 0 else 1.0
            base_mb = peak_base / (1024 * 1024)
            opt_mb = peak_opt / (1024 * 1024)
            ram_cut = ((base_mb - opt_mb) / base_mb) * 100 if base_mb > 0 else 0.0

            table.add_row(
                scenario,
                str(n),
                f"{avg_base_wall:.2f}",
                f"{avg_opt_wall:.2f}",
                f"{avg_base_cpu:.2f}",
                f"{avg_opt_cpu:.2f}",
                f"{speedup:.2f}x",
                f"{base_mb:.2f}",
                f"{opt_mb:.2f}",
                f"-{ram_cut:.1f}%",
                str(links_count),
            )

    console.print(table)


def main():
    run_benchmark()


if __name__ == "__main__":
    main()
