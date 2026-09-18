"""
MLX implementation of jina-reranker-v3 for Apple Silicon.

This file is adapted from the official model repository:
  https://huggingface.co/jinaai/jina-reranker-v3-mlx/blob/main/rerank.py

License: CC BY-NC 4.0 (contact Jina AI for commercial usage)

Changes from upstream:
- Removed the __main__ example block
- Type annotations added to public methods
- top_n parameter added to rerank() (upstream only exposed it implicitly)
"""

import numpy as np


class _MLPProjector:
    def __init__(self):
        import mlx.nn as nn  # ty: ignore[unresolved-import]

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)

    def __call__(self, x):
        import mlx.nn as nn  # ty: ignore[unresolved-import]

        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x


def _load_projector(projector_path: str) -> _MLPProjector:
    import mlx.core as mx  # ty: ignore[unresolved-import]
    from safetensors import safe_open

    projector = _MLPProjector()
    with safe_open(projector_path, framework="numpy") as f:
        projector.linear1.weight = mx.array(f.get_tensor("linear1.weight"))
        projector.linear2.weight = mx.array(f.get_tensor("linear2.weight"))
    return projector


def _sanitize(text: str, special_tokens: dict[str, str]) -> str:
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text


def _format_prompt(query: str, docs: list[str], special_tokens: dict[str, str]) -> str:
    query = _sanitize(query, special_tokens)
    docs = [_sanitize(d, special_tokens) for d in docs]

    doc_token = special_tokens["doc_embed_token"]
    query_token = special_tokens["query_embed_token"]

    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    body = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    body += "\n".join(f'<passage id="{i}">\n{doc}{doc_token}\n</passage>' for i, doc in enumerate(docs))
    body += f"\n<query>\n{query}{query_token}\n</query>"
    return prefix + body + suffix


class MLXReranker:
    """
    MLX-accelerated jina-reranker-v3 for Apple Silicon.

    Loads the model from a local directory (use huggingface_hub.snapshot_download
    to fetch jinaai/jina-reranker-v3-mlx if you don't have it already).
    """

    _SPECIAL_TOKENS = {
        "query_embed_token": "<|rerank_token|>",
        "doc_embed_token": "<|embed_token|>",
    }
    _DOC_TOKEN_ID = 151670
    _QUERY_TOKEN_ID = 151671

    def __init__(self, model_path: str, projector_path: str):
        from mlx_lm import load  # ty: ignore[unresolved-import]

        self.model, self.tokenizer = load(model_path)
        self.model.eval()
        self.projector = _load_projector(projector_path)

    def rerank(self, query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
        """
        Rank documents by relevance to a query.

        Returns a list of dicts with keys: document, relevance_score, index.
        Sorted by descending relevance_score.
        """
        import mlx.core as mx  # ty: ignore[unresolved-import]

        prompt = _format_prompt(query, documents, self._SPECIAL_TOKENS)
        input_ids = self.tokenizer.encode(prompt)
        hidden_states = self.model.model([input_ids])[0]  # [seq_len, hidden_size]

        input_ids_np = np.array(input_ids)
        query_positions = np.where(input_ids_np == self._QUERY_TOKEN_ID)[0]
        doc_positions = np.where(input_ids_np == self._DOC_TOKEN_ID)[0]

        if len(query_positions) == 0:
            raise ValueError("Query embed token not found in prompt")
        if len(doc_positions) == 0:
            raise ValueError("Document embed tokens not found in prompt")

        query_hidden = mx.expand_dims(hidden_states[int(query_positions[0])], axis=0)
        doc_hidden = mx.stack([hidden_states[int(p)] for p in doc_positions])

        query_emb = self.projector(query_hidden)  # [1, 512]
        doc_emb = self.projector(doc_hidden)  # [num_docs, 512]

        query_exp = mx.broadcast_to(mx.expand_dims(query_emb, 0), (1, len(documents), 512))
        doc_exp = mx.expand_dims(doc_emb, 0)

        scores = mx.sum(doc_exp * query_exp, axis=-1) / (
            mx.sqrt(mx.sum(doc_exp * doc_exp, axis=-1)) * mx.sqrt(mx.sum(query_exp * query_exp, axis=-1))
        )  # [1, num_docs]
        scores_np = np.array(scores[0])

        order = np.argsort(scores_np)[::-1]
        n = min(top_n, len(documents)) if top_n is not None else len(documents)
        return [
            {
                "document": documents[order[i]],
                "relevance_score": float(scores_np[order[i]]),
                "index": int(order[i]),
            }
            for i in range(n)
        ]
