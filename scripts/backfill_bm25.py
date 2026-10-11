"""Audit or backfill Qdrant BM25 sparse vectors for existing Mem0 points.

The command is dry-run by default. Pass --write to mutate the selected points.
It must run in the mem0 image so the Qdrant client and Qdrant/bm25 fastembed
model exactly match the online retrieval service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable
from typing import Any

from qdrant_client import QdrantClient, models


def build_filter(run_id: str | None, user_ids: list[str]) -> models.Filter | None:
    """Build an exact payload filter for the migration scope."""
    conditions = []
    if run_id:
        conditions.append(models.FieldCondition(key="run_id", match=models.MatchValue(value=run_id)))
    if user_ids:
        conditions.append(models.FieldCondition(key="user_id", match=models.MatchAny(any=user_ids)))
    return models.Filter(must=conditions) if conditions else None


def _point_vectors(point: Any) -> Any:
    if isinstance(point, dict):
        return point.get("vector")
    return getattr(point, "vector", None)


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return point.get("payload") or {}
    return getattr(point, "payload", None) or {}


def _point_id(point: Any) -> Any:
    if isinstance(point, dict):
        return point.get("id")
    return getattr(point, "id", None)


def has_bm25_vector(point: Any) -> bool:
    vectors = _point_vectors(point)
    return isinstance(vectors, dict) and vectors.get("bm25") is not None


def searchable_text(point: Any) -> str:
    payload = _point_payload(point)
    return str(payload.get("text_lemmatized") or payload.get("data") or "").strip()


def assert_bm25_slot(client: QdrantClient, collection_name: str) -> None:
    info = client.get_collection(collection_name=collection_name)
    sparse_config = info.config.params.sparse_vectors
    if not sparse_config or "bm25" not in sparse_config:
        raise RuntimeError(
            f"Collection '{collection_name}' has no 'bm25' sparse-vector slot. "
            "Create or migrate the collection schema before backfilling points."
        )


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _as_list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def backfill_bm25(
    client: QdrantClient,
    encoder: Any,
    *,
    collection_name: str,
    scroll_filter: models.Filter | None,
    page_size: int = 256,
    batch_size: int = 64,
    write: bool = False,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Audit selected points and optionally write missing BM25 named vectors."""
    if page_size <= 0 or batch_size <= 0:
        raise ValueError("page_size and batch_size must be positive")

    assert_bm25_slot(client, collection_name)
    progress = progress or (lambda _: None)
    summary: dict[str, Any] = {
        "collection": collection_name,
        "dry_run": not write,
        "force": force,
        "scanned": 0,
        "already_indexed": 0,
        "missing_vector": 0,
        "missing_text": 0,
        "planned": 0,
        "updated": 0,
    }
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=["bm25"],
        )
        targets: list[tuple[Any, str]] = []
        for point in points:
            summary["scanned"] += 1
            present = has_bm25_vector(point)
            if present:
                summary["already_indexed"] += 1
            else:
                summary["missing_vector"] += 1

            if present and not force:
                continue
            text = searchable_text(point)
            if not text:
                summary["missing_text"] += 1
                continue
            point_id = _point_id(point)
            if point_id is None:
                raise RuntimeError("Qdrant scroll returned a point without an id")
            targets.append((point_id, text))

        summary["planned"] += len(targets)
        if write:
            for batch in _chunks(targets, batch_size):
                sparse_vectors = list(encoder.embed([text for _, text in batch]))
                if len(sparse_vectors) != len(batch):
                    raise RuntimeError(
                        f"BM25 encoder returned {len(sparse_vectors)} vectors for a batch of {len(batch)} texts"
                    )
                updates = [
                    models.PointVectors(
                        id=point_id,
                        vector={
                            "bm25": models.SparseVector(
                                indices=_as_list(sparse.indices),
                                values=_as_list(sparse.values),
                            )
                        },
                    )
                    for (point_id, _), sparse in zip(batch, sparse_vectors)
                ]
                client.update_vectors(collection_name=collection_name, points=updates, wait=True)
                summary["updated"] += len(updates)

        progress(
            "scanned={scanned} missing_vector={missing_vector} missing_text={missing_text} "
            "planned={planned} updated={updated}".format(**summary)
        )
        if next_offset is None:
            break
        offset = next_offset

    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("QDRANT_URL"))
    parser.add_argument("--host", default=os.getenv("QDRANT_HOST", "qdrant"))
    parser.add_argument("--port", type=int, default=int(os.getenv("QDRANT_PORT", "6333")))
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--collection", default=os.getenv("COLLECTION_NAME", "memories"))
    parser.add_argument("--run-id", help="Exact run_id payload value")
    parser.add_argument("--user-id", action="append", default=[], help="Exact user_id; repeat for multiple users")
    parser.add_argument("--all", action="store_true", help="Select the whole collection")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default="Qdrant/bm25")
    parser.add_argument("--force", action="store_true", help="Re-encode points that already have a BM25 vector")
    parser.add_argument("--write", action="store_true", help="Apply updates; without this flag the command is read-only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.all and not args.run_id and not args.user_id:
        parser.error("choose a scope with --run-id, one or more --user-id, or --all")
    if args.all and (args.run_id or args.user_id):
        parser.error("--all cannot be combined with --run-id or --user-id")

    from fastembed import SparseTextEmbedding

    client_kwargs: dict[str, Any] = {}
    if args.api_key:
        client_kwargs["api_key"] = args.api_key
    if args.url:
        client_kwargs["url"] = args.url
    else:
        client_kwargs.update(host=args.host, port=args.port)

    client = QdrantClient(**client_kwargs)
    encoder = SparseTextEmbedding(model_name=args.model)
    result = backfill_bm25(
        client,
        encoder,
        collection_name=args.collection,
        scroll_filter=build_filter(args.run_id, args.user_id),
        page_size=args.page_size,
        batch_size=args.batch_size,
        write=args.write,
        force=args.force,
        progress=lambda message: print(message, file=sys.stderr),
    )
    result["model"] = args.model
    result["scope"] = {
        "all": args.all,
        "run_id": args.run_id,
        "user_ids": args.user_id,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
