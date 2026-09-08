from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from common import (
    load_model_and_tokenizer,
    load_run,
    pool_hidden,
    read_jsonl,
    resolve_hidden_indices,
    save_yaml,
    tokenizer_fingerprint,
    update_status,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract model-specific state representations and action probabilities."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("student", "teacher"))
    return parser.parse_args()


def compute_prefix_ppl(
    model: torch.nn.Module,
    states: list[dict],
    device: torch.device,
) -> np.ndarray:
    """Score each student-generated prefix once per trajectory."""
    result = np.full(len(states), np.nan, dtype=np.float32)
    grouped: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        grouped.setdefault(state["trajectory_id"], []).append(index)

    for trajectory_index, indices in enumerate(grouped.values()):
        longest = max(indices, key=lambda item: len(states[item]["input_ids"]))
        token_ids = torch.tensor(
            [states[longest]["input_ids"]], dtype=torch.long, device=device
        )
        prompt_count = int(states[longest]["prompt_token_count"])
        with torch.inference_mode():
            outputs = model(
                input_ids=token_ids,
                output_hidden_states=False,
                use_cache=False,
                logits_to_keep=0,
            )
            logits = outputs.logits[0, :-1]
            targets = token_ids[0, 1:]
            generated_start = max(0, prompt_count - 1)
            nll_parts = []
            for start in range(generated_start, len(targets), 64):
                stop = min(start + 64, len(targets))
                nll_parts.append(
                    F.cross_entropy(
                        logits[start:stop].float(),
                        targets[start:stop],
                        reduction="none",
                    )
                )
            generated_nll = torch.cat(nll_parts) if nll_parts else None

        if generated_nll is not None:
            cumulative = generated_nll.cumsum(dim=0)
            for state_index in indices:
                generated_count = len(states[state_index]["input_ids"]) - prompt_count
                if generated_count > 0:
                    mean_nll = cumulative[generated_count - 1] / generated_count
                    result[state_index] = float(torch.exp(mean_nll.clamp(max=50)))
        del outputs, logits, targets, generated_nll
        print(
            f"prefix PPL: scored {trajectory_index + 1}/{len(grouped)} trajectories",
            flush=True,
        )
    return result


def main() -> int:
    args = parse_args()
    run_dir, config = load_run(args.run_dir)
    states = read_jsonl(run_dir / "artifacts" / "states.jsonl")
    if not states:
        raise RuntimeError("No states were found; run collect_states.py first")
    update_status(run_dir, f"extracting_{args.role}")

    try:
        model_path = config["models"][args.role]
        model, tokenizer, device = load_model_and_tokenizer(model_path, config)
        expected = yaml.safe_load(
            (run_dir / "artifacts" / "tokenizer.yaml").read_text(encoding="utf-8")
        )
        fingerprint = tokenizer_fingerprint(tokenizer)
        if fingerprint != expected["fingerprint"]:
            raise RuntimeError(
                f"{args.role} tokenizer differs from the student tokenizer used to collect token IDs; "
                "discrete action comparisons would be invalid"
            )

        action_vocab = np.array(
            sorted({int(state["action_token_id"]) for state in states}), dtype=np.int64
        )
        action_column = {
            token_id: index for index, token_id in enumerate(action_vocab.tolist())
        }
        definitions = config["representations"]["definitions"]
        num_layers = int(model.config.num_hidden_layers)
        resolved_layers = {
            definition["name"]: resolve_hidden_indices(definition["layers"], num_layers)
            for definition in definitions
        }
        representation_rows: dict[str, list[np.ndarray]] = {
            definition["name"]: [] for definition in definitions
        }
        action_log_probs: list[np.ndarray] = []
        entropies: list[np.ndarray] = []
        max_probabilities: list[np.ndarray] = []
        topk_ids: list[np.ndarray] = []
        topk_log_probs: list[np.ndarray] = []
        full_log_probs: list[np.ndarray] = []
        batch_size = int(config["models"]["feature_batch_size"])
        top_k = int(config["models"]["top_k_distribution"])
        action_vocab_tensor = torch.tensor(
            action_vocab, device=device, dtype=torch.long
        )

        for start in range(0, len(states), batch_size):
            batch_states = states[start : start + batch_size]
            encoded = tokenizer.pad(
                {"input_ids": [state["input_ids"] for state in batch_states]},
                padding=True,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    logits_to_keep=1,
                )
                logits = outputs.logits[:, -1, :].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
                action_log_probs.append(
                    log_probs.index_select(-1, action_vocab_tensor).cpu().numpy()
                )
                entropies.append(
                    (-(probabilities * log_probs).sum(dim=-1)).cpu().numpy()
                )
                max_probabilities.append(probabilities.max(dim=-1).values.cpu().numpy())
                values, indices = torch.topk(log_probs, k=top_k, dim=-1)
                topk_ids.append(indices.cpu().numpy())
                topk_log_probs.append(values.cpu().numpy())
                if config["models"].get("store_full_log_probs", False):
                    full_log_probs.append(log_probs.cpu().numpy().astype(np.float16))

                for definition in definitions:
                    layers = [
                        outputs.hidden_states[index]
                        for index in resolved_layers[definition["name"]]
                    ]
                    hidden = torch.stack(layers, dim=0).mean(dim=0)
                    pooled = pool_hidden(hidden, attention_mask, definition).float()
                    if config["representations"].get("normalize", True):
                        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                    representation_rows[definition["name"]].append(
                        pooled.cpu().numpy().astype(np.float16)
                    )

            print(
                f"{args.role}: extracted {min(start + batch_size, len(states))}/{len(states)} states",
                flush=True,
            )

        prefix_ppl = np.full(len(states), np.nan, dtype=np.float32)
        if args.role in config["models"].get("prefix_ppl_roles", []):
            prefix_ppl = compute_prefix_ppl(model, states, device)

        arrays: dict[str, np.ndarray] = {
            "state_ids": np.asarray([state["state_id"] for state in states]),
            "action_token_ids": np.asarray(
                [state["action_token_id"] for state in states], dtype=np.int64
            ),
            "action_columns": np.asarray(
                [action_column[int(state["action_token_id"])] for state in states],
                dtype=np.int64,
            ),
            "action_vocab": action_vocab,
            "action_log_probs": np.concatenate(action_log_probs, axis=0).astype(
                np.float32
            ),
            "entropy": np.concatenate(entropies, axis=0).astype(np.float32),
            "max_probability": np.concatenate(max_probabilities, axis=0).astype(
                np.float32
            ),
            "topk_ids": np.concatenate(topk_ids, axis=0).astype(np.int64),
            "topk_log_probs": np.concatenate(topk_log_probs, axis=0).astype(np.float32),
            "prefix_ppl": prefix_ppl,
        }
        if full_log_probs:
            arrays["full_log_probs"] = np.concatenate(full_log_probs, axis=0)
        for name, rows in representation_rows.items():
            arrays[f"representation__{name}"] = np.concatenate(rows, axis=0)

        output_path = run_dir / "features" / f"{args.role}.npz"
        np.savez(output_path, **arrays)
        save_yaml(
            run_dir / "features" / f"{args.role}.yaml",
            {
                "model_path": model_path,
                "tokenizer_fingerprint": fingerprint,
                "states": len(states),
                "action_vocabulary_size": len(action_vocab),
                "representations": resolved_layers,
                "output": str(output_path),
            },
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        update_status(run_dir, f"extracted_{args.role}")
        return 0
    except Exception as exc:
        update_status(
            run_dir,
            "failed",
            stage=f"extract_{args.role}",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
