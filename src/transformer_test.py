#!/usr/bin/env python3
"""experiment to compare rule vs HF zero-shot (plain vs cue).

loads cards with `QuizletLoader`, uses `rule_label_and_confidence`, runs a Hugging Face
zero-shot classifier plain and with a cue prepended.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parents[2]))
from quizlet_loader import QuizletLoader

DEFAULT_INDICES: List[int] = [1, 10, 11, 17, 18, 24, 25, 32, 33, 38, 39, 46, 47, 50]


def parse_indices(s: str) -> set[int]:
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="./src/flashcards/high-quality-testcards.json")
    p.add_argument("--indices", default=",".join(map(str, DEFAULT_INDICES)))
    p.add_argument("--template", default="This is {}.")
    p.add_argument("--model", default="facebook/bart-large-mnli")
    p.add_argument("--max-cards", type=int, default=500)
    # tie-break option: pass transformer threshold to loader
    p.add_argument("--transformer-threshold", type=float, default=0.5, help="If set, loader will prefer rule when transformer top-score < threshold")
    args = p.parse_args()

    indices = parse_indices(args.indices)

    # instantiate loader, pass transformer threshold
    loader_kwargs = {}
    if args.transformer_threshold is not None:
        loader_kwargs["transformer_confidence_threshold"] = args.transformer_threshold
    loader = QuizletLoader(file_path=args.file, file_format="json", **loader_kwargs)

    # Create or get shared LLM client (lazy init inside client)
    from llm import get_shared_zero_shot
    try:
        hf_clf = get_shared_zero_shot(args.model)
    except Exception as e:
        print("Failed to configure LLM client:", e)
        hf_clf = None

    rows = []

    for i, doc in enumerate(loader.lazy_load(), start=1):
        if i > args.max_cards:
            break
        if indices and i not in indices:
            continue

        term = doc.metadata.get("term") or ""
        definition = doc.metadata.get("definition") or ""
        text = f"Term: {term}\nDefinition: {definition}".strip()

        # get rule label
        rule_label, strong = loader.rule_label_and_confidence(term, definition)

        # choose cue using loader's cue_map and use_cues_when
        cue = ""
        if getattr(loader, "cue_map", None) and rule_label in loader.cue_map:
            cue_val = loader.cue_map[rule_label]
            chosen = cue_val[0] if isinstance(cue_val, (list, tuple)) else cue_val
            if loader.use_cues_when == "always":
                cue = chosen
            elif loader.use_cues_when == "strong_only" and strong:
                cue = chosen

        plain_pred = None
        plain_scores = None
        cue_pred = None
        cue_scores = None

        if hf_clf is not None:
            try:
                # reuse loader's label phrases
                candidate_phrases = list(loader.LABEL_PHRASES.values())
                inverse_map = {v: k for k, v in loader.LABEL_PHRASES.items()}
                out_plain = hf_clf.classify(text, candidate_labels=candidate_phrases, hypothesis_template=args.template)
                plain_phrase = out_plain["labels"][0]
                plain_pred = inverse_map.get(plain_phrase, plain_phrase)
                plain_scores = out_plain.get("scores")
            except Exception as e:
                plain_pred = f"error: {e}"

            if cue:
                text_with_cue = f"{cue} {text}"
                try:
                    out_cue = hf_clf.classify(text_with_cue, candidate_labels=candidate_phrases, hypothesis_template=args.template)
                    cue_phrase = out_cue["labels"][0]
                    cue_pred = inverse_map.get(cue_phrase, cue_phrase)
                    cue_scores = out_cue.get("scores")
                except Exception as e:
                    cue_pred = f"error: {e}"

            # final choice is what the loader stored (it ran its own tie-break)
            final_choice = doc.metadata.get("flashcard_type") or loader._classify_card(term, definition)

            rows.append({
                "index": i,
                "term": term,
                "definition": definition,
                "rule_label": rule_label,
                "plain_pred": plain_pred,
                "final_choice": final_choice,
            })

    # print final chosen label, the rule label, and the plain transformer prediction
    for r in rows:
        print(f"Card {r['index']}: {r['term']}")
        print("  def:", r["definition"]) 
        print("  final_choice:", r["final_choice"]) 
        print("  rule_label:", r["rule_label"]) 
        print("  plain_pred:", r["plain_pred"]) 


if __name__ == "__main__":
    main()
