"""Tests for the benchmark adapters.

**No network, no 112 MB download.** Every test here runs against inline
fixtures. A test suite that fetched a third-party dataset would be slow, would
fail on someone else's laptop for reasons unrelated to the code, and would make
CI depend on a host nobody here controls.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from oiax.eval.benchmarks import (
    SKILLRET_FILES,
    SRABENCH_DOMAIN,
    SRABENCH_FILES,
    SkillRetCorpus,
    SRABenchCorpus,
    _build_family_map,
    family_confusion,
    load_skillret_labelled,
    load_srabench_labelled,
    subsample,
)

SKILLS = [
    {"id": "id-a", "name": "deploy", "description": "deploying to production", "body": "A"},
    {"id": "id-b", "name": "deploy", "description": "rolling back a release", "body": "B"},
    {"id": "id-c", "name": "test", "description": "writing unit tests", "body": "C"},
    {"id": "id-d", "name": "nodesc", "description": "   ", "body": "D"},
]
QUERIES = [
    {"id": "q1", "query": "how do I ship to prod"},
    {"id": "q2", "query": "how do I write a test"},
    {"id": "q3", "query": "unanswerable after truncation"},
    {"id": "q4", "query": ""},
]
QRELS = [
    {"query_id": "q1", "skill_id": "id-a", "relevance": 1},
    {"query_id": "q1", "skill_id": "id-b", "relevance": 1},
    {"query_id": "q2", "skill_id": "id-c", "relevance": 1},
    {"query_id": "q3", "skill_id": "id-zzz", "relevance": 1},
    {"query_id": "q4", "skill_id": "id-a", "relevance": 1},
    {"query_id": "q2", "skill_id": "id-b", "relevance": 0},  # judged NOT relevant
]


@pytest.fixture
def bench(tmp_path):
    def dump(name, rows):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return p

    return {
        "skills": dump("skills.jsonl", SKILLS),
        "queries": dump("queries.jsonl", QUERIES),
        "qrels": dump("qrels.jsonl", QRELS),
    }


# ── corpus ──────────────────────────────────────────────────────────────────


def test_documents_are_keyed_by_id_not_name(bench):
    # 6,006 real skills carry only 5,801 distinct names, and the qrels key on
    # id. Keying by name would silently merge 205 skills into their neighbours.
    docs = list(SkillRetCorpus(bench["skills"]).documents())
    names = [d.name for d in docs]
    assert "id-a" in names and "id-b" in names
    assert "deploy" not in names


def test_the_description_becomes_the_routing_surface(bench):
    docs = {d.name: d for d in SkillRetCorpus(bench["skills"]).documents()}
    assert docs["id-a"].trigger_line == "deploying to production"
    # The body is carried for delivery and is not the scored surface.
    assert docs["id-a"].body == "A"


def test_a_skill_with_no_description_is_dropped(bench):
    # It has no routing surface at all. Including it would put a document in the
    # index that nothing can ever match.
    assert "id-d" not in {d.name for d in SkillRetCorpus(bench["skills"]).documents()}


def test_limit_truncates_deterministically(bench):
    a = SkillRetCorpus(bench["skills"], limit=2, seed=7).ids()
    b = SkillRetCorpus(bench["skills"], limit=2, seed=7).ids()
    assert a == b and len(a) == 2
    # A size curve measured on a different random subset at each point is
    # measuring the subset.


# ── labels ──────────────────────────────────────────────────────────────────


def test_labels_carry_every_gold_skill(bench):
    loaded = load_skillret_labelled(bench["queries"], bench["qrels"])
    items = {i["prompt"]: i["expected"] for i in loaded}
    assert sorted(items["how do I ship to prod"]) == ["id-a", "id-b"]


def test_a_zero_relevance_judgement_is_not_gold(bench):
    loaded = load_skillret_labelled(bench["queries"], bench["qrels"])
    items = {i["prompt"]: i["expected"] for i in loaded}
    assert items["how do I write a test"] == ["id-c"]


def test_an_empty_query_is_dropped(bench):
    prompts = [i["prompt"] for i in load_skillret_labelled(bench["queries"], bench["qrels"])]
    assert "" not in prompts


def test_restrict_to_drops_queries_unanswerable_after_truncation(bench):
    # Truncating the corpus without truncating the queries would leave prompts
    # whose gold document does not exist — depressing recall for a reason that
    # has nothing to do with the router.
    ids = {"id-c"}
    items = load_skillret_labelled(bench["queries"], bench["qrels"], restrict_to=ids)
    assert [i["expected"] for i in items] == [["id-c"]]


def test_a_partially_present_gold_set_is_dropped_whole(bench):
    # q1 needs id-a AND id-b. With only id-a present it is unanswerable at
    # recall 1.0 and would silently cap the ceiling.
    items = load_skillret_labelled(bench["queries"], bench["qrels"], restrict_to={"id-a"})
    assert items == []


def test_the_benchmark_has_no_negatives_so_false_alarms_are_unmeasurable(bench):
    items = load_skillret_labelled(bench["queries"], bench["qrels"])
    assert all(i["expected"] for i in items), (
        "every SkillRet query has a gold skill — a false-alarm rate cannot be "
        "measured here, and any result quoted from it carries that gap"
    )


# ── sampling ────────────────────────────────────────────────────────────────


def test_subsample_is_deterministic():
    items = [{"prompt": str(i), "expected": ["x"]} for i in range(100)]
    assert subsample(items, 10, seed=3) == subsample(items, 10, seed=3)
    assert len(subsample(items, 10)) == 10


def test_subsample_larger_than_the_set_returns_everything():
    items = [{"prompt": "a", "expected": ["x"]}]
    assert len(subsample(items, 50)) == 1


# ── the fetch contract ──────────────────────────────────────────────────────


def test_the_test_split_is_what_gets_fetched():
    # Scoring against the split the benchmark designates for testing is the only
    # honest option; the 335 MB combined corpus is deliberately not used.
    assert all("test" in rel for rel in SKILLRET_FILES.values())

# ── family confusion ─────────────────────────────────────────────────────


def test_build_family_map_from_skillret_records():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        f.write(
            '{"id": "a", "name": "Python",'
            ' "major": "Software Engineering", "sub": "Development"}\n'
            '{"id": "b", "name": "Python",'
            ' "major": "Data & ML", "sub": "Data Analysis"}\n'
            '{"id": "c", "name": "Unique Skill", "major": "DevOps", "sub": ""}\n'
        )
        path = f.name

    try:
        out: dict[str, str] = {}
        _build_family_map(path, out)

        assert "name:python" in out["a"]
        assert "name:python" in out["b"]
        assert "major:Software Engineering" in out["a"]
        assert "major:Data & ML" in out["b"]
        assert "name:unique skill" not in out.get("c", "")
    finally:
        import os
        os.unlink(path)


def test_family_confusion_empty_returns_zero():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as f:
        json.dump(
            {"id": "a", "name": "Test", "description": "testing code",
             "major": "DevOps", "sub": "", "body": "body"}, f
        )
        f.write("\n")
        path = f.name

    try:
        result = family_confusion(path, [])
        assert result["n_queries"] == 0
        assert result["rate"] == 0.0
    finally:
        import os
        os.unlink(path)


# ── SRA-Bench (second out-of-org corpus, #46) ────────────────────────────────

SRABENCH_CORPUS = [
    {
        "skill_id": "medcalcbench_000",
        "name": "BMI",
        "description": "Compute BMI from weight and height.",
        "content": "full doc A",
    },
    {
        "skill_id": "medcalcbench_001",
        "name": "CHA2DS2-VASc",
        "description": "Stroke risk in atrial fibrillation.",
        "content": "full doc B",
    },
    {
        "skill_id": "medcalcbench_002",
        "name": "nodesc",
        "description": "   ",
        "content": "full doc C",
    },
    {
        "skill_id": "champ_000",
        "name": "Other domain",
        "description": "should never appear in a medcalcbench slice",
        "content": "full doc D",
    },
]
SRABENCH_INSTANCES = [
    {
        "instance_id": "medcalcbench_00000",
        "dataset": "medcalcbench",
        "question": "what is the BMI",
        "skill_annotations": ["medcalcbench_000"],
    },
    {
        "instance_id": "medcalcbench_00001",
        "dataset": "medcalcbench",
        "question": "stroke risk please",
        "skill_annotations": ["medcalcbench_001"],
    },
    {
        "instance_id": "medcalcbench_00002",
        "dataset": "medcalcbench",
        "question": "unanswerable after truncation",
        "skill_annotations": ["medcalcbench_999"],
    },
    {
        "instance_id": "medcalcbench_00003",
        "dataset": "medcalcbench",
        "question": "",
        "skill_annotations": ["medcalcbench_000"],
    },
    {
        "instance_id": "champ_00000",
        "dataset": "champ",
        "question": "wrong domain, must be excluded by dataset filter",
        "skill_annotations": ["champ_000"],
    },
]


@pytest.fixture
def srabench(tmp_path):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(SRABENCH_CORPUS), encoding="utf-8")
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(json.dumps(SRABENCH_INSTANCES), encoding="utf-8")
    return {"corpus": corpus_path, "instances": instances_path}


def test_srabench_files_point_at_the_configured_domain():
    assert SRABENCH_FILES["instances"] == f"instances/{SRABENCH_DOMAIN}.json"
    assert SRABENCH_DOMAIN == "medcalcbench"


def test_srabench_corpus_filters_to_the_domain_prefix(srabench):
    docs = {d.name: d for d in SRABenchCorpus(srabench["corpus"]).documents()}
    assert "medcalcbench_000" in docs
    assert "medcalcbench_001" in docs
    # The shared corpus.json also carries other domains; they must not leak in.
    assert "champ_000" not in docs


def test_srabench_description_becomes_the_routing_surface(srabench):
    docs = {d.name: d for d in SRABenchCorpus(srabench["corpus"]).documents()}
    assert docs["medcalcbench_000"].trigger_line == "Compute BMI from weight and height."
    assert docs["medcalcbench_000"].body == "full doc A"


def test_srabench_skill_with_no_description_is_dropped(srabench):
    assert "medcalcbench_002" not in {
        d.name for d in SRABenchCorpus(srabench["corpus"]).documents()
    }


def test_srabench_ids_reflects_the_domain_filter(srabench):
    ids = SRABenchCorpus(srabench["corpus"]).ids()
    assert ids == {"medcalcbench_000", "medcalcbench_001", "medcalcbench_002"}


def test_srabench_other_domain_slice_is_independent(srabench):
    docs = {d.name: d for d in SRABenchCorpus(srabench["corpus"], domain="champ").documents()}
    assert docs.keys() == {"champ_000"}


def test_srabench_labels_carry_the_gold_skill(srabench):
    loaded = load_srabench_labelled(srabench["instances"])
    items = {i["prompt"]: i["expected"] for i in loaded}
    assert items["what is the BMI"] == ["medcalcbench_000"]
    assert items["stroke risk please"] == ["medcalcbench_001"]


def test_srabench_labels_are_dataset_scoped(srabench):
    # The champ instance must never appear when loading the medcalcbench domain,
    # even though it sits in the same instances file structure.
    prompts = [i["prompt"] for i in load_srabench_labelled(srabench["instances"])]
    assert not any("wrong domain" in p for p in prompts)


def test_srabench_an_empty_question_is_dropped(srabench):
    prompts = [i["prompt"] for i in load_srabench_labelled(srabench["instances"])]
    assert "" not in prompts


def test_srabench_restrict_to_drops_unanswerable_after_truncation(srabench):
    ids = {"medcalcbench_000", "medcalcbench_001"}
    items = load_srabench_labelled(srabench["instances"], restrict_to=ids)
    prompts = {i["prompt"] for i in items}
    assert "unanswerable after truncation" not in prompts
    assert "what is the BMI" in prompts


def test_srabench_has_no_negatives_so_false_alarms_are_unmeasurable(srabench):
    items = load_srabench_labelled(srabench["instances"])
    assert all(i["expected"] for i in items), (
        "every SRA-Bench instance has a gold skill — a false-alarm rate cannot "
        "be measured here either, and any result quoted from it carries the gap"
    )


def test_srabench_fingerprint_changes_with_domain(srabench):
    fp_default = SRABenchCorpus(srabench["corpus"]).fingerprint()
    fp_other = SRABenchCorpus(srabench["corpus"], domain="champ").fingerprint()
    assert fp_default != fp_other
