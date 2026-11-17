"""Tests for Milestones 1–4 of Educational Content Expander."""
from app.core import NgramLM, milestone_1, milestone_2, milestone_3, milestone_4


def test_milestone_1(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Photosynthesis converts sunlight into chemical energy. "
        "Plants absorb carbon dioxide and water.",
        encoding="utf-8",
    )
    data = milestone_1(sample)
    assert len(data["sentences"]) == 2
    assert "photosynthesis" in data["vocabulary"]
    assert len(data["tokens"]) == 12


def test_milestone_2(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Energy flows through ecosystems. "
        "Energy transforms from potential energy into kinetic energy.",
        encoding="utf-8",
    )
    model = milestone_2(corpus_path=sample, n=2, k=0.5)
    assert isinstance(model, NgramLM)
    # Unseen pair must still receive strictly positive probability under add-k smoothing
    unseen_prob = model.prob("ecosystems", "quantum")
    seen_prob = model.prob("kinetic", "energy")
    assert unseen_prob > 0.0
    assert seen_prob > unseen_prob
    assert model.perplexity("Energy flows through ecosystems.") > 1.0


def test_milestone_3(tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Cellular respiration breaks down glucose molecules to release usable energy. "
        "Cellular respiration powers essential metabolic functions inside living cells.",
        encoding="utf-8",
    )
    expanded = milestone_3(
        "cellular respiration",
        corpus_path=sample,
        n=3,
        k=0.5,
        seed=1,
        min_len=3,
        max_len=12,
    )
    assert expanded.startswith("Cellular respiration")
    assert expanded.endswith(".")
    assert len(expanded.split()) >= 5


def test_milestone_4(capsys, tmp_path):
    sample = tmp_path / "edu.txt"
    sample.write_text(
        "Natural selection drives evolutionary adaptation across generations.",
        encoding="utf-8",
    )
    milestone_4(["expand", "natural selection", "--corpus", str(sample), "--min-len", "2", "--max-len", "8"])
    out = capsys.readouterr().out.strip()
    assert out.startswith("Natural selection")
    assert out.endswith(".")