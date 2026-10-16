import copy
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch", reason="Optional CPU training mechanics")

from src.full_reranker_training import ListwiseRanker, summarize_rows
from src.resumable_training import load_progress, optimize_epoch, save_progress


class TinyScalarHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.2)
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return SimpleNamespace(logits=self.linear(self.dropout(x)))


def test_listwise_objective_and_group_geometry():
    model = ListwiseRanker(TinyScalarHead()).eval()
    x = torch.tensor([[1., 0.], [0., 1.], [.2, .3], [.4, .5], [.6, .7]])
    logits = model.ranker(x).logits[:, 0]
    expected = (-logits[:2].log_softmax(0)[0]-logits[2:].log_softmax(0)[0])/2
    assert torch.equal(model(x=x, counts=[2, 3]).loss, expected)
    for counts in ([1, 4], [True, 4], [], [2, 2]):
        with pytest.raises(ValueError):
            model(x=x, counts=counts)


def test_real_listwise_wrapper_optimizer_resume_is_exact(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(42)
    initial = ListwiseRanker(TinyScalarHead())
    rng = torch.get_rng_state().clone()
    batches = [lambda i=i: {"x": torch.tensor([[1., i/10], [.2, .3], [.4, .5]]), "counts": [3]}
               for i in range(5)]
    full = copy.deepcopy(initial); full_opt = torch.optim.AdamW(full.parameters(), lr=.01)
    final = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.}
    optimize_epoch(full, full_opt, batches, final, 2, lambda _: None)
    interrupted = copy.deepcopy(initial); opt = torch.optim.AdamW(interrupted.parameters(), lr=.01)
    torch.set_rng_state(rng)
    progress = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.}
    checkpoint = lambda p: save_progress(tmp_path, interrupted, opt, {"fixture": "ranking"}, p)
    assert not optimize_epoch(interrupted, opt, batches, progress, 2, checkpoint, deadline=0)
    restored = ListwiseRanker(TinyScalarHead()); restored_opt = torch.optim.AdamW(restored.parameters(), lr=.01)
    progress = load_progress(tmp_path, restored, restored_opt, {"fixture": "ranking"})
    assert optimize_epoch(restored, restored_opt, batches, progress, 2, lambda _: None)
    assert progress == final
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in full.state_dict().items())


def test_shard_summary_matches_weighted_question_metrics():
    rows = [{"recall": .5, "precision": .25, "candidate_recall": 1., "top4_ceiling": 1.,
             "origin": "a", "attribute": "action", "explicit": "explicit"},
            {"recall": 1., "precision": .5, "candidate_recall": 1., "top4_ceiling": .5,
             "origin": "b", "attribute": "action", "explicit": "implicit"}]
    report = summarize_rows(rows)
    assert report["recall_at_4"] == .75 and report["n"] == 2
    assert report["origin"]["a"]["recall_at_4"] == .5
    with pytest.raises(ValueError):
        summarize_rows([])
