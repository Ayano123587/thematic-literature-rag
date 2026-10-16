import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.resumable_training import (epoch_order, load_progress, optimize_epoch,
                                    restore_rng, rng_state, save_progress)


class TinyMechanicsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(.3)
        self.linear = torch.nn.Linear(3, 1)
    def forward(self, x, target):
        prediction = self.linear(self.dropout(x))
        return SimpleNamespace(loss=(prediction-target).square().mean())


def test_optimizer_dropout_and_rng_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(42); random.seed(42); np.random.seed(42)
    initial = TinyMechanicsModel()
    initial_rng = rng_state()
    batches = [lambda i=i: {"x": torch.tensor([[i/10, .2, -.3]]), "target": torch.tensor([[.4]])}
               for i in range(7)]
    full = copy.deepcopy(initial); optimizer = torch.optim.AdamW(full.parameters(), lr=.01)
    state = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.}
    assert optimize_epoch(full, optimizer, batches, state, 2, lambda _: None)
    expected_tail = (torch.rand(2), random.random(), np.random.random())

    model = copy.deepcopy(initial); opt = torch.optim.AdamW(model.parameters(), lr=.01)
    restore_rng(initial_rng)
    progress = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.}
    identity = {"model": "tiny-mechanics", "source": "fixture"}
    checkpoint = lambda p: save_progress(tmp_path, model, opt, identity, p)
    assert not optimize_epoch(model, opt, batches, progress, 2, checkpoint, deadline=0.)
    assert progress["next_microstep"] == 2
    restored = TinyMechanicsModel(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=.01)
    restored_progress = load_progress(tmp_path, restored, restored_opt, identity)
    assert optimize_epoch(restored, restored_opt, batches, restored_progress, 2,
                          lambda p: save_progress(tmp_path, restored, restored_opt, identity, p))
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in full.state_dict().items())
    assert restored_progress == state
    assert torch.equal(torch.rand(2), expected_tail[0])
    assert random.random() == expected_tail[1] and np.random.random() == expected_tail[2]
    assert next(iter(restored_opt.state.values()))["step"] == 4  # Partial final group gets a complete step.


def test_resume_identity_optimizer_and_byte_guards(tmp_path):
    model = TinyMechanicsModel(); opt = torch.optim.AdamW(model.parameters(), lr=.01)
    identity = {"source": "one"}; progress = {"epoch": 1, "next_microstep": 0, "loss_sum": 0.}
    receipt = save_progress(tmp_path, model, opt, identity, progress)
    with pytest.raises(ValueError, match="identity changed"):
        load_progress(tmp_path, model, opt, {"source": "two"})
    with pytest.raises(ValueError, match="lineage"):
        save_progress(tmp_path, model, opt, {"source": "two"}, progress)
    with pytest.raises(ValueError, match="optimizer configuration"):
        load_progress(tmp_path, model, torch.optim.AdamW(model.parameters(), lr=.02), identity)
    path = tmp_path / f"slot-{receipt['slot']}.pt"; path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        load_progress(tmp_path, model, opt, identity)


def test_order_does_not_consume_dropout_rng_and_midgroup_refused():
    state = torch.get_rng_state().clone()
    assert epoch_order(20,42,2) == epoch_order(20,42,2)
    assert torch.equal(state, torch.get_rng_state())
    model=TinyMechanicsModel();opt=torch.optim.AdamW(model.parameters())
    with pytest.raises(ValueError, match="inside an accumulation"):
        optimize_epoch(model,opt,[lambda: {}]*3,{"epoch":1,"next_microstep":1,"loss_sum":0.},2,lambda _:None)
