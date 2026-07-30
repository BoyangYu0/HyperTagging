import random

import numpy as np
import torch

from hypertagging.training.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)


def test_checkpoint_restores_scaler_and_all_rng_states(tmp_path):
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=1,
        schema_version="direct-mdst-tree-v4",
    )
    expected = (random.random(), np.random.rand(), torch.rand(1))
    _ = (random.random(), np.random.rand(), torch.rand(1))
    restore_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        restore_random_states=True,
    )
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_one_step_checkpoint_resume_matches_uninterrupted_two_steps(tmp_path):
    def initialize():
        torch.manual_seed(19)
        module = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        return module, optimizer, scheduler, scaler

    def step(module, optimizer, scheduler):
        optimizer.zero_grad(set_to_none=True)
        sample = torch.rand(4, 3)
        module(sample).square().mean().backward()
        optimizer.step()
        scheduler.step()

    uninterrupted = initialize()
    step(*uninterrupted[:3])
    step(*uninterrupted[:3])

    resumed = initialize()
    step(*resumed[:3])
    checkpoint = save_training_checkpoint(
        tmp_path / "exact.pt",
        model=resumed[0],
        optimizer=resumed[1],
        scheduler=resumed[2],
        scaler=resumed[3],
        step=1,
    )
    replacement = initialize()
    restore_training_checkpoint(
        checkpoint,
        model=replacement[0],
        optimizer=replacement[1],
        scheduler=replacement[2],
        scaler=replacement[3],
        restore_random_states=True,
    )
    step(*replacement[:3])
    for expected, actual in zip(
        uninterrupted[0].parameters(), replacement[0].parameters()
    ):
        torch.testing.assert_close(actual, expected)
    assert replacement[1].state_dict()["state"].keys() == uninterrupted[1].state_dict()[
        "state"
    ].keys()
    assert replacement[2].state_dict() == uninterrupted[2].state_dict()
