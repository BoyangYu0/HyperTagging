import torch

from hypertagging.data.direct_gpt import (
    build_direct_multi_gpt_batch,
    collate_direct_gpt_events,
    load_direct_gpt_events,
)
from hypertagging.losses.gpt_losses import distance, radius_loss
from hypertagging.losses.link_losses import link_metrics
from hypertagging.models.gpt_like import MultiGPT, ParticleEmbedder
from hypertagging.preprocessing.export_dataset import export_trees
from hypertagging.preprocessing.levelize_tree import assign_levels
from hypertagging.preprocessing.mdst_tree_builder import EventTree, FourVector, TreeNode


def _direct_tree(event_id: int) -> EventTree:
    tree = EventTree(event_id=event_id)
    tree.add_node(
        TreeNode(
            node_id=0,
            pdg=421,
            charge=0.0,
            p4=FourVector(0.3, 0.1, 0.0, 1.2),
            daughter_ids=[1, 2],
            n_daughters=2,
        )
    )
    tree.add_node(
        TreeNode(
            node_id=1,
            pdg=321,
            charge=1.0,
            p4=FourVector(0.2, 0.0, 0.0, 0.7),
            parent_id=0,
            reco_id="Track:1",
        )
    )
    tree.add_node(
        TreeNode(
            node_id=2,
            pdg=-211,
            charge=-1.0,
            p4=FourVector(0.1, 0.1, 0.0, 0.5),
            parent_id=0,
            reco_id="Track:2",
        )
    )
    tree.root_ids = [0]
    assign_levels(tree)
    return tree


def test_direct_parquet_to_multi_gpt_forward_and_backward(tmp_path):
    parquet = export_trees([_direct_tree(17), _direct_tree(18)], tmp_path / "direct.parquet")
    events = load_direct_gpt_events(parquet)
    structure = collate_direct_gpt_events(events)

    assert structure["feature"].shape == (2, 3, 11)
    assert structure["level_ids"].tolist() == [[0, 0, 1], [0, 0, 1]]
    assert structure["parent_indices"].tolist() == [[2, 2, -1], [2, 2, -1]]

    embedder = ParticleEmbedder(
        n_features=11,
        tr_width=8,
        tr_n_head=1,
        tr_n=1,
        tr_hidden_size=16,
        pdg_emb=3,
        dim_hyper=4,
        num_pdg=40,
        device="cpu",
    )
    embeddings = embedder(structure)
    batch = build_direct_multi_gpt_batch(embeddings, structure)

    assert torch.count_nonzero(batch["emb"][:, 2]) == 0
    assert not batch["src_mask"][0, 0, 1]
    assert batch["src_mask"][0, 0, 2]
    assert not batch["src_mask"][0, 2, 0]
    assert batch["src_mask"][0, 2, 2]

    model = MultiGPT(
        rec_width=8,
        rec_n_head=1,
        rec_n=1,
        rec_hidden_size=16,
        link_width=8,
        link_n_head=1,
        link_n=1,
        link_hidden_size=16,
        dim_hyper=4,
        device="cpu",
    )
    predicted_embeddings, predicted_links = model(batch)
    active = batch["lvl_code"].bool()
    linked = batch["links"] >= 0
    reconstruction_loss = distance(predicted_embeddings, batch["target"], active)
    link_loss, link_accuracy = link_metrics(predicted_links, batch["links"], linked)
    radial_loss = radius_loss(predicted_embeddings, batch["mass"], linked)
    loss = reconstruction_loss + link_loss + radial_loss
    loss.backward()

    assert predicted_embeddings.shape == (2, 3, 4)
    assert predicted_links.shape == (2, 3, 3)
    assert torch.isfinite(loss)
    assert torch.isfinite(link_accuracy)
    assert any(parameter.grad is not None for parameter in model.parameters())
