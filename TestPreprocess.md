Note: make sure virtual environment and output files are placed to the correct path, avoid putting huge files under the home repository `/afs/desy.de/user/b/boyangyu/`.
Note: current working node is CPU only, avoid calling CUDA, but should be aware of GPU compatibility.
Note: avoid running long tasks, the debugging tests should be done on small amount of samples (less than 10000)

1. Build working virtual environment for the current project with `uv`, necessary packages include `uproot`, `pytorch-cpu`, `pdg`, `awkward`, `onnx`. Install further packages when necessary.
2. Verify the data production and preprocessing functions on generic mdst dataset. Make corrections when necessary.
3. Make ipynb under `/afs/desy.de/user/b/boyangyu/HyperTagging/notebooks/` to compare computed four-momentum (from final state particles) and MC four-momentum by visualising the distributions of:
   3.1  each component (E, px, py, pz, invariant mass) for all the samples
   3.2  the event-by-event total difference of each component
   3.2  the particle-by-particle difference of each component

project path: `/afs/desy.de/user/b/boyangyu/HyperTagging/`
necessary documentation: `https://software.belle2.org/`
preprocessing script `/afs/desy.de/user/b/boyangyu/HyperTagging/scripts/preprocess_mdst.py`
raw mdst files: `/pnfs/desy.de/belle/local/belle/MC/release-08-03-00/DB00003335/MC16ri_run2/**/*.root`
path to place virtual environment: `/data/dust/user/boyangyu/uv_env/`
path to place output files: `/data/dust/user/boyangyu/hypertagging`