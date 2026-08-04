# ERF-GS
### [[Project]]() [[Paper]]()

> [**ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints**](),            
> [Xiaoyang Bai](https://andrewbxy.github.io/)\*, [Zhenyang Li](https://lagrangeli.github.io/)\*, [Weiwei Xu](http://www.cad.zju.edu.cn/home/weiweixu/weiweixu_en.htm), [Edmund Y. Lam](https://eee.hku.hk/~elam/), [Yifan Peng](https://www.eee.hku.hk/~evanpeng/)  
> **in production for CVMJ**

**Official code repository of "ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints" (in production for CVMJ).**

## Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>


## Get started
### Installation
Follow the instruction below to set up the environment.

```bash
# clone the repo
git clone -b main --single-branch https://github.com/andrewbxy/ERF-GS.git
# create conda environment
conda create -n erfgs python=3.11
conda activate erfgs

conda install cudatoolkit=12.4 -c pytorch
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# install submodules
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
# install other denpendencies
pip install -r requirements.txt
```

### Dataset
We have uploaded our parsed [Neu3D](https://huggingface.co/datasets/andrewbxy/ERFGS-Neu3D) and [Nvidia](https://huggingface.co/datasets/andrewbxy/ERFGS-Nvidia) datasets, as well as the corresponding simulated events, to Huggingface. You should change `dataset_paths` in `data/multiview.py` to the actual path of downloaded datasets before using them.

The downloaded datasets should have the following structure:
```
Neu3D
|--coffee_martini
    |-- cam00
    |-- cam01
    |-- cam02
    |-- ...
    |-- events
    |-- colmap/sparse/0 (or /sparse_ for Nvidia)
      |-- cameras.bin
      |-- images.bin
      |-- points3D.bin
    |-- dynamic_masks
    |    |-- cam00
    |    |-- cam01
    |    |-- cam02
    |    |-- ...
    |-- points3D_multipleview.ply
    |-- poses_bounds_multipleview.npy
|
|--cook_spinach
|
|--cut_roasted_beef
|-- ...
```

To recreate the **-mb**, **-ts** and **-dv** version of Neu3D and Nvidia-long, run the following command:
```python
python scripts/process_motion_blur.py \
  --scene_root "${path}" \
  --subsample "${subsample_factor}" \
  --scene_name "${scene_name}"
```

When parsing your custom dataset, remember to store the event stream corresponding to each RGB frame in a separate `.h5` file (named `cam${cam_idx}_${img_idx}.h5` under `events`) so that they can be processed by the codebase into more efficient (but more memory expensive) `.bin` format for train-time loading.

## Training and Evaluation
After preparing corresponding data, you can train ERF-GS by running
```python
python train.py \
    --expname "${exp_name}" \
    --config_file "${config_name}"
```
Use YAML files in `config` marked with **baseline** to run the 4DGS baseline, and those marked with **ours** to run ERF-GS. All output files will be placed under `./output/${exp_name}`. Optionally, use the argument `--checkpoint_file "${ckpt_path}"` to load from an intermediate checkpoint.

### Evaluation
You can perform metric evaluation on a trained ERF-GS checkpoint by running: 
```python
python metrics.py \
    --expname "${exp_name}" \
    --test_iteration "${test_iteration}" \
    --config_file "${config_name}"
```
By default, this script will load from the checkpoint `./output/${exp_name}/chkpnt_fine_${test_teration}.pth`.

## Acknowledgement

This codebase is based on [4DGS](https://github.com/hustvl/4DGaussians) by Guanjun Wu et al., which is itself based on the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) implementation.

## 📜 BibTeX
```bibtex
@article{bai2026erfgs,
  title={ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints},
  author={Bai, Xiaoyang and Li, Zhenyang and Xu, Weiwei and Lam, Edmund Y. and Peng, Yifan},
  journal={in production for CVMJ},
  year={2026}
}
```