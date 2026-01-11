# UTMorph

Official Implementation of **UTMorph: A Hybrid CNN-Transformer Network for Weakly-supervised Multimodal Image Registration in Biopsy Puncture**

A hybrid CNN-Transformer network for multimodal image registration between MR and US images.

## Installation

```bash
git clone https://github.com/Prps7/UTMorph.git
cd UTMorph
pip install -r requirements.txt
```

## Dataset Structure

Organize your dataset as follows:

```
datapath/
├── train/
│   ├── mr_images/
│   │   ├── case001/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── case002/
│   │   └── ...
│   ├── mr_labels/
│   │   ├── case001/
│   │   │   ├── label1.jpg
│   │   │   ├── label2.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── us_images/
│   │   ├── case001/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── us_labels/
│       ├── case001/
│       │   ├── label1.jpg
│       │   └── ...
│       └── ...
└── val/
    ├── mr_images/
    ├── mr_labels/
    ├── us_images/
    └── us_labels/
```

Each case folder contains multiple JPG image files. Case names should match across all four directories (mr_images, mr_labels, us_images, us_labels).

## Usage

### Training

```bash
python train.py --datapath /path/to/dataset --mode MRtoUS
```

Main arguments:
- `--datapath`: Dataset directory path
- `--mode`: `MRtoUS` or `UStoMR` (default: `MRtoUS`)
- `--batchsize`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 300)
- `--lr`: Learning rate (default: 1e-3)
- `--output_dir`: Checkpoint save directory (default: `./output`)

### Inference

```bash
python predict.py \
    --datapath /path/to/dataset \
    --checkpoints ./output/MRtoUS.pth \
    --mode MRtoUS \
    --save True
```

Main arguments:
- `--datapath`: Validation dataset path
- `--checkpoints`: Model checkpoint path
- `--mode`: `MRtoUS` or `UStoMR`
- `--save`: Save visualization results (default: False)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{utmorph,
  title={UTMorph: A Hybrid CNN-Transformer Network for Weakly-supervised Multimodal Image Registration in Biopsy Puncture},
  author={Xudong Guo, Peiyu Chen, Haifeng Wang, Zhichao Yan, Qinfen Jiang, Rongjiang Wang, Ji Bin
},
  journal={Medical Image Analysis},
  year={2026},
  note={Accepted}
}
```
