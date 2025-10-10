<div align="center">

# 🔬 MultiVI: Single-Cell Multi-Omics Integration Platform

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![scvi-tools](https://img.shields.io/badge/scvi--tools-latest-purple.svg)](https://scvi-tools.org/)

**A powerful, interactive web application for integrating and visualizing single-cell RNA and ATAC sequencing data using state-of-the-art MultiVI (Multi-modal Variational Inference) methodology.**

[Features](#-features) • [Installation](#️-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Citation](#-citation)

<img src="https://img.shields.io/badge/Bioinformatics-Genomics-teal" /> <img src="https://img.shields.io/badge/Machine_Learning-Deep_Learning-orange" />

---

</div>

## 🌟 Overview

Single-cell sequencing technologies have revolutionized our understanding of cellular heterogeneity by enabling the measurement of multiple molecular modalities at single-cell resolution. However, integrating and analyzing multi-omics datasets (scRNA-seq + scATAC-seq) presents significant computational challenges.

**MultiVI** addresses these challenges by providing an intuitive, browser-based interface built on cutting-edge machine learning frameworks. This tool leverages **variational autoencoders** to create unified representations of multi-modal single-cell data, enabling researchers to:

- 🔗 **Integrate** RNA expression and chromatin accessibility data seamlessly
- 🎯 **Discover** cell populations and their regulatory landscapes
- 🔍 **Explore** gene-peak relationships interactively
- 📊 **Visualize** high-dimensional data in intuitive 2D/3D spaces

> **Inspired by:** The Machine Learning for Integrative Genomics (G5) Group at **Institut Pasteur, France**, this project brings advanced computational genomics tools to a wider research community.

---

## ✨ Features

### Core Functionality
- 🧬 **Multi-Modal Integration**: Seamlessly integrate scRNA-seq and scATAC-seq datasets using the MultiVI framework
- 📉 **Dimensionality Reduction**: Advanced UMAP and PCA implementations for data visualization
- 🎨 **Leiden Clustering**: Automatic cell population identification with adjustable resolution
- 🧪 **Differential Analysis**: Identify marker genes and accessible regions for each cluster
- 📈 **Interactive 3D Visualization**: Explore gene expression landscapes with Plotly-powered 3D plots

### User Experience
- ⚡ **Fast & Responsive**: Built with Streamlit for smooth, real-time interactions
- 📂 **Flexible Data Input**: Support for `.h5ad` (AnnData) and `.csv` formats
- 💾 **Export Results**: Download integrated datasets and analysis results
- 🔄 **Session Management**: Reset and re-run analyses without browser refresh
- 🎓 **Educational Resources**: Comprehensive tutorials and usage guides built-in
- 🎯 **Demo Dataset**: Pre-loaded example data for immediate exploration

### Technical Highlights
- 🤖 **GPU Acceleration**: Optional GPU support for faster model training
- 🔧 **Customizable Parameters**: Fine-tune integration and clustering parameters
- 📊 **Quality Control**: Built-in QC metrics and filtering options
- 🎨 **Publication-Ready Figures**: High-quality visualizations ready for manuscripts

---

## 📁 Repository Structure

```
MultiVI-SingleCell-App/
│
├── 📄 app.py                          # Main Streamlit application
├── 📓 MultiVI_SingleCell.ipynb        # Jupyter notebook for offline analysis
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # This file
├── 📜 LICENSE                         # License information
│
├── 📂 example_data/                   # Sample datasets
│   ├── rna_sample.h5ad               # Example RNA-seq data
│   └── atac_sample.h5ad              # Example ATAC-seq data
│
├── 📂 utils/                          # Utility functions (optional)
│   ├── preprocessing.py              # Data preprocessing helpers
│   └── visualization.py              # Plotting utilities
│
└── 📂 docs/                           # Additional documentation
    ├── USER_GUIDE.md                 # Detailed user guide
    └── API_REFERENCE.md              # Function documentation
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for accelerated training

### Option 1: Quick Install

```bash
# Clone the repository
git clone https://github.com/<your-username>/MultiVI-SingleCell-App.git
cd MultiVI-SingleCell-App

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

### Option 2: Using Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv multivi_env

# Activate environment
# On Windows:
multivi_env\Scripts\activate
# On macOS/Linux:
source multivi_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Option 3: Using Conda

```bash
# Create conda environment
conda create -n multivi python=3.10
conda activate multivi

# Install dependencies
pip install -r requirements.txt

# Launch app
streamlit run app.py
```

---

## 🚀 Quick Start

### 1. **Launch the Application**
After installation, the app will open automatically in your default browser at `http://localhost:8501`

### 2. **Load Your Data**
- **Option A**: Upload your own `.h5ad` files (RNA + ATAC)
- **Option B**: Use the pre-loaded demo dataset to explore features

### 3. **Configure Integration**
- Adjust model parameters (latent dimensions, training epochs)
- Set preprocessing thresholds (min cells, min genes)

### 4. **Run Analysis**
Click **"Run Integration"** to:
- Train the MultiVI model
- Generate latent representations
- Compute UMAP embeddings
- Perform Leiden clustering

### 5. **Explore Results**
- Navigate through interactive 2D/3D visualizations
- Examine cluster-specific marker genes
- Analyze gene-peak correlations
- Download processed data for downstream analysis

### 6. **Export & Share**
- Save integrated `.h5ad` files
- Export high-resolution figures
- Generate analysis reports

---

## 📊 Supported Data Formats

### Input Formats

| Format | Description | Requirements |
|--------|-------------|--------------|
| `.h5ad` | AnnData format (recommended) | RNA and ATAC modalities with matching cell IDs |
| `.csv` | Cell × Gene/Peak matrices | Separate files for RNA and ATAC with cell barcodes |
| `.h5` | HDF5 format | Scanpy-compatible structure |

### Dataset Requirements
- **Matched cells**: Same cell barcodes in both RNA and ATAC datasets
- **Minimum cells**: ≥ 100 cells recommended
- **Minimum features**: ≥ 500 genes (RNA), ≥ 1000 peaks (ATAC)
- **Format consistency**: Properly annotated observation and variable names

---

## 🎨 Interface Overview

### Main Sections

#### 🧬 Data Management
- Upload custom datasets or load examples
- Preview data dimensions and cell counts
- Quality control visualization

#### ⚙️ Model Configuration
- **Latent Dimensions**: Control representation complexity (default: 20)
- **Training Epochs**: Balance speed vs. accuracy (default: 400)
- **Learning Rate**: Fine-tune optimization (default: 0.001)
- **Batch Size**: Adjust for memory constraints (default: 128)

#### 🧩 Integration & Analysis
- Real-time training progress monitoring
- ELBO (Evidence Lower Bound) convergence plots
- Latent space quality metrics

#### 📈 Visualization Suite
- **2D UMAP**: Color by clusters, genes, or metadata
- **3D UMAP**: Interactive rotation and zoom
- **Gene Expression Heatmaps**: Cluster-wise patterns
- **Accessibility Profiles**: Peak accessibility across clusters
- **Marker Gene Volcano Plots**: Statistical significance visualization

#### 💾 Results Export
- Download integrated AnnData object
- Save publication-ready figures (PNG, SVG, PDF)
- Export cluster annotations and marker tables

---

## 🧠 Technical Details

### MultiVI Model Architecture
MultiVI employs a **conditional variational autoencoder** framework:
- **Encoder**: Maps RNA + ATAC data to shared latent space
- **Latent Space**: Low-dimensional representation capturing cellular identity
- **Decoders**: Reconstruct RNA counts and ATAC peaks independently
- **Integration**: Aligns modalities through shared latent variables

### Computational Pipeline
```
Raw Data → QC Filtering → Normalization → Model Training → 
Latent Representation → UMAP/Clustering → Marker Discovery → Visualization
```

### Key Dependencies
- **scvi-tools**: Probabilistic modeling framework
- **Scanpy**: Single-cell analysis toolkit
- **PyTorch**: Deep learning backend
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization
- **AnnData**: Data structure for single-cell data

---

## 🔧 Advanced Usage

### Customizing Preprocessing
```python
# Example: Adjust filtering thresholds in your notebook
import scanpy as sc

# Stricter quality control
sc.pp.filter_cells(adata_rna, min_genes=500)
sc.pp.filter_genes(adata_rna, min_cells=10)
```

### Batch Effect Correction
MultiVI inherently handles batch effects through its probabilistic framework. For explicit batch correction:
- Include batch information in `adata.obs['batch']`
- Model automatically conditions on batch during training

### Cell Type Annotation
After clustering, annotate cell types using:
- Manual marker gene inspection
- Integration with reference atlases (e.g., via `celltypist`)
- Automated annotation pipelines

---

## ⚠️ Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| 🐌 Slow model training | Large dataset or CPU-only | Use GPU acceleration or subsample data |
| 📉 Poor integration | Mismatched cell IDs | Ensure identical cell barcodes in both modalities |
| 🚫 Upload fails | Incorrect file format | Convert to `.h5ad` using Scanpy |
| 🔴 Memory error | Dataset too large | Increase system RAM or reduce features |
| ⚡ CUDA out of memory | Batch size too large | Reduce batch size in settings |
| 📊 Empty clusters | Over-clustering | Decrease Leiden resolution parameter |
| 🎨 Visualization lag | Too many cells | Subsample cells for plotting (preserves analysis) |

### Debug Mode
Enable detailed logging by running:
```bash
streamlit run app.py --logger.level=debug
```

---

## 📚 Example Workflow

### Research Scenario: Brain Cell Atlas
```python
# 1. Load multimodal data
rna = sc.read_h5ad("brain_rna.h5ad")
atac = sc.read_h5ad("brain_atac.h5ad")

# 2. Upload to MultiVI app and configure:
#    - Latent dims: 30 (complex tissue)
#    - Epochs: 500 (high accuracy)
#    - Resolution: 0.8 (moderate clustering)

# 3. Post-integration analysis
#    - Identify neuron subtypes via markers
#    - Analyze neuron-specific accessible regions
#    - Export for trajectory analysis
```

---

## 🌟 Future Roadmap

- [ ] **Multi-Dataset Integration**: Harmonize multiple experiments simultaneously
- [ ] **Real scRNA + scATAC Datasets**: Curated public datasets (10x Multiome, SHARE-seq)
- [ ] **Automated Cell Type Annotation**: Integration with reference databases
- [ ] **Trajectory Inference**: Pseudotime analysis for developmental studies
- [ ] **Gene Regulatory Networks**: Infer transcription factor-target relationships
- [ ] **Batch Effect Visualization**: Diagnostic plots for data quality
- [ ] **Collaborative Features**: Share sessions and annotations
- [ ] **API Access**: Programmatic integration for pipelines
- [ ] **Docker Container**: Simplified deployment
- [ ] **Cloud Deployment**: Run without local installation

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Propose enhancements via GitHub Discussions
3. **Submit Pull Requests**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/AmazingFeature`)
   - Commit changes (`git commit -m 'Add AmazingFeature'`)
   - Push to branch (`git push origin feature/AmazingFeature`)
   - Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 app.py
black app.py --check
```

---

## 👩‍🔬 Acknowledgments

This project draws inspiration and builds upon the groundbreaking work of:

- **[Machine Learning for Integrative Genomics (G5)](https://research.pasteur.fr/en/team/machine-learning-for-integrative-genomics/)** – Institut Pasteur, France
- **Dr. Gabriele Scalia** and the **Cantini Lab** for methodological insights
- **[scvi-tools Development Team](https://scvi-tools.org/)** for the robust probabilistic framework
- **[Scanpy Developers](https://scanpy.readthedocs.io/)** for single-cell analysis foundations
- The **open-source community** for continuous innovation in computational biology

### Key Publications
- MultiVI: *Ashuach et al. (2021). "MultiVI: deep generative model for the integration of multi-modal data." Nature Methods.*
- UMAP: *McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection."*
- Leiden: *Traag et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities."*

---

## 📖 Citation

If this tool contributes to your research, please cite:

```bibtex
@software{mahindrakar2025multivi,
  author       = {Mahindrakar, Shubham},
  title        = {MultiVI: Streamlit-based Tool for Single-Cell Multi-Omics Integration and Visualization},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/<your-username>/MultiVI-SingleCell-App},
  note         = {M.Sc. Bioinformatics | AI/ML for Genomics}
}
```

Additionally, please cite the original MultiVI publication:
```bibtex
@article{ashuach2021multivi,
  title={MultiVI: deep generative model for the integration of multi-modal data},
  author={Ashuach, Tal and Reidenbach, Daniel A and Gayoso, Adam and Yosef, Nir},
  journal={Nature Methods},
  year={2021},
  publisher={Nature Publishing Group}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Shubham Mahindrakar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

---

## 📞 Contact & Support

**Developer**: Shubham Mahindrakar  
**Degree**: M.Sc. Bioinformatics  
**Focus**: AI/ML Applications in Genomics  

- 📧 **Email**: [your.email@example.com](mailto:your.email@example.com)
- 🐙 **GitHub**: [@your-username](https://github.com/your-username)
- 💼 **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/your-profile)
- 🌐 **Portfolio**: [your-website.com](https://your-website.com)

### Get Help
- 📖 **Documentation**: Check our [Wiki](https://github.com/your-username/MultiVI-SingleCell-App/wiki)
- 💬 **Discussions**: Join [GitHub Discussions](https://github.com/your-username/MultiVI-SingleCell-App/discussions)
- 🐛 **Issues**: Report bugs via [Issue Tracker](https://github.com/your-username/MultiVI-SingleCell-App/issues)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ for the single-cell genomics community**

[⬆ Back to Top](#-multivi-single-cell-multi-omics-integration-platform)

</div>
