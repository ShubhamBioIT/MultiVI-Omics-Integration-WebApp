# 🔬 MultiVI: Single-Cell Multi-Omics Integration & Visualization Tool

**A Streamlit-based interactive web app for visualizing and analyzing integrated single-cell RNA and ATAC sequencing data using MultiVI (multi-modal Variational Inference).**

---

## 🌍 Overview

Single-cell sequencing allows scientists to study gene expression and chromatin accessibility at the level of individual cells. However, handling multi-omics datasets (e.g., scRNA + scATAC) requires advanced computational methods.

This project uses **Scanpy**, **scvi-tools**, and **Streamlit** to create a simple yet powerful web app for exploring single-cell multi-omics datasets.  
It is inspired by the research focus of the *Machine Learning for Integrative Genomics (G5) Group* at the **Institut Pasteur, France**.

---

## 🚀 Features

✅ Integrate **scRNA-seq** and **scATAC-seq** datasets using MultiVI  
✅ Perform **dimensionality reduction (UMAP/PCA)**  
✅ **Cluster** cells using Leiden clustering  
✅ **3D Visualization** of gene expression  
✅ Download and upload your own datasets  
✅ Reset and re-run analysis easily  
✅ Preloaded example dataset for quick demo  
✅ Attractive, fast, and responsive Streamlit interface  
✅ Informative “About” and “How to Use” sections

---

## 📁 Repository Structure

📦 MultiVI-SingleCell-App
│
├── app.py # Main Streamlit app
├── MultiVI_SingleCell.ipynb # Jupyter/Colab notebook for analysis
├── requirements.txt # Required dependencies
├── README.md # Documentation (this file)
└── example_data/
├── rna_sample.h5ad
└── atac_sample.h5ad

yaml
Copy code

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/MultiVI-SingleCell-App.git
cd MultiVI-SingleCell-App
Step 2: Install dependencies
It’s recommended to use a virtual environment.

bash
Copy code
pip install -r requirements.txt
Step 3: Run Streamlit app
bash
Copy code
streamlit run app.py
📊 Dataset Information
You can use any paired single-cell dataset with RNA and ATAC modalities.
For demo purposes, this project includes a synthetic preloaded dataset (using scvi-tools demo data).

File formats supported:

.h5ad (AnnData format used in Scanpy/scvi-tools)

.csv (cell × gene expression matrices)

🧠 How to Use the App
Open the App:
Run the app in your browser at http://localhost:8501 after launching with Streamlit.

Upload Data:
You can upload your own .h5ad files or use the preloaded dataset.

Run Integration:
The app automatically preprocesses and integrates your data using MultiVI.

Visualize Results:

Explore 3D UMAP plots of integrated data

View cluster maps

Check marker genes for each cluster

Download Results:
After analysis, download integrated results for further offline analysis.

Reset Analysis:
Use the Reset button to start a new analysis session.

🎨 Streamlit Interface Sections
Section	Description
🧬 Upload / Load Data	Upload or auto-load example dataset
⚙️ Integration	MultiVI model training and latent representation
🧩 Visualization	2D & 3D plots for cluster exploration
📈 Top Genes	3D marker gene expression plots
🔄 Reset Analysis	Start fresh without refreshing the browser
ℹ️ About / Help	Learn how to use the tool effectively

🧩 Technologies Used
Python 3.10+

Streamlit

scvi-tools

Scanpy

Plotly

Pandas / NumPy / Matplotlib

AnnData

⚠️ Troubleshooting
Issue	Cause	Solution
⚙️ Model training slow	MultiVI model is complex	Use GPU or smaller dataset
📂 Uploaded file not visible	Wrong file format	Ensure .h5ad or .csv
🧪 Marker gene analysis error	Cluster too small	Filter out clusters with <3 cells
⚠️ Streamlit figure warning	Missing fig argument	Ensure st.pyplot(fig) syntax

🌟 Future Improvements
Add multi-dataset comparison support

Include real scRNA + scATAC datasets

Implement batch correction & cell-type annotation

Add interactive filtering panel

👩‍🔬 Acknowledgments
This project is inspired by:

Machine Learning for Integrative Genomics (G5) – Institut Pasteur, France

Dr. Gabriele Scalia and Cantini Lab

scvi-tools development team

Developed by Shubham Mahindrakar
M.Sc. Bioinformatics | Research Enthusiast in AI, ML, and Genomics

📚 Citation
If you use this repository or code, please cite:

Mahindrakar, S. (2025). MultiVI: Streamlit-based Tool for Single-Cell Multi-Omics Integration and Visualization. GitHub Repository.
