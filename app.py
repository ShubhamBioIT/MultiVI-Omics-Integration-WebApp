# app.py
# Enhanced MultiVI Integrator with Modern UI
# Multi-Omics Integration & Visualization Tool
# Author: Shubham Mahindrakar

import streamlit as st
import os
import tempfile
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Lazy imports for heavy dependencies
HAS_SCVI = False
try:
    import muon as mu
    import scanpy as sc
    import scvi
    import torch
    HAS_SCVI = True
except Exception as e:
    HAS_SCVI = False
    print(f"Warning: scvi-tools/muon not available: {e}")

# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    page_title="MultiVI Omics Integrator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        margin-bottom: 2rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Warning/Error cards */
    .success-box {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c715, #fbbf2415);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fee2e215, #ef444415);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea15 0%, #764ba215 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 2px solid #e5e7eb;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Download link */
    .download-link {
        display: inline-block;
        padding: 0.8rem 1.5rem;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    
    .download-link:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def save_uploaded_file(uploaded_file, suffix=".h5mu"):
    """Save uploaded Streamlit file to temporary path."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        raise RuntimeError(f"Failed to save uploaded file: {e}")

def safe_remove(path):
    """Safely remove a file."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def reset_session():
    """Reset all session state and clean up temp files."""
    # Remove temp files
    path = st.session_state.get("mdata_path", None)
    if path:
        safe_remove(path)
    
    # Clear session state
    keys_to_clear = [
        "mdata_path", "mdata", "model", "latent", 
        "embedding", "clusters", "latent_concat", 
        "dataset_loaded", "integration_done"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def get_obs_names(mdata):
    """Robust method to get observation names from MuData."""
    try:
        names = mdata.obs_names
        if names is not None and len(names) > 0:
            return names
    except Exception:
        pass
    
    mods = list(mdata.mod.keys())
    if mods:
        return mdata.mod[mods[0]].obs_names
    return []

def display_metrics(n_cells, n_rna_genes, n_atac_peaks):
    """Display dataset metrics in cards."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{n_cells:,}</p>
            <p class="metric-label">Cells</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{n_rna_genes:,}</p>
            <p class="metric-label">RNA Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{n_atac_peaks:,}</p>
            <p class="metric-label">ATAC Peaks</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APP ====================

# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 MultiVI Multi-Omics Integrator</h1>
    <p>Seamlessly integrate scRNA-seq and scATAC-seq data with advanced visualization</p>
</div>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("---")
    st.markdown("#### 🔧 Integration Mode")
    mode = st.radio(
        "Select mode:",
        ["🚀 Fast Demo (Quick)", "🎯 MultiVI (Full Model)"],
        help="Fast Demo uses PCA for quick results. MultiVI uses deep learning for accurate integration."
    )
    
    st.markdown("---")
    st.markdown("#### 📊 Dataset Options")
    downsample = st.checkbox("Enable downsampling", value=True, help="Reduce dataset size for faster processing")
    if downsample:
        downsample_n = st.slider("Max cells to keep:", 100, 5000, 1000, step=100)
    else:
        downsample_n = None
    
    st.markdown("---")
    st.markdown("#### 🎓 Model Parameters")
    if "MultiVI" in mode:
        max_epochs = st.slider("Training epochs:", 10, 200, 50, step=10)
        latent_dims = st.slider("Latent dimensions:", 10, 50, 20, step=5)
    else:
        quick_pcs = st.slider("PCA components:", 5, 50, 20, step=5)
        max_epochs = 50
        latent_dims = 20
    
    st.markdown("---")
    st.markdown("#### 📈 Clustering")
    n_clusters = st.slider("Number of clusters:", 3, 15, 8, step=1)
    
    st.markdown("---")
    if st.button("🔄 Reset Application", use_container_width=True):
        reset_session()
        st.success("✅ Application reset successfully!")
        st.rerun()
    
    st.markdown("---")
    st.markdown("#### ℹ️ System Info")
    st.info(f"""
    **MultiVI Status:** {'✅ Available' if HAS_SCVI else '❌ Not Available'}
    
    **Python Packages:**
    - Scanpy: {'✅' if HAS_SCVI else '❌'}
    - Muon: {'✅' if HAS_SCVI else '❌'}
    - PyTorch: {'✅' if HAS_SCVI and torch else '❌'}
    """)

# ==================== MAIN CONTENT ====================

# Step 1: Dataset Loading
st.markdown('<p class="section-header">📂 Step 1: Dataset Management</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📥 Download Demo", "📤 Upload Data", "ℹ️ Dataset Info"])

with tab1:
    st.markdown("""
    <div class="info-card">
        <h3>🧪 Demo PBMC Multiome Dataset</h3>
        <p>This dataset contains paired scRNA-seq and scATAC-seq data from peripheral blood mononuclear cells (PBMCs).</p>
        <p><strong>Size:</strong> ~50MB | <strong>Cells:</strong> ~10,000 | <strong>Format:</strong> .h5mu</p>
    </div>
    """, unsafe_allow_html=True)
    
    demo_url = "https://figshare.com/ndownloader/files/54794234"
    st.markdown(f"""
    <a href="{demo_url}" class="download-link" target="_blank">
        ⬇️ Download Demo Dataset
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 After downloading, switch to the **'Upload Data'** tab to load the file into the app.")

with tab2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose your .h5mu file",
        type=["h5mu"],
        help="Upload a MuData file containing paired scRNA-seq and scATAC-seq data"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("💾 Saving uploaded file..."):
                saved_path = save_uploaded_file(uploaded_file, suffix=".h5mu")
                st.session_state["mdata_path"] = saved_path
                st.session_state["dataset_loaded"] = False
            
            st.markdown(f"""
            <div class="success-box">
                <strong>✅ File uploaded successfully!</strong><br>
                Path: <code>{saved_path}</code><br>
                Size: {uploaded_file.size / 1024 / 1024:.2f} MB
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                <strong>❌ Upload failed:</strong> {str(e)}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="info-card">
        <h4>📋 Supported Format</h4>
        <p><strong>.h5mu</strong> - MuData format for multi-modal single-cell data</p>
        
        <h4>🔍 Requirements</h4>
        <ul>
            <li>Paired RNA and ATAC modalities</li>
            <li>Matching cell barcodes across modalities</li>
            <li>Minimum 100 cells recommended</li>
            <li>Pre-processed and QC-filtered data</li>
        </ul>
        
        <h4>💾 Expected Structure</h4>
        <pre>
MuData object with modalities:
  - 'rna' or 'rna_subset': Gene expression matrix
  - 'atac' or 'atac_subset': Peak accessibility matrix
        </pre>
    </div>
    """, unsafe_allow_html=True)

# Load Dataset Button
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    load_button = st.button("🔍 Load & Inspect Dataset", use_container_width=True, type="primary")

if load_button:
    if "mdata_path" not in st.session_state or not st.session_state["mdata_path"]:
        st.error("❌ No file available. Please upload or download a dataset first.")
    else:
        if not HAS_SCVI:
            st.warning("⚠️ Muon/Scanpy not installed. Cannot load .h5mu files. Please install required packages.")
        else:
            try:
                with st.spinner("🔄 Loading dataset..."):
                    path = st.session_state["mdata_path"]
                    mdata = mu.read_h5mu(path)
                    
                    # Update MuData
                    try:
                        mdata.update()
                    except Exception:
                        pass
                    
                    # Standardize modality names
                    if "rna_subset" not in mdata.mod and "rna" in mdata.mod:
                        mdata.mod["rna_subset"] = mdata.mod["rna"]
                    if "atac_subset" not in mdata.mod and "atac" in mdata.mod:
                        mdata.mod["atac_subset"] = mdata.mod["atac"]
                    
                    # Downsample if requested
                    if downsample and mdata.n_obs > 0:
                        names = get_obs_names(mdata)
                        if len(names) > 0 and downsample_n and downsample_n < mdata.n_obs:
                            sel = names[:downsample_n]
                            mdata = mdata[sel, :]
                    
                    st.session_state["mdata"] = mdata
                    st.session_state["dataset_loaded"] = True
                
                # Display success and metrics
                mdata = st.session_state["mdata"]
                n_cells = mdata.n_obs
                n_rna = mdata.mod['rna_subset'].var.shape[0] if 'rna_subset' in mdata.mod else 0
                n_atac = mdata.mod['atac_subset'].var.shape[0] if 'atac_subset' in mdata.mod else 0
                
                st.success("✅ Dataset loaded successfully!")
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Display metrics
                display_metrics(n_cells, n_rna, n_atac)
                
                # Display observation table
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📊 Cell Metadata Preview")
                
                obs_df = None
                try:
                    if isinstance(mdata.obs, pd.DataFrame) and mdata.obs.shape[0] > 0:
                        obs_df = mdata.obs.copy()
                except Exception:
                    pass
                
                if obs_df is None or obs_df.shape[0] == 0:
                    if "rna_subset" in mdata.mod:
                        obs_df = mdata.mod["rna_subset"].obs.copy()
                    else:
                        mods = list(mdata.mod.keys())
                        obs_df = mdata.mod[mods[0]].obs.copy() if mods else pd.DataFrame()
                
                if obs_df.shape[0] > 0:
                    st.dataframe(obs_df.head(20), use_container_width=True)
                else:
                    st.warning("⚠️ No cell metadata found.")
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <strong>❌ Failed to load dataset:</strong><br>
                    {str(e)}
                </div>
                """, unsafe_allow_html=True)
                st.code(traceback.format_exc())

# Display current dataset status
if st.session_state.get("dataset_loaded", False):
    st.markdown("""
    <div class="success-box">
        ✅ <strong>Dataset ready for integration!</strong> Proceed to Step 2 below.
    </div>
    """, unsafe_allow_html=True)

# ==================== STEP 2: INTEGRATION ====================
st.markdown('<p class="section-header">🔬 Step 2: Multi-Omics Integration</p>', unsafe_allow_html=True)

if not st.session_state.get("dataset_loaded", False):
    st.info("ℹ️ Please load a dataset first (Step 1) before running integration.")
else:
    integration_tabs = st.tabs(["⚡ Quick Integration", "🎯 Advanced Settings", "📊 Results"])
    
    with integration_tabs[0]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        
        if "Fast Demo" in mode:
            st.markdown("""
            ### 🚀 Fast Demo Integration
            Uses PCA-based dimensionality reduction for rapid exploration.
            **Advantages:** Very fast, no training required
            """)
            
            if st.button("▶️ Run Fast Integration", use_container_width=True, type="primary"):
                try:
                    with st.spinner("🔄 Running fast integration..."):
                        progress_bar = st.progress(0)
                        
                        mdata = st.session_state["mdata"]
                        progress_bar.progress(20)
                        
                        # Extract matrices
                        try:
                            rna = mdata.mod["rna_subset"].X
                            atac = mdata.mod["atac_subset"].X
                        except Exception:
                            mods = list(mdata.mod.keys())
                            rna = mdata.mod[mods[0]].X
                            atac = mdata.mod[mods[1]].X if len(mods) > 1 else mdata.mod[mods[0]].X
                        
                        progress_bar.progress(40)
                        
                        # Convert to dense
                        def to_dense(x):
                            if hasattr(x, "toarray"):
                                return x.toarray()
                            return np.array(x)
                        
                        rna_dense = to_dense(rna)
                        atac_dense = to_dense(atac)
                        
                        progress_bar.progress(60)
                        
                        # PCA on both modalities
                        from sklearn.decomposition import TruncatedSVD
                        
                        n_pcs = quick_pcs if 'quick_pcs' in locals() else 20
                        pca = PCA(n_components=min(n_pcs, rna_dense.shape[1]-1, rna_dense.shape[0]-1))
                        rna_pc = pca.fit_transform(rna_dense)
                        
                        svd = TruncatedSVD(n_components=min(n_pcs, atac_dense.shape[1]-1, atac_dense.shape[0]-1))
                        atac_pc = svd.fit_transform(atac_dense)
                        
                        progress_bar.progress(80)
                        
                        # Concatenate and reduce to 2D
                        concat = np.concatenate([rna_pc, atac_pc], axis=1)
                        emb2d = PCA(n_components=2).fit_transform(concat)
                        
                        # Clustering
                        labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(concat)
                        
                        # Save to session
                        st.session_state["embedding"] = emb2d
                        st.session_state["latent_concat"] = concat
                        st.session_state["clusters"] = labels.astype(str)
                        st.session_state["integration_done"] = True
                        
                        progress_bar.progress(100)
                    
                    st.success("✅ Fast integration completed successfully!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Integration failed: {str(e)}")
                    st.code(traceback.format_exc())
        
        else:  # MultiVI mode
            st.markdown("""
            ### 🎯 MultiVI Integration
            Deep learning-based multi-modal variational inference for accurate integration.
            **Advantages:** High quality, handles batch effects, biologically meaningful
            """)
            
            if not HAS_SCVI:
                st.error("❌ scvi-tools and muon must be installed to run MultiVI.")
            else:
                if st.button("▶️ Train MultiVI Model", use_container_width=True, type="primary"):
                    try:
                        with st.spinner("🧠 Training MultiVI model... This may take a few minutes."):
                            progress_bar = st.progress(0)
                            
                            mdata = st.session_state["mdata"]
                            progress_bar.progress(10)
                            
                            # Setup MultiVI
                            try:
                                scvi.model.MULTIVI.setup_mudata(
                                    mdata,
                                    modalities={"rna_layer": "rna_subset", "atac_layer": "atac_subset"}
                                )
                            except Exception:
                                mods = list(mdata.mod.keys())
                                rna_key = "rna_subset" if "rna_subset" in mdata.mod else mods[0]
                                atac_key = "atac_subset" if "atac_subset" in mdata.mod else (mods[1] if len(mods) > 1 else mods[0])
                                scvi.model.MULTIVI.setup_mudata(
                                    mdata,
                                    modalities={"rna_layer": rna_key, "atac_layer": atac_key}
                                )
                            
                            progress_bar.progress(30)
                            
                            # Create model
                            model = scvi.model.MULTIVI(
                                mdata,
                                n_genes=len(mdata.mod["rna_subset"].var),
                                n_regions=len(mdata.mod["atac_subset"].var),
                                n_latent=latent_dims
                            )
                            
                            progress_bar.progress(50)
                            
                            # Train
                            if torch.cuda.is_available():
                                st.info("🚀 CUDA detected - using GPU acceleration")
                            
                            model.train(max_epochs=max_epochs)
                            progress_bar.progress(90)
                            
                            # Get latent representation
                            latent = model.get_latent_representation()
                            emb2d = PCA(n_components=2).fit_transform(latent)
                            
                            # Clustering
                            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(latent)
                            
                            # Save to session
                            st.session_state["model"] = model
                            st.session_state["latent"] = latent
                            st.session_state["embedding"] = emb2d
                            st.session_state["clusters"] = labels.astype(str)
                            st.session_state["integration_done"] = True
                            
                            progress_bar.progress(100)
                        
                        st.success("✅ MultiVI training completed successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
                        st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with integration_tabs[1]:
        st.markdown("""
        <div class="info-card">
            <h4>⚙️ Current Configuration</h4>
        </div>
        """, unsafe_allow_html=True)
        
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.markdown(f"""
            **Integration Mode:** {mode}  
            **Downsampling:** {'Enabled' if downsample else 'Disabled'}  
            **Max Cells:** {downsample_n if downsample else 'All'}  
            """)
        
        with config_col2:
            st.markdown(f"""
            **Training Epochs:** {max_epochs}  
            **Latent Dimensions:** {latent_dims}  
            **Number of Clusters:** {n_clusters}  
            """)
    
    with integration_tabs[2]:
        if st.session_state.get("integration_done", False):
            st.markdown("""
            <div class="success-box">
                <h4>✅ Integration Complete!</h4>
                <p>Your data has been successfully integrated. Proceed to visualization below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display integration metrics
            if "embedding" in st.session_state:
                emb = st.session_state["embedding"]
                clusters = st.session_state.get("clusters", [])
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Cells Integrated", emb.shape[0])
                with metrics_col2:
                    st.metric("Embedding Dimensions", emb.shape[1])
                with metrics_col3:
                    st.metric("Clusters Identified", len(np.unique(clusters)))
        else:
            st.info("ℹ️ Run integration to see results here.")

# ==================== STEP 3: VISUALIZATION ====================
st.markdown('<p class="section-header">📊 Step 3: Interactive Visualization</p>', unsafe_allow_html=True)

if not st.session_state.get("integration_done", False):
    st.info("ℹ️ Please complete integration (Step 2) before viewing visualizations.")
else:
    viz_tabs = st.tabs([
        "🗺️ 2D Embedding",
        "📊 Cluster Analysis", 
        "🔥 Expression Heatmap",
        "🎨 3D Gene Expression",
        "📈 Quality Metrics"
    ])
    
    # Tab 1: 2D Embedding
    with viz_tabs[0]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Integrated Cell Embedding")
        
        try:
            emb = st.session_state["embedding"]
            clusters = st.session_state["clusters"]
            
            # Create dataframe
            df_plot = pd.DataFrame({
                "UMAP_1": emb[:, 0],
                "UMAP_2": emb[:, 1],
                "Cluster": clusters
            })
            
            # Color options
            color_by = st.radio(
                "Color by:",
                ["Cluster", "Density"],
                horizontal=True
            )
            
            if color_by == "Cluster":
                fig = px.scatter(
                    df_plot,
                    x="UMAP_1",
                    y="UMAP_2",
                    color="Cluster",
                    title="Integrated Multi-Omics Embedding (2D)",
                    width=1000,
                    height=700,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='white')))
            else:
                fig = px.density_contour(
                    df_plot,
                    x="UMAP_1",
                    y="UMAP_2",
                    title="Cell Density Map",
                    width=1000,
                    height=700
                )
                fig.update_traces(contours_coloring="fill", contours_showlabels=True)
            
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(size=12),
                title_font=dict(size=18, color='#667eea')
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("🔍 Each point represents a cell. Colors indicate cluster assignments. Hover for details.")
            
        except Exception as e:
            st.error(f"❌ Visualization failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Cluster Analysis
    with viz_tabs[1]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Cluster Distribution & Statistics")
        
        try:
            clusters = st.session_state["clusters"]
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig_bar = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={"x": "Cluster ID", "y": "Number of Cells"},
                    title="Cells per Cluster",
                    color=cluster_counts.values,
                    color_continuous_scale="Viridis"
                )
                fig_bar.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    showlegend=False,
                    title_font=dict(size=16, color='#667eea')
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title="Cluster Proportions",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(
                    title_font=dict(size=16, color='#667eea')
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Statistics table
            st.markdown("#### 📋 Cluster Statistics")
            stats_df = pd.DataFrame({
                "Cluster": cluster_counts.index,
                "Cell Count": cluster_counts.values,
                "Percentage": (cluster_counts.values / cluster_counts.sum() * 100).round(2)
            })
            stats_df["Percentage"] = stats_df["Percentage"].astype(str) + "%"
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: Expression Heatmap
    with viz_tabs[2]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Mean Expression Heatmap")
        
        if not HAS_SCVI or st.session_state.get("mdata") is None:
            st.warning("⚠️ Heatmap requires muon/scanpy and a loaded dataset.")
        else:
            try:
                mdata = st.session_state["mdata"]
                
                if "rna_subset" not in mdata.mod:
                    st.error("❌ RNA modality 'rna_subset' not found.")
                else:
                    rna = mdata.mod["rna_subset"]
                    rna.obs["cluster"] = st.session_state["clusters"]
                    
                    # Get top variable genes
                    n_top_genes = st.slider("Number of top genes to display:", 10, 100, 30, step=5)
                    
                    with st.spinner("Computing expression patterns..."):
                        df_expr = rna.to_df()
                        
                        # Select top variable genes
                        gene_var = df_expr.var(axis=0).sort_values(ascending=False)
                        top_genes = gene_var.index[:n_top_genes]
                        
                        # Calculate mean expression per cluster
                        mean_expr = df_expr[top_genes].groupby(rna.obs["cluster"]).mean().T
                        
                        # Create heatmap
                        fig_heatmap = px.imshow(
                            mean_expr,
                            labels=dict(x="Cluster", y="Gene", color="Expression"),
                            title=f"Mean Expression of Top {n_top_genes} Variable Genes",
                            color_continuous_scale="RdBu_r",
                            aspect="auto"
                        )
                        fig_heatmap.update_layout(
                            width=1000,
                            height=600,
                            title_font=dict(size=18, color='#667eea')
                        )
                        fig_heatmap.update_xaxes(side="bottom")
                        
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.caption("🔍 Rows = genes, Columns = clusters. Color intensity shows mean expression level.")
                        
                        # Download option
                        csv = mean_expr.to_csv()
                        st.download_button(
                            label="📥 Download Expression Matrix (CSV)",
                            data=csv,
                            file_name="cluster_expression.csv",
                            mime="text/csv"
                        )
                
            except Exception as e:
                st.error(f"❌ Heatmap generation failed: {str(e)}")
                st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 4: 3D Gene Expression
    with viz_tabs[3]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 🎨 3D Interactive Gene Expression")
        
        if not HAS_SCVI or st.session_state.get("mdata") is None:
            st.warning("⚠️ 3D visualization requires muon/scanpy and a loaded dataset.")
        else:
            try:
                mdata = st.session_state["mdata"]
                rna = mdata.mod.get("rna_subset", None)
                
                if rna is None:
                    st.error("❌ RNA modality not found.")
                else:
                    # Ensure clusters are attached
                    if "cluster" not in rna.obs.columns:
                        rna.obs["cluster"] = st.session_state["clusters"]
                    
                    # Method selection
                    method = st.radio(
                        "Select visualization method:",
                        ["🔬 Automatic (Top Markers)", "✏️ Manual (Choose Genes)"],
                        horizontal=True
                    )
                    
                    if method == "🔬 Automatic (Top Markers)":
                        if st.button("🚀 Generate 3D Plot", type="primary"):
                            with st.spinner("Computing differential expression..."):
                                # Filter clusters with sufficient cells
                                cluster_counts = rna.obs["cluster"].value_counts()
                                valid_clusters = cluster_counts[cluster_counts >= 3].index.tolist()
                                
                                if len(valid_clusters) < 1:
                                    st.error("❌ No clusters with ≥3 cells found.")
                                else:
                                    # Filter data
                                    rna_filtered = rna[rna.obs["cluster"].isin(valid_clusters)].copy()
                                    
                                    # Differential expression
                                    sc.tl.rank_genes_groups(
                                        rna_filtered,
                                        groupby="cluster",
                                        method="wilcoxon",
                                        pts=True
                                    )
                                    
                                    # Get top 3 genes from first cluster
                                    names = rna_filtered.uns["rank_genes_groups"]["names"]
                                    first_cluster = list(names.dtype.names)[0]
                                    top_genes = list(names[first_cluster][:3])
                                    
                                    st.info(f"📌 Using top 3 marker genes from cluster {first_cluster}: **{', '.join(top_genes)}**")
                                    
                                    # Get expression
                                    expr_df = rna.to_df()
                                    
                                    # Verify genes exist
                                    missing = [g for g in top_genes if g not in expr_df.columns]
                                    if missing:
                                        st.error(f"❌ Genes not found: {missing}")
                                    else:
                                        # Create 3D plot
                                        fig_3d = px.scatter_3d(
                                            x=expr_df[top_genes[0]].values,
                                            y=expr_df[top_genes[1]].values,
                                            z=expr_df[top_genes[2]].values,
                                            color=st.session_state["clusters"],
                                            labels={
                                                "x": top_genes[0],
                                                "y": top_genes[1],
                                                "z": top_genes[2]
                                            },
                                            title=f"3D Expression: {' × '.join(top_genes)}",
                                            width=1000,
                                            height=800,
                                            color_discrete_sequence=px.colors.qualitative.Set3
                                        )
                                        
                                        fig_3d.update_traces(
                                            marker=dict(size=5, opacity=0.7, line=dict(width=0))
                                        )
                                        fig_3d.update_layout(
                                            scene=dict(
                                                bgcolor='white',
                                                xaxis=dict(backgroundcolor='#f8f9fa', gridcolor='#e9ecef'),
                                                yaxis=dict(backgroundcolor='#f8f9fa', gridcolor='#e9ecef'),
                                                zaxis=dict(backgroundcolor='#f8f9fa', gridcolor='#e9ecef')
                                            ),
                                            title_font=dict(size=18, color='#667eea')
                                        )
                                        
                                        st.plotly_chart(fig_3d, use_container_width=True)
                                        st.caption("🎮 Click and drag to rotate. Scroll to zoom. Each point is a cell colored by cluster.")
                    
                    else:  # Manual gene selection
                        st.markdown("#### ✏️ Choose 3 Genes for Visualization")
                        
                        # Get available genes
                        available_genes = rna.var_names.tolist()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            gene1 = st.selectbox("Gene 1 (X-axis):", available_genes, index=0)
                        with col2:
                            gene2 = st.selectbox("Gene 2 (Y-axis):", available_genes, index=min(1, len(available_genes)-1))
                        with col3:
                            gene3 = st.selectbox("Gene 3 (Z-axis):", available_genes, index=min(2, len(available_genes)-1))
                        
                        if st.button("🎨 Create 3D Plot", type="primary"):
                            try:
                                expr_df = rna.to_df()
                                
                                fig_3d = px.scatter_3d(
                                    x=expr_df[gene1].values,
                                    y=expr_df[gene2].values,
                                    z=expr_df[gene3].values,
                                    color=st.session_state["clusters"],
                                    labels={"x": gene1, "y": gene2, "z": gene3},
                                    title=f"3D Expression: {gene1} × {gene2} × {gene3}",
                                    width=1000,
                                    height=800,
                                    color_discrete_sequence=px.colors.qualitative.Set3
                                )
                                
                                fig_3d.update_traces(marker=dict(size=5, opacity=0.7))
                                fig_3d.update_layout(
                                    scene=dict(bgcolor='white'),
                                    title_font=dict(size=18, color='#667eea')
                                )
                                
                                st.plotly_chart(fig_3d, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"❌ Plot failed: {str(e)}")
            
            except Exception as e:
                st.error(f"❌ 3D visualization failed: {str(e)}")
                st.code(traceback.format_exc())
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 5: Quality Metrics
    with viz_tabs[4]:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Integration Quality Metrics")
        
        try:
            if "latent_concat" in st.session_state or "latent" in st.session_state:
                latent = st.session_state.get("latent", st.session_state.get("latent_concat"))
                
                # Compute metrics
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                
                clusters = st.session_state["clusters"]
                cluster_numeric = pd.Categorical(clusters).codes
                
                silhouette = silhouette_score(latent, cluster_numeric)
                davies_bouldin = davies_bouldin_score(latent, cluster_numeric)
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{silhouette:.3f}</p>
                        <p class="metric-label">Silhouette Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Higher is better (range: -1 to 1)")
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{davies_bouldin:.3f}</p>
                        <p class="metric-label">Davies-Bouldin Index</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Lower is better")
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{latent.shape[1]}</p>
                        <p class="metric-label">Latent Dimensions</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Embedding dimensionality")
                
                # Interpretation
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### 📊 Quality Interpretation")
                
                if silhouette > 0.5:
                    quality = "Excellent"
                    color = "success"
                elif silhouette > 0.3:
                    quality = "Good"
                    color = "info"
                else:
                    quality = "Fair"
                    color = "warning"
                
                st.markdown(f"""
                <div class="{color}-box">
                    <strong>Overall Quality: {quality}</strong><br>
                    Your integration shows {quality.lower()} cluster separation. 
                    Silhouette score of {silhouette:.3f} indicates 
                    {'well-defined' if silhouette > 0.5 else 'moderate' if silhouette > 0.3 else 'weak'} 
                    cluster structure.
                </div>
                """, unsafe_allow_html=True)
                
                # Explained variance (if PCA was used)
                if "embedding" in st.session_state and st.session_state["embedding"].shape[1] == 2:
                    st.markdown("#### 📉 Dimensionality Reduction")
                    st.info("2D embedding computed via PCA. Some information loss is expected when reducing from high-dimensional space.")
            
            else:
                st.warning("⚠️ No latent representation found. Complete integration first.")
        
        except Exception as e:
            st.error(f"❌ Metrics computation failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== EXPORT & DOWNLOAD ====================
st.markdown('<p class="section-header">💾 Step 4: Export Results</p>', unsafe_allow_html=True)

if st.session_state.get("integration_done", False):
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Download Data</h4>
            <p>Export your integrated results for further analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Embedding export
        if "embedding" in st.session_state:
            emb = st.session_state["embedding"]
            clusters = st.session_state["clusters"]
            
            export_df = pd.DataFrame({
                "Cell_ID": range(len(clusters)),
                "UMAP_1": emb[:, 0],
                "UMAP_2": emb[:, 1],
                "Cluster": clusters
            })
            
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Embedding (CSV)",
                data=csv_data,
                file_name="multivi_embedding.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with export_col2:
        st.markdown("""
        <div class="info-card">
            <h4>📋 Analysis Report</h4>
            <p>Generate a summary of your analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📄 Generate Report", use_container_width=True):
            report = f"""
# MultiVI Integration Report

## Dataset Information
- Total Cells: {st.session_state['mdata'].n_obs if 'mdata' in st.session_state else 'N/A'}
- Integration Mode: {mode}
- Number of Clusters: {len(np.unique(st.session_state['clusters']))}

## Parameters
- Training Epochs: {max_epochs}
- Latent Dimensions: {latent_dims}
- Downsampling: {'Yes' if downsample else 'No'}

## Results
Integration completed successfully with {len(np.unique(st.session_state['clusters']))} distinct cell populations identified.

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report,
                file_name="multivi_report.txt",
                mime="text/plain",
                use_container_width=True
            )
else:
    st.info("ℹ️ Complete integration to enable export options.")

# ==================== FOOTER ====================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p><strong>🧬 MultiVI Multi-Omics Integrator</strong></p>
    <p>Developed by <strong>Shubham Mahindrakar</strong> | M.Sc. Bioinformatics</p>
    <p>Powered by: <code>scvi-tools</code> • <code>scanpy</code> • <code>muon</code> • <code>streamlit</code></p>
    <p style="font-size:0.8rem; margin-top:1rem;">
        🔬 Inspired by the Machine Learning for Integrative Genomics (G5) Group, Institut Pasteur
    </p>
</div>
""", unsafe_allow_html=True)
