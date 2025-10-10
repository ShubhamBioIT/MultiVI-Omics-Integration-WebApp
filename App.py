# app.py
# MultiVI Integrator — polished: explicit demo controls, fixed upload, 3D marker plots, reset button
# Author: adjusted for Shubham

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
from sklearn.neighbors import NearestNeighbors

# lazy heavy imports
HAS_SCVI = False
try:
    import muon as mu
    import scanpy as sc
    import scvi
    import torch
    HAS_SCVI = True
except Exception:
    # not fatal; we will fallback to fast demo for UI
    HAS_SCVI = False

# ----------------------- Utility helpers -----------------------
def save_uploaded_file(uploaded_file, suffix=".h5mu"):
    """Save the uploaded streamlit file to a temp path and return path."""
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_file.getbuffer())
        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception as e:
        raise RuntimeError(f"Failed to save uploaded file: {e}")

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def reset_session_and_files():
    # remove saved temp file if exists
    p = st.session_state.get("mdata_path", None)
    if p:
        safe_remove(p)
    # clean keys
    for k in ["mdata_path", "mdata", "model", "latent", "embedding", "clusters", "latent_concat", "demo_bytes"]:
        if k in st.session_state:
            del st.session_state[k]

# ----------------------- Page & CSS styling -----------------------
st.set_page_config(page_title="MultiVI WebTool", page_icon="🧬", layout="wide")
st.markdown("""
<style>
body { background: linear-gradient(#ffffff, #f7fbfb); }
.header { padding: 8px 12px; border-radius:12px; background: linear-gradient(90deg,#0ea5a6,#06b6d4); color:white; }
.card { background: white; border-radius:12px; padding:12px; box-shadow: 0 6px 18px rgba(14,165,166,0.06); }
.small { font-size:0.9rem; color:#6b7280; }
.right { text-align:right; }
.controls { display:flex; gap:8px; flex-wrap:wrap; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='header'><h2>🧬 Multi-Omics Integrator (MultiVI)</h2></div>", unsafe_allow_html=True)
st.markdown("<div class='small'>Integrates scRNA + scATAC (MultiVI) with a fast demo fallback. Upload .h5mu or use demo.</div>", unsafe_allow_html=True)
st.write("")

# ----------------------- Sidebar Options -----------------------
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Integration Mode", ["Fast Demo (very fast)", "MultiVI (full)"])
use_demo = st.sidebar.checkbox("Show demo options (do not auto-load)", value=True)
downsample = st.sidebar.checkbox("Downsample for speed", value=True)
downsample_n = st.sidebar.slider("If downsample: keep N cells", 100, 5000, 1000, step=100)
max_epochs = st.sidebar.slider("Max epochs (MultiVI)", 1, 200, 25)
quick_pcs = st.sidebar.slider("Fast Demo: PCs per modality", 5, 50, 20)

# ----------------------- Top controls (Dataset area) -----------------------
st.header("1) Dataset — Download / Upload / Load")
st.markdown("You can **download the demo** and then press **Load demo** to load it into the app. Or upload your `.h5mu` file using Upload control. The app will not auto-download anything.")

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    st.markdown("**Demo PBMC multiome**")
    demo_url = "https://figshare.com/ndownloader/files/54794234"
    st.markdown(
        f"[Download demo dataset (.h5mu)]({demo_url})",
        unsafe_allow_html=True
    )
    st.info("Click the link above to download the demo dataset directly to your system. After downloading, use the Upload control to load it into the app.")

with col2:
    uploaded = st.file_uploader("Upload your .h5mu file (optional)", type=["h5mu"])
    if uploaded is not None:
        try:
            saved_path = save_uploaded_file(uploaded, suffix=".h5mu")
            st.session_state["mdata_path"] = saved_path
            st.success(f"Uploaded and saved to {saved_path}")
        except Exception as e:
            st.error(f"Upload failed: {e}")

with col3:
    if st.button("Reset app"):
        reset_session_and_files()
        st.rerun()

st.markdown("---")

# ----------------------- Prepare / Inspect dataset -----------------------
st.header("2) Prepare & Inspect")
if "mdata_path" in st.session_state and st.session_state["mdata_path"]:
    st.markdown(f"Temp file path: `{st.session_state['mdata_path']}`")
else:
    st.info("No demo loaded or file uploaded yet. Use the controls above to provide a file.")

if st.button("Prepare / Load dataset"):
    if "mdata_path" not in st.session_state or not st.session_state["mdata_path"]:
        st.error("No file path available. Download demo or upload a file first.")
    else:
        path = st.session_state["mdata_path"]
        if not HAS_SCVI:
            st.warning("muon/scvi-tools not installed. Full dataset loading won't work. Install muon & scvi-tools to use full features.")
            st.session_state["mdata"] = None
        else:
            try:
                mdata = mu.read_h5mu(path)
                # ensure joint indices/metadata are materialized
                try:
                    mdata.update()
                except Exception:
                    pass
                # standardize expected names (common tutorial uses rna_subset / atac_subset)
                if "rna_subset" not in mdata.mod and "rna" in mdata.mod:
                    mdata.mod["rna_subset"] = mdata.mod["rna"]
                if "atac_subset" not in mdata.mod and "atac" in mdata.mod:
                    mdata.mod["atac_subset"] = mdata.mod["atac"]

                # optional downsample with robust obs_names fallback
                def _obs_names(md):
                    try:
                        names = md.obs_names
                        if names is not None and len(names) > 0:
                            return names
                    except Exception:
                        pass
                    mods = list(md.mod.keys())
                    if mods:
                        return md.mod[mods[0]].obs_names
                    return []
                if downsample and mdata.n_obs > 0:
                    names = _obs_names(mdata)
                    if len(names) > 0 and downsample_n < mdata.n_obs:
                        sel = names[:downsample_n]
                        mdata = mdata[sel, :]

                st.session_state["mdata"] = mdata
                st.success(f"MuData loaded. Cells: {mdata.n_obs}, RNA genes: {mdata.mod['rna_subset'].var.shape[0] if 'rna_subset' in mdata.mod else 'NA'}")
                # Display the obs table robustly (fallback to modality obs if joint obs is empty)
                obs_df = None
                try:
                    if isinstance(mdata.obs, pd.DataFrame) and mdata.obs.shape[0] > 0:
                        obs_df = mdata.obs.copy()
                except Exception:
                    obs_df = None
                if obs_df is None or obs_df.shape[0] == 0:
                    if "rna_subset" in mdata.mod:
                        obs_df = mdata.mod["rna_subset"].obs.copy()
                    else:
                        mods = list(mdata.mod.keys())
                        obs_df = mdata.mod[mods[0]].obs.copy() if mods else pd.DataFrame(index=_obs_names(mdata))
                if obs_df.shape[0] == 0:
                    st.warning("No cells found in obs table after loading/downsampling.")
                else:
                    st.dataframe(obs_df.head(10))
            except Exception as e:
                st.error("Failed to read .h5mu: " + str(e))
                st.error(traceback.format_exc())

# ----------------------- Integration / Train -----------------------
st.header("3) Integration / Train")
st.markdown("Choose `Fast Demo` for instant results or `MultiVI` for full model training (requires muon & scvi-tools).")

col_a, col_b = st.columns([2, 2])
with col_a:
    if mode == "Fast Demo (very fast)":
        if st.button("Run Fast Demo Integration"):
            try:
                # if we have a loaded mdata and muon, extract matrices
                if HAS_SCVI and st.session_state.get("mdata") is not None:
                    mdata = st.session_state["mdata"]
                    try:
                        rna = mdata.mod["rna_subset"].X
                        atac = mdata.mod["atac_subset"].X
                    except Exception:
                        # fallback: use first two modality layers
                        mods = list(mdata.mod.keys())
                        rna = mdata.mod[mods[0]].X
                        atac = mdata.mod[mods[1]].X if len(mods) > 1 else mdata.mod[mods[0]].X

                    # quick PCA/SVD concatenation
                    from sklearn.decomposition import TruncatedSVD
                    def to_dense(x):
                        if hasattr(x, "toarray"):
                            return x.toarray()
                        return np.array(x)
                    rna_dense = to_dense(rna)
                    atac_dense = to_dense(atac)
                    pca = PCA(n_components=min(quick_pcs, max(2, rna_dense.shape[1]-1)))
                    rna_pc = pca.fit_transform(rna_dense)
                    svd = TruncatedSVD(n_components=min(quick_pcs, max(2, atac_dense.shape[1]-1)))
                    atac_pc = svd.fit_transform(atac_dense)
                    concat = np.concatenate([rna_pc, atac_pc], axis=1)
                    emb2d = PCA(n_components=2).fit_transform(concat)
                    st.session_state["embedding"] = emb2d
                    st.session_state["latent_concat"] = concat
                    k = min(8, max(2, int(np.sqrt(concat.shape[0]))))
                    labels = KMeans(n_clusters=k, random_state=0).fit_predict(concat)
                    st.session_state["clusters"] = labels.astype(str)
                    st.success("Fast integration done. Go to Visualization.")
                else:
                    # synthetic demo if muon missing
                    n = downsample_n
                    rng = np.random.RandomState(0)
                    emb = rng.normal(size=(n, 2))
                    labels = (rng.randint(0, 4, size=n)).astype(str)
                    st.session_state["embedding"] = emb
                    st.session_state["clusters"] = labels
                    st.success("Synthetic fast demo ready.")
            except Exception as e:
                st.error("Fast demo failed: " + str(e))
                st.error(traceback.format_exc())
    else:
        # MultiVI mode
        if st.button("Train MultiVI (full)"):
            if not HAS_SCVI:
                st.error("scvi-tools & muon must be installed to run MultiVI.")
            elif st.session_state.get("mdata") is None:
                st.error("No MuData loaded. Prepare dataset first.")
            else:
                try:
                    mdata = st.session_state["mdata"]
                    # robust setup_mudata: try tutorial keys, fallback to first two modalities
                    try:
                        scvi.model.MULTIVI.setup_mudata(mdata,
                                                       modalities={"rna_layer": "rna_subset", "atac_layer": "atac_subset"})
                    except Exception:
                        mods = list(mdata.mod.keys())
                        rna_key = "rna_subset" if "rna_subset" in mdata.mod else mods[0]
                        atac_key = "atac_subset" if "atac_subset" in mdata.mod else (mods[1] if len(mods)>1 else mods[0])
                        scvi.model.MULTIVI.setup_mudata(mdata, modalities={"rna_layer": rna_key, "atac_layer": atac_key})

                    model = scvi.model.MULTIVI(mdata,
                                               n_genes=len(mdata.mod["rna_subset"].var),
                                               n_regions=len(mdata.mod["atac_subset"].var))
                    if torch.cuda.is_available():
                        st.info("CUDA available - training on GPU if environment configured")
                    with st.spinner("Training MultiVI (this may take a while)..."):
                        model.train(max_epochs=max_epochs)
                    st.session_state["model"] = model
                    st.session_state["latent"] = model.get_latent_representation()
                    st.success("MultiVI training finished.")
                except Exception as e:
                    st.error("MultiVI failed: " + str(e))
                    st.error(traceback.format_exc())

# ----------------------- Visualization -----------------------
st.header("4) Visualization")
st.markdown("Explore embedding, cluster sizes, expression heatmap and interactive 3D marker gene plots.")

viz_tabs = st.tabs(["Embedding", "Cluster sizes", "Expression heatmap", "Top genes (3D)"])

with viz_tabs[0]:
    st.subheader("UMAP-like / 2D embedding")
    if st.button("Show embedding"):
        try:
            if "embedding" not in st.session_state or st.session_state["embedding"] is None:
                if "latent" in st.session_state and st.session_state["latent"] is not None:
                    emb = PCA(n_components=2).fit_transform(st.session_state["latent"])
                    st.session_state["embedding"] = emb
                else:
                    st.error("No embedding available. Run integration or train MultiVI.")
            emb = st.session_state.get("embedding")
            labels = st.session_state.get("clusters", np.array(["0"]* (emb.shape[0] if emb is not None else 0)))
            df = pd.DataFrame({"x": emb[:,0], "y": emb[:,1], "cluster": labels})
            fig = px.scatter(df, x="x", y="y", color="cluster", title="Integrated embedding (2D)", width=900, height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each point = a cell. Colors = clusters (KMeans or MultiVI+Leiden depending on mode).")
        except Exception as e:
            st.error("Embedding failed: " + str(e))
            st.error(traceback.format_exc())

with viz_tabs[1]:
    st.subheader("Cluster sizes")
    if st.session_state.get("clusters") is not None:
        counts = pd.Series(st.session_state["clusters"]).value_counts().sort_index()
        fig = px.bar(x=counts.index, y=counts.values, labels={"x":"Cluster", "y":"Cell count"}, title="Cluster sizes")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Cluster sizes help detect tiny clusters that may be noise.")

with viz_tabs[2]:
    st.subheader("Mean expression heatmap (top genes)")
    if HAS_SCVI and st.session_state.get("mdata") is not None:
        try:
            mdata = st.session_state["mdata"]
            if "rna_subset" in mdata.mod:
                rna = mdata.mod["rna_subset"]
                if st.session_state.get("clusters") is None:
                    st.warning("Run integration to compute clusters first.")
                else:
                    rna.obs["cluster"] = st.session_state["clusters"]
                    # compute mean expression per cluster, show top 30 genes by variance
                    df_expr = rna.to_df()
                    var_genes = df_expr.var(axis=0).sort_values(ascending=False).index[:30]
                    mean_by_cluster = df_expr[var_genes].groupby(rna.obs["cluster"]).mean().T
                    fig = px.imshow(mean_by_cluster, labels={"x":"Cluster","y":"Gene"}, title="Mean expression (top 30 variable genes)")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("RNA modality 'rna_subset' not found in this dataset.")
        except Exception as e:
            st.error("Heatmap error: " + str(e))
            st.error(traceback.format_exc())
    else:
        st.info("Heatmap requires muon & scvi-tools and a loaded MuData.")

with viz_tabs[3]:
    st.subheader("Top genes 3D plot (interactive)")
    st.markdown("This uses the top 3 marker genes per valid cluster and shows each cell's expression (x,y,z). If MultiVI/DE not available, you can enter any 3 genes manually.")

    if st.button("Compute 3D top-gene plot"):
        try:
            if not HAS_SCVI or st.session_state.get("mdata") is None:
                st.error("3D marker plot requires muon & scvi-tools and a loaded MuData.")
            else:
                mdata = st.session_state["mdata"]
                rna = mdata.mod.get("rna_subset", None)
                if rna is None:
                    st.error("RNA modality not found.")
                else:
                    # ensure cluster labels present
                    if st.session_state.get("clusters") is None:
                        st.error("No clusters found. Run integration first.")
                        st.stop()

                    # attach cluster and filter tiny clusters
                    rna.obs["cluster"] = st.session_state["clusters"]
                    counts = rna.obs["cluster"].value_counts()
                    valid = counts[counts >= 3].index.tolist()
                    if len(valid) < 1:
                        st.error("No cluster with >=3 cells; cannot compute reliable markers.")
                        st.stop()

                    # run DE on filtered data
                    rna_filtered = rna[rna.obs["cluster"].isin(valid)].copy()
                    sc.tl.rank_genes_groups(rna_filtered, groupby="cluster", method="wilcoxon", pts=True)
                    names = rna_filtered.uns["rank_genes_groups"]["names"]
                    # choose the first cluster (as example) and top 3 genes
                    first_group = list(names.dtype.names)[0]
                    top3 = list(names[first_group][:3])
                    st.write(f"Using top 3 genes from cluster {first_group}: {top3}")

                    # get expression dataframe for these genes
                    expr_df = rna.to_df()
                    for g in top3:
                        if g not in expr_df.columns:
                            st.error(f"Gene {g} not found in expression matrix; cannot plot.")
                            st.stop()

                    # 3D scatter: axes are expression of each gene; color by cluster
                    x = expr_df[top3[0]].values
                    y = expr_df[top3[1]].values
                    z = expr_df[top3[2]].values
                    clusters = st.session_state["clusters"]

                    fig = px.scatter_3d(
                        x=x, y=y, z=z,
                        color=clusters,
                        labels={"x": top3[0], "y": top3[1], "z": top3[2]},
                        title=f"3D expression plot: {top3[0]}, {top3[1]}, {top3[2]}",
                        width=900, height=700
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Each point is a cell; axes = expression of the 3 genes. Rotate and zoom for inspection.")
        except Exception as e:
            st.error("3D plot failed: " + str(e))
            st.error(traceback.format_exc())

# ----------------------- Footer -----------------------
st.markdown("---")
st.markdown("""
**Notes & Tips**:
- Demo must be downloaded explicitly and then loaded. This avoids automatic downloads.
- Upload writes your file to a temporary file and loads it — check the temp path printed above.
- Fast Demo is useful for quick interactive exploration. Full MultiVI requires muon & scvi-tools and a machine with enough RAM (GPU recommended).
- Use Reset to clear temporary files and state.
""")
