import os
os.system('clear')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""Broadly useful python packages"""
import pandas as pd
import os
import numpy as np
import pickle
from copy import deepcopy
from shutil import move
import warnings
import torch.nn as nn
"""Machine learning and single cell packages"""
import sklearn.metrics as metrics
from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi
import scanpy as sc
from anndata import AnnData
import seaborn as sns

"""CarDEC"""
from CarDEC import CarDEC_API

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


"""Miscellaneous useful functions"""

def read_pancreas(path, cache=True):
    """A function to read and preprocess the pancreas data"""
    pathlist = os.listdir(path)
    adata = sc.read(os.path.join(path, pathlist[0]))
          
    for i in range(1,len(pathlist)):
        adata = adata.concatenate(sc.read(os.path.join(path, pathlist[i])))

    sc.pp.filter_cells(adata, min_genes = 200)
    sc.pp.filter_genes(adata, min_cells = 30)
    mito_genes = adata.var_names.str.startswith('mt-')
    adata.obs['percent_mito'] = np.sum(
        adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    
    notmito_genes = [not x for x in mito_genes]
    adata = adata[:,notmito_genes]
    del adata.obs['batch']
    print(adata)
    
    return adata

def purity_score(y_true, y_pred):
    """A function to compute cluster purity"""
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def convert_string_to_encoding(string, vector_key):
    """A function to convert a string to a numeric encoding"""
    return np.argwhere(vector_key == string)[0][0]

def convert_vector_to_encoding(vector):
    """A function to convert a vector of strings to a dense numeric encoding"""
    vector_key = np.unique(vector)
    vector_strings = list(vector)
    
    vector_num = [convert_string_to_encoding(string, vector_key) for string in vector_strings]
    
    return vector_num

def build_dir(dir_path):
    subdirs = [dir_path]
    substring = dir_path

    while substring != '':
        splt_dir = os.path.split(substring)
        substring = splt_dir[0]
        subdirs.append(substring)
        
    subdirs.pop()
    subdirs = [x for x in subdirs if os.path.basename(x) != '..']

    n = len(subdirs)
    subdirs = [subdirs[n - 1 - x] for x in range(n)]
    
    for dir_ in subdirs:
        if not os.path.isdir(dir_):
            os.mkdir(dir_)
            
def find_resolution(adata_, n_clusters, random = 0): 
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution = current_res, random_state = random)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))
        
        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        
        iteration = iteration + 1
        
    return current_res

metrics_ = [ari, nmi, purity_score]


# Now, let's read the data in, and take a look at the cell type distribution.

# In[3]:

adata = read_pancreas("../Data/pancreas", cache=True)

celltype = adata.obs['celltype'].values
celltype_vec = convert_vector_to_encoding(celltype)

# sns.distplot(celltype_vec, hist=True, kde=False, 
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})

celltype_key = np.unique(adata.obs['celltype'].values)
for type_ in celltype_key:
    ntype = sum([x == type_ for x in celltype])
    print("The number of cells of type: " + type_ + " is " + str(ntype))


# ## Figure Data

# In[4]:


ARI_data_gene = {'ARI': [0] * 7,
            'NMI': [0] * 7,
            'Purity': [0] * 7,
           'Method': ['Branching'] * 2 + ['Raw'] * 2 + ['Naive All'] * 2 + ['Naive HVG'],
                'Type': ['HVG', 'LVG'] * 3 + ['HVG']}

ARI_data_gene = pd.DataFrame(ARI_data_gene)

ARI_data = {'ARI': [0]*1,
            'NMI': [0]*1,
            'Purity': [0]*1,
           'Method': ['CarDEC']}
ARI_data = pd.DataFrame(ARI_data)

ARI_raw = {'ARI': [0]*2,
            'NMI': [0]*2,
            'Purity': [0]*2,
                'Type': ['HVG', 'LVG']}
ARI_raw = pd.DataFrame(ARI_raw)

figure_path = "../Figures/pancreas"
figure_path_supplement = "../Figures/pancreas_supplement"
figure_path_embedding = "../Figures/pancreas_embedding"

build_dir(figure_path)
build_dir(figure_path_supplement)
build_dir(figure_path_embedding)


# ## Analyze the data with CarDEC
# 
# First initialize the CarDEC class. Doing so will preprocess the data. When normalizing the data, we need to decide how many of the most variable genes should be deemed "highly variable." High variance genes drive clustering, so a reasonably stringent cutoff to deem a gene highly variable is important to getting high quality clusters in some datasets.
# 
# Here, I retain the top 2000 highly variable genes as designated HVGs for driving cluster assignments. Highly variable genes are selected with Scanpy using the "Seurat" selection flavor. All other genes are retained as low variance genes to be fed into the low variance encoder of the CarDEC Model.
# 
# The data we are working with will be stored as an annotated dataframe in the dataset attribute of the CarDEC API object. The main array, accessible via CarDEC.dataset.X, is the original count data. CarDEC.dataset will also inherit observation and variable metadata from the adata object. This metadata is accessible via CarDEC.dataset.obs and CarDEC.dataset.var respectively. CarDEC.dataset.var["Variance Type"] identifies which genes are treated as high variance features and which genes are treated as low variance features.

# In[5]:


CarDEC = CarDEC_API(adata, weights_dir = "Pancreas All/CarDEC_LVG Weights", batch_key = 'tech', n_high_var = 2000)


# Now, let's build the CarDEC model in TensorFlow by calling CarDEC.build_model. Doing so will initialize the high variance feature autoencoder. By default, the CarDEC API will try to load the weights for this autoencoder from the weights directory. If the weights cannot be found, then the autoencoder will be trained from scratch with automatic differentiation, and its weights will be saved.
# 
# We use the mean squared error noise model as the loss of our autoencoder since this noise distribution is very flexible and adaptable to different data modalities (UMI and non-UMI datasets, among others). We will set the cluster weight to be 1, which will weight the clustering loss and reconstruction loss equally during the cluster refinement step.

# In[6]:


CarDEC.build_model(n_clusters = 8)


# # Now we will make inference on the dataset with the CarDEC model. By default the CarDEC API will attempt to load the weights for the full CarDEC model from the weights directory. If the weights cannot be found, then the main CarDEC model will be trained from scratch with automatic differentiation, and its weights will be saved.
# # 
# # Once the weights for the full CarDEC model are initialized, the API will use the model to make inference on the data. This will add several outputs to CarDEC.dataset
# # 
# # layers:
# # 
# # 1. ["denoised"] contains denoised batch corrected features in the gene space. These features are useful for downstream analyses in the gene expression space, like differential expression analysis.
# # obsm:
# # 
# # 1. ["cluster memberships"] an array, where element (i,j) is the probability that cell i is in cluster j
# # 2. ["precluster denoised"] an array, where element (i,j) is the denoised expression value of high variance gene j in cell i. This denoised representation is avalible only for high variance genes.
# # 3. ["embedding"] the low-dimension embedding corresponding to high variance features (after fine tuning)
# # 4. ["LVG embedding"] the combined low-dimension embedding (more precisely, the embedding from the high variance encoder concatenated to the embedding from the low variance encoder). This axis array will be created ONLY if 
# # low variance genes are modeled.
# # 5. ["precluster embedding"] the low-dimension embedding corresponding to high variance features (before fine 
# # tuning)
# # 6. ["initial assignments"] the cluster assignments before finetuning. These assignments are identified only if the autoencoder is trained from scratch. If the autoencoder weights are loaded instead of trained from scratch, this vector will be set to all zeros.

# # In[7]:

CarDEC.make_inference()

# 整个模型是一个多输入和多输出的模型
# # If our data are read counts, then the user may want access to denoised and batch corrected genewise features on the count scale, rather than on the zscore standardized scale that is provided via CarDEC.dataset["denoised"]. Having these denoised expression values on the count scale is useful for certain downstream analyses.
# # 
# # The pancreas data is in fact not count data, so it is not wholly appropriate to model these expression values with the negative binomial model. However, we can still maximize the negative binomial likelihood to obtain denoised counts to see how the results look. In fact, we will see that they turn out pretty well.
# # 
# # To obtained denoised counts, the user can call CarDEC.model_counts. The resultant denoised counts are added as a layer: CarDEC.dataset.layers['denoised counts']. If the keep_dispersion argument for CarDEC.model_counts is set equal to true, then dispersions for the negative binomial likelihood will also be added as a layer: CarDEC.dataset.layers["dispersion"].

# # In[8]:


CarDEC.model_counts()


# # ## Evaluate Cell Clustering

# # First, let us evaluate the cluster assignments of CarDEC. For each cell, we get its most probable cluster by finding which membership probability is maximized. We visualize the clusters by computing the UMAP representation of CarDEC's bottleneck embedding space.
# # 
# # To evaluate cluster assignments, we compute adjusted rand index (ARI), normalized mutual information (NMI), and purity. Visually, the UMAP representation of the embedding space is colored by CarDEC's predicted cluster assignment, gold standard experimentally determined cell type, batch by sample ID, and batch by Macaque ID. Our hope is that clusters are separated by cell type, CarDEC cluster predictions recover true cell types, and that batches are mixed. Indeed, we see high clustering accuracy, and batch effect is successfully removed in the low dimensional space.

# # In[9]:


"""Assessing finetuned cluster assignments"""

temporary = AnnData(CarDEC.dataset.obsm['embedding'])
temporary.obs = CarDEC.dataset.obs
temporary.obs['cell_type'] = temporary.obs['celltype']

sc.tl.pca(temporary, svd_solver='arpack')
sc.pp.neighbors(temporary, n_neighbors = 15)

q = CarDEC.dataset.obsm['cluster memberships']
labels = np.argmax(q, axis=1)
temporary.obs['cluster assignment'] = [str(x) for x in labels]

sc.tl.umap(temporary)
sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# """Figure info for paper"""
# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = 'CarDEC.csv'
# path = os.path.join(figure_path_embedding, base_path)

# DF.to_csv(path)

ARI_data.iloc[0,:3] = ARI, NMI, Purity
ARI_data.to_csv(os.path.join(figure_path_embedding, 'ARIsummary.csv'))

print("CarDEC Clustering Results")
print ("ARI = {0:.4f}".format(ARI)) 
print ("NMI = {0:.4f}".format(NMI)) 
print ("Purity = {0:.4f}".format(Purity))


# Now, let's compute the same metrics and visualizations for the pretrained autoencoder embedding. Since the pretrained autoencoder doesn't have cluster assignments, we compute these ourselves by running Louvain's algorithm on the pretrained autoencoder's embedding.
# 
# We see that the pretrained autoencoder also has very good clustering results when we set the number of clusters to 8, although numerical metrics are a somewhat worse as compared to the full CarDEC model.

# In[10]:


"""Assessing pretrained embedding"""

temporary = AnnData(deepcopy(CarDEC.dataset.obsm['precluster embedding']))
temporary.obs = CarDEC.dataset.obs
temporary.obs['cell_type'] = temporary.obs['celltype']

sc.tl.pca(temporary, svd_solver='arpack')
sc.pp.neighbors(temporary, n_neighbors = 15)

res = find_resolution(temporary, 8)
sc.tl.louvain(temporary, resolution = res)
temporary.obs['cluster assignment'] = temporary.obs['louvain']

sc.tl.umap(temporary)
sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

print("CarDEC Pretrained Clustering Results")
print ("ARI = {0:.4f}".format(ARI)) 
print ("NMI = {0:.4f}".format(NMI)) 
print ("Purity = {0:.4f}".format(Purity))


# ## Evaluating Denoising and Batch Correction of Features

# Here, we evaluate the capacity of CarDEC to denoise and batch correct features both on the zscore scale and on the original count scale. Our strategy is as follows. If genes are of high quality and reflect true cell type differences rather than enduring technical batch effects, then Louvain Clustering that is run on the normalized count matrix of these genes should be good at recovering true cell types. Then if denoising and batch correcting is working and uncovering true cell type differences, we expect that cell type clustering that uses the normalized denoised count matrix should yield higher clustering accuracy than the the original noisy counts.
# 
# To check clustering accuracy and if batch effects are present, we can evaluate our results both numerically using ARI, NMI, Purity, and visually with UMAP. Indeed, we will see exactly what we hope to see. Namely, Louvain's clustering run on the denoised features (both zscore scale and count scale) is far better than Louvain's clustering run on the raw (normalized) counts.
# 
# To get a baseline, let's cluster the original macaque data using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very poor, suggesting that batch effects are very severe and that a more sophisticated method like CarDEC is needed to analyze cell types in this data.

# In[11]:


# """Assessing original counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.X))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering original counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # Now, let's cluster the denoised zscore scale features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[12]:


# """Assessing denoised zscore features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Denoising Results using all denoised features")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # Now, let's cluster the denoised count scale features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[13]:


# """Assessing denoised Counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised counts']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Denoising Results using all denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # ## Evaluating Denoising and Batch Correction of High Variance Features

# # Here, we evaluate the capacity of CarDEC to denoise and batch correct high variance features specifically both on the zscore scale and on the original count scale. Our strategy is as follows.
# # 
# # To check clustering accuracy and if batch effects are present, we can evaluate our results both numerically using ARI, NMI, Purity, and visually with UMAP. Indeed, we will see exactly what we hope to see. Namely, Louvain's clustering run on the denoised features (both zscore scale and count scale) is far better than Louvain's clustering run on the raw (normalized) counts.
# # 
# # To get a baseline, let's cluster the macaque data with only high variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are poor, suggesting that batch effects are very severe and that a more sophisticated method like CarDEC is needed to analyze cell types in this data.

# # In[14]:


# """Assessing original high variance counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.X[:, CarDEC.dataset.var['Variance Type'] == 'HVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering original high variance counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# # """Figure info for paper"""
# # DF = pd.DataFrame(temporary.obsm['X_umap'])
# # DF.columns = ['UMAP1', 'UMAP2']
# # DF['Technology'] = temporary.obs['tech'].values
# # DF['Cell Type'] = temporary.obs['celltype'].values
# # DF.index = temporary.obs.index

# # base_path = 'Raw_HVG.csv'
# # path = os.path.join(figure_path, base_path)
# # DF.to_csv(path)

# ARI_data_gene.iloc[2,:3] = ARI, NMI, Purity
# ARI_raw.iloc[0,:3] = ARI, NMI, Purity


# # Now, let's consider the denoised zscore scale features obtained from the pretrained autoencoder, before finetuning to see the effect of finetuning. We cluster these denoised zscore scale high variance features using scanpy's louvain clustering workflow.

# # In[15]:


# """Assessing pretrained denoising of high variance features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.obsm['precluster denoised']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering pretrained high variance features")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # Now, let's cluster the denoised zscore scale high variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[16]:


# """Assessing denoised zscore features for high variance features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised'][:, CarDEC.dataset.var['Variance Type'] == 'HVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC high variance denoised features")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for main paper"""
# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[0,3] + "_" + ARI_data_gene.iloc[0,4] + ".csv"
# path = os.path.join(figure_path, base_path)
# ARI_data_gene.iloc[0,:3] = ARI, NMI, Purity

# DF.to_csv(path)


# # Now, let's cluster the denoised count scale high variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[17]:


# """Assessing HVG denoised Counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised counts'][:, CarDEC.dataset.var['Variance Type'] == 'HVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC high variance denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # ## Evaluating Denoising and Batch Correction of Low Variance Features

# # Here, we evaluate the capacity of CarDEC to denoise and batch correct low variance features specifically both on the zscore scale and on the original count scale. Our strategy is as follows.
# # 
# # To check clustering accuracy and if batch effects are present, we can evaluate our results both numerically using ARI, NMI, Purity, and visually with UMAP. Indeed, we will see exactly what we hope to see. Namely, Louvain's clustering run on the denoised features (both zscore scale and count scale) is far better than Louvain's clustering run on the raw (normalized) counts.
# # 
# # To get a baseline, let's cluster the macaque data with only low variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are poor, suggesting that batch effects are very severe and that a more sophisticated method like CarDEC is needed to analyze cell types in this data.

# # In[18]:


# """Assessing original low variance counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.X[:, CarDEC.dataset.var['Variance Type'] == 'LVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering original low variance counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for paper"""
# base_path = 'Raw_ARIs.csv'
# path = os.path.join(figure_path, base_path)
# ARI_raw.to_csv(path)

# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[3,3] + "_" + ARI_data_gene.iloc[3,4] + ".csv"
# path = os.path.join(figure_path, base_path)

# ARI_data_gene.iloc[3,:3] = ARI, NMI, Purity
# ARI_raw.iloc[1,:3] = ARI, NMI, Purity

# DF.to_csv(path)


# # Now, let's cluster the denoised zscore scale low variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[19]:


# """Assessing denoised zscore features for low variance features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised'][:, CarDEC.dataset.var['Variance Type'] == 'LVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering low variance denoised features")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for paper"""
# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[1,3] + "_" + ARI_data_gene.iloc[1,4] + ".csv"
# path = os.path.join(figure_path, base_path)

# DF.to_csv(path)

# ARI_data_gene.iloc[1,:3] = ARI, NMI, Purity


# # Now, let's cluster the denoised count scale low variance features using scanpy's louvain clustering workflow. We see that all three clustering accuracy metrics are very good, suggesting that CarDEC successfully removed batch effects while preserving biological signal.

# # In[20]:


# """Assessing LVG denoised Counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised counts'][:, CarDEC.dataset.var['Variance Type'] == 'LVG']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("Clustering low variance denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # ## Potential Question: Why not just model all genes as highly variable?
# # 
# # Retort: Including all genes as highly variable features which can affect CarDEC's cluster assignments sometimes leads to poor clustering results. In this dataset, we see that CarDEC performs very badly if all genes are fed in as highly variable features, rather than using the special architecture to treat highly variable features distinctly from lowly variable features.
# # 
# # Let's explore the performance of CarDEC when treating all genes as highly variable. First, initialze the CarDEC API, build the model, make inference, and get denoised counts.

# # In[21]:


# geneinfo = CarDEC.dataset.var

# CarDEC = CarDEC_API(adata, weights_dir = "Pancreas All/CarDEC_All", batch_key = 'tech', 
#                     n_high_var = None, LVG = False)


# # In[22]:


# CarDEC.build_model(n_clusters = 8)


# # In[23]:


# CarDEC.make_inference()


# # In[24]:


# CarDEC.model_counts()


# # Now, let's assess the performance of this model for clustering.

# # In[25]:


# """Assessing finetuned cluster assignments"""

# temporary = AnnData(CarDEC.dataset.obsm['embedding'])
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# q = CarDEC.dataset.obsm['cluster memberships']
# labels = np.argmax(q, axis=1)
# temporary.obs['cluster assignment'] = [str(x) for x in labels]

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Clustering Results")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # We can also assess the performance of this framework for producing denoised features in the gene space on the zscore scale for HVGs.

# # In[26]:


# """Assessing denoised zscore features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised']))
# HVGs = geneinfo.index.values[geneinfo['Variance Type'].values == 'HVG']
# indices = [x in HVGs for x in list(CarDEC.dataset.var.index)]
# temporary = temporary[:, indices]

# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Denoising Results using all denoised features, for HVGs")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for paper"""

# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[4,3] + "_" + ARI_data_gene.iloc[4,4] + ".csv"
# path = os.path.join(figure_path, base_path)
# ARI_data_gene.iloc[4,:3] = ARI, NMI, Purity

# DF.to_csv(path)


# # We can also assess the performance of this framework for producing denoised features in the gene space on the zscore scale for LVGs.

# # In[27]:


# """Assessing denoised zscore features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised']))
# HVGs = geneinfo.index.values[geneinfo['Variance Type'].values == 'LVG']
# indices = [x in HVGs for x in list(CarDEC.dataset.var.index)]
# temporary = temporary[:, indices]

# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Denoising Results using all denoised features")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for paper"""

# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[5,3] + "_" + ARI_data_gene.iloc[5,4] + ".csv"
# path = os.path.join(figure_path, base_path)
# ARI_data_gene.iloc[5,:3] = ARI, NMI, Purity

# DF.to_csv(path)


# # We can also assess the performance of this framework for producing denoised features in the gene space on the count scale.

# # In[28]:


# """Assessing denoised Counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised counts']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Denoising Results using all denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # ## Potential Question: By modeling low variance genes through the introduction of millions of parameters, does CarDEC compromise performance on high variance genes?
# # 
# # Retort: Our special architecture is explicitly designed to maintain maximal performance on the high variance features even when modeling low variance features as well. To demonstrate this, let's fit CarDEC only with highly variable features and demonstrate that the model does not do any better at clustering and denoising highly variable features than the HVG/LVG CarDEC model that also models low variance features.
# # 
# # Let's explore the performance of CarDEC when modeling only the highly variable genes. First, initialze the CarDEC API, build the model, make inference, and get denoised counts.

# # In[29]:


# ARI_HVGo_gene = {'ARI': [0] * 1,
#             'NMI': [0] * 1,
#             'Purity': [0] * 1,
#            'Method': ['CarDEC']}
# ARI_HVGo_gene = pd.DataFrame(ARI_HVGo_gene)


# # In[30]:


# CarDEC = CarDEC_API(adata, weights_dir = "Pancreas All/CarDEC_HVG Weights", batch_key = 'tech', 
#                     n_high_var = 2000, LVG = False)


# # In[31]:


# CarDEC.build_model(n_clusters = 8)


# # In[32]:


# CarDEC.make_inference()


# # In[33]:


# CarDEC.model_counts()


# # First, let's demonstrate that we maintain excellent clustering performance by computing clustering accuracy from the outputed membership probabilities and examining the UMAP output. As we can see, this HVG model does no better at clustering than our HVG/LVG model.

# # In[34]:


# """Assessing finetuned cluster assignments"""

# temporary = AnnData(CarDEC.dataset.obsm['embedding'])
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# q = CarDEC.dataset.obsm['cluster memberships']
# labels = np.argmax(q, axis=1)
# temporary.obs['cluster assignment'] = [str(x) for x in labels]

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# print("CarDEC Clustering Results")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))


# # Also, let's verify that our HVG/LVG model denoises highly variable genes on the zscore scale just as well as the HVG only model. We can do this by running Louvain clustering on the denoised counts outputted by the HVG model and demonstrating that the clustering is no better than the HVG/LVG model both quantitatively (ARI, NMI, Purity) and visually (UMAP).

# # In[35]:


# """Assessing denoised zscore features"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]
# ARI_HVGo_gene.iloc[0, :3] = ARI, NMI, Purity
# ARI_HVGo_gene.to_csv(os.path.join(figure_path_supplement, 'genespaceARIs_HVGo.csv'))

# print("CarDEC Denoising Results using all denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

# """Figure info for paper"""

# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['celltype'].values
# DF.index = temporary.obs.index

# base_path = ARI_data_gene.iloc[6,3] + "_" + ARI_data_gene.iloc[6,4] + ".csv"
# path = os.path.join(figure_path, base_path)
# ARI_data_gene.iloc[6,:3] = ARI, NMI, Purity

# DF.to_csv(path)
# ARI_data_gene.to_csv(os.path.join(figure_path, 'genespaceARIs.csv'))


# # Also, let's verify that our HVG/LVG model denoises highly variable genes on the count scale just as well as the HVG only model. We can do this by running Louvain clustering on the denoised counts outputted by the HVG model and demonstrating that the clustering is no better than the HVG/LVG model both quantitatively (ARI, NMI, Purity) and visually (UMAP).

# # In[36]:


# """Assessing denoised Counts"""

# temporary = AnnData(deepcopy(CarDEC.dataset.layers['denoised counts']))
# temporary.obs = CarDEC.dataset.obs
# temporary.obs['cell_type'] = temporary.obs['celltype']

# sc.pp.normalize_total(temporary)
# sc.pp.log1p(temporary)
# sc.pp.scale(temporary)

# sc.tl.pca(temporary, svd_solver='arpack')
# sc.pp.neighbors(temporary, n_neighbors = 15)

# res = find_resolution(temporary, 8)
# sc.tl.louvain(temporary, resolution = res)
# temporary.obs['cluster assignment'] = temporary.obs['louvain']

# sc.tl.umap(temporary)
# sc.pl.umap(temporary, color = ["cell_type", "cluster assignment", "tech"], return_fig = True)

# ARI, NMI, Purity = [metric(temporary.obs['cell_type'], temporary.obs['cluster assignment']) for metric in metrics_]

# """Figure info for paper"""
# ARI_HVGo_gene.iloc[0,:3] = ARI, NMI, Purity
# DF = pd.DataFrame(temporary.obsm['X_umap'])
# DF.columns = ['UMAP1', 'UMAP2']
# DF['Technology'] = temporary.obs['tech'].values
# DF['Cell Type'] = temporary.obs['cell_type'].values
# DF.index = temporary.obs.index

# base_path = 'CarDEC_HVGOnly.csv'
# path = os.path.join(figure_path_supplement, base_path)

# DF.to_csv(path)
# ARI_HVGo_gene.to_csv(os.path.join(figure_path_supplement, 'genespaceARIs_HVGo.csv'))

# print("CarDEC Denoising Results using all denoised counts")
# print ("ARI = {0:.4f}".format(ARI)) 
# print ("NMI = {0:.4f}".format(NMI)) 
# print ("Purity = {0:.4f}".format(Purity))

