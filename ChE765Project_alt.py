import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap

import sklearn.discriminant_analysis
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression as pls

training_data = pd.read_excel('PV_train_nonZero_alt.xls')

tc = training_data['Label']
tv = training_data.drop('Label', axis=1)

print (tc)
# separation of class labels and (non-zero)

training_class = tc.to_numpy()
training_vals = tv.to_numpy()

# standardize/scale dataset

X = StandardScaler().fit_transform(training_vals)

# PCA dimensionality red/vis; n_components 3

principal1 = PCA(n_components=3)
pca_3comp = principal1.fit_transform(X)

pca_3comp = np.vstack((pca_3comp.T, training_class)).T
principal1_DF = pd.DataFrame({
    'PCA Component 1':pca_3comp[:,0],
    'PCA Component 2':pca_3comp[:,1],
    'PCA Component 3':pca_3comp[:,2],
    'Label':pca_3comp[:,3]})

fig_principal1 = plt.figure(figsize=(10,10))

sns.scatterplot(
    x='PCA Component 1', y='PCA Component 2',
    hue='Label',
    palette=sns.color_palette("dark",2),
    data=principal1_DF,
    alpha=0.5,
    legend=False
)

# PCA dimensionality red/vis; n_components 4

time_start = time.time()

principal2 = PCA(n_components=4)
pca_4comp = principal2.fit_transform(X)

print('\nPCA compute time: {} seconds'.format(time.time()-time_start))

pca_4comp = np.vstack((pca_4comp.T, training_class)).T
principal2_DF = pd.DataFrame({
    'PCA Component 1 (n=4)':pca_4comp[:,0],
    'PCA Component 2 (n=4)':pca_4comp[:,1],
    'PCA Component 3 (n=4)':pca_4comp[:,2], 
    'PCA Component 4 (n=4)':pca_4comp[:,3],
    'Label':pca_4comp[:,4]}
)

print('\nExplained variation per principal component: {}\n'.format(principal2.explained_variance_ratio_))

# tSNE
# PCA init, early_exaggeration yields no discernible benefit 

time_start = time.time()

tsne = TSNE(
    n_components=2,
    verbose=1,
    method='barnes_hut',
    metric='euclidean',
    learning_rate=50,
    angle=0.2,

    # variables
    perplexity=400,
    early_exaggeration=60,
    n_iter=5000
)

tsne_output=tsne.fit_transform(X)

print('\ntSNE compute time: {} seconds'.format(time.time()-time_start))

tsne_df=pd.DataFrame({'t-SNE Component 1':tsne_output[:,0],'t-SNE Component 2':tsne_output[:,1], 'Label':pca_3comp[:,3]})
tsne_df.to_csv("./tsne_output.csv", sep = '\t', index = True, header = True)

fig_tsne=plt.figure(figsize=(10,10))
sns.scatterplot(
    x="t-SNE Component 1", y ="t-SNE Component 2",
    hue="Label",
    palette=sns.color_palette("dark",2),
    data=tsne_df,
    alpha=0.3,
    legend=False
)

# UMAP in 2D space

time_start=time.time()

umap=umap.UMAP(
    n_components=2,
    metric='euclidean',
    spread=1,
    learning_rate=10,
    verbose=1,

    # variables
    n_neighbors=50,
    min_dist=0.50,
    transform_queue_size=800
)

umap_output=umap.fit_transform(X)

print('\nUMAP compute time: {} seconds\n'.format(time.time()-time_start))

umap_df=pd.DataFrame({'UMAP Component 1':umap_output[:,0],'UMAP Component 2':umap_output[:,1], 'Label':pca_3comp[:,3]})
umap_df.to_csv("./umap_output.csv", sep = '\t', index = True, header = True)

fig_umap=plt.figure(figsize=(10,10))
sns.scatterplot(
    x="UMAP Component 1", y="UMAP Component 2",
    hue="Label",
    palette=sns.color_palette("dark",2),
    data=umap_df,
    alpha=0.3,
    legend=False
)

plt.show()