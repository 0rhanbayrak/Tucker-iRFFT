from library import load_images,rfft_visualize,tucker_spatial,tucker_fft_reconstruct,plot_tucker_fft_grid,sizes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
folder=r"C:\Users\Orhan Bayrak\Desktop\Tucker-FFT\cat_data" #Path of the folder
data=load_images(folder) 

#Applying 3D-RFFT and visualization of the real part.
rfft_real=rfft_visualize(data)

# %%
#Applying tucker reconstruction 

tucker=tucker_spatial(data)

#Applying iFFT reconstruction 
tucker_fft, X_fft3, Us, G = tucker_fft_reconstruct(data)

#Grid visualization
X_np, X_rec_spatial = tucker          
X_restored_np = tucker_fft            

spat_list = []
fft_list = []                          

for ii in range(9):
    rel_spat, rel_fft = plot_tucker_fft_grid(X_np, X_rec_spatial, X_restored_np, ii)
    spat_list.append(rel_spat)
    fft_list.append(rel_fft)

spat_list = np.array(spat_list)
fft_list = np.array(fft_list)
x = np.array(range(9))

diff = fft_list - spat_list

print("FFT better", np.sum(diff > 0))
print("SPAT better", np.sum(diff < 0))

# Make sure differences are positive if using log scale

# Plot
plt.figure(figsize=(6,4))
plt.plot(x, diff, marker='o', label="|A - B|")

plt.yscale('symlog')  # log scale on y-axis
plt.xlabel("Index")
plt.ylabel("Difference (log scale)")
plt.title("Difference between arrays A and B")
plt.legend()
plt.grid(True, which="both", ls="--", linewidth=0.5)    

sizes(data, X_fft3, Us, G)

"""""""""""
#PLS

# Design Matrix
X_design = np.array([
    (16,16,1,6),
    (16,32,2,12),
    (16,48,3,24),
    (32,16,2,24),
    (32,32,3,6),
    (32,48,1,12),
    (48,16,3,12),
    (48,32,1,24),
    (48,48,2,6),
])

y_spat = spat_list.reshape(-1,1)
y_fft = fft_list.reshape(-1,1)

# PLS fit
pls_spat = PLSRegression(n_components=2)
pls_spat.fit(X_design, y_spat)

pls_fft = PLSRegression(n_components=2)
pls_fft.fit(X_design, y_fft)

# PLS latent scores
T_spat = pls_spat.x_scores_  # X tarafı
U_spat = pls_spat.y_scores_  # y tarafı

T_fft = pls_fft.x_scores_
U_fft = pls_fft.y_scores_

# Visualize
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(T_spat[:,0], T_spat[:,1], c='blue', label='Spatial X scores')
plt.scatter(U_spat[:,0], U_spat[:,1], c='cyan', marker='x', label='Spatial Y scores')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("PLS Scores - Spatial")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(T_fft[:,0], T_fft[:,1], c='red', label='FFT X scores')
plt.scatter(U_fft[:,0], U_fft[:,1], c='orange', marker='x', label='FFT Y scores')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title("PLS Scores - FFT")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
"""""""""""