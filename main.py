from library import load_images,rfft_visualize,tucker_spatial,tucker_fft_reconstruct,plot_tucker_fft_grid
import numpy as np
import matplotlib.pyplot as plt
folder=r"cat_data" #Path of the folder
data=load_images(folder) 

#Applying 3D-RFFT and visualization of the real part.
rfft_real=rfft_visualize(data)

# %%
#Applying tucker reconstruction 

tucker=tucker_spatial(data)

#Applying iFFT reconstruction 
tucker_fft=tucker_fft_reconstruct(data)

#Grid visualization
X_np, X_rec_spatial = tucker          
X_restored_np = tucker_fft            

spat_list = []
fft_list = []                          

for ii in range(500):
    rel_spat, rel_fft = plot_tucker_fft_grid(X_np, X_rec_spatial, X_restored_np, ii)
    spat_list.append(rel_spat)
    fft_list.append(rel_fft)

spat_list = np.array(spat_list)
fft_list = np.array(fft_list)
x = np.array(range(500))

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
