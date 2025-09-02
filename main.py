from library import load_images,rfft_visualize,tucker_spatial,tucker_fft_reconstruct,plot_tucker_fft_grid

folder=r"cat_data" #Path of the folder
data=load_images(folder) 

#Applying 3D-RFFT and visualization of the real part.
rfft_real=rfft_visualize(data)

#Applying tucker reconstruction 
tucker=tucker_spatial(data)

#Applying iFFT reconstruction 
tucker_fft=tucker_fft_reconstruct(data)

#Grid visualization
X_np, X_rec_spatial = tucker          
X_restored_np = tucker_fft            
img_index = 0                          

plot_tucker_fft_grid(X_np, X_rec_spatial, X_restored_np, img_index)