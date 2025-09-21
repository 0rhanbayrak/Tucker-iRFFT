🐱 Image Compression with FFT and Tucker Decomposition

This project explores image compression and reconstruction using advanced tensor decomposition techniques. The main goal was to compare two approaches:

Spatial Tucker Decomposition

FFT-Tucker Decomposition (Fast Fourier Transform + Tucker)

🚀 Key Highlights

Used memory-saving strategies for contextual image compression.

Implemented FFT, Tucker modeling, and GPU-accelerated computing.

Dataset: ~900 cat images 🐱.

Experimented with UMAP and SOM for automated clustering and dataset cleaning (later done manually due to high similarity of images).

Built a GPU passthrough JupyterHub server for experiments.

📊 Results

Achieved 100x image compression.

Over 99% reconstruction accuracy.

Outperformed comparable methods in all tests.

Completed within 3.5 months of research and development.

🔬 Methods

Tucker Decomposition: tensor factorization into a core tensor and factor matrices.

FFT / RFFT: frequency-domain representation to improve compression efficiency.

Mode-n unfolding & n-mode product: key tensor operations for decomposition.

Visualization: error maps, PLS analysis, and clustering attempts.

📈 Future Work

Explore CP decomposition.

Apply methods to larger, real-world datasets.

Move towards real-time image compression applications.

🙏 Acknowledgements

This work was supervised by Prof. Michael Sorochan Armstrong, with contributions from Jesús García Sánchez, Daniel Vallejo España, and José Camacho Páez.

Project supported by:

MuSTARD Project (link
), grant no. PID2023-1523010B-IOO.

Agencia Estatal de Investigación in Spain (MICIU/AEI/10.13039/501100011033).

European Regional Development Fund.

Horizon Europe Marie Skłodowska-Curie project (MAHOD), grant no. 101106986 🇪🇺 🇪🇦

💻 Developed during my Erasmus+ internship at the Computational Data Science (CoDaS) Lab, University of Granada
