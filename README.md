#📌 Image Captioning using CNN and Transformers + VAE Implementation**

##📄 Research Overview<br>
This project is based on our research paper titled:<br>
**“Image Captioning using CNN and Transformers”**<br>
The work focuses on generating **contextually meaningful captions from images** by combining:
- Convolutional Neural Networks (CNNs) for feature extraction<br>
- Transformer-based architecture for sequence generation<br>
We use VGG16 as the encoder and a Transformer decoder with attention mechanisms to generate captions. The model is trained on the Flickr8k dataset, achieving promising results in generating fluent and coherent captions .

##💡 Key Contributions
- Hybrid CNN + Transformer architecture
- Use of multi-head attention for better context understanding
- Implementation of image feature extraction using VGG16
- Evaluation using BLEU score metrics
- Additional implementation of a Variational Autoencoder (VAE) for image reconstruction

##🧠 Code Overview<br>
This repository includes:

**1. 📷 Image Captioning (Research Work)**
- CNN (VGG16) for feature extraction
- Transformer-based encoder-decoder
- Multi-head attention mechanism
- Caption generation pipeline

**2. 🔁 Variational Autoencoder (VAE)<br>**
Implemented using PyTorch for:
- Image compression
- Latent representation learning
- Image reconstruction (CIFAR-10 dataset)

##🏗️ VAE Architecture<br>
**Encoder**
- Conv layers → Feature extraction
- Fully connected layers → Latent space
- Outputs:
  - Mean (μ)
  - Log variance (log σ²)

**Sampling (Reparameterization Trick)**<br>
z = μ + ε * σ

**Decoder**
- Fully connected → reshape
- Transposed convolutions → reconstruct image

**Loss Function**
- Reconstruction Loss (MSE)
- KL Divergence

##⚙️ Tech Stack
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

##📂 Dataset<br>
**Research Work:**
- **Flickr8k Dataset**<br>
  - 8000 images
  - Each image has 5 captions

  **VAE:**
- **CIFAR-10 Dataset**<br>
  - 32x32 RGB images
  - 10 classes

##🚀 How to Run
1. **Clone the repository**<br>
git clone [https://github.com/himanshimittal/image-captioning-cnn-transformer](https://github.com/himanshimittal/image-captioning-cnn-transformer)<br>
cd your-repo
2. **Install dependencies**<br>
pip install torch torchvision matplotlib
3. **Run the VAE model**<br>
python vae.py

##📊 Output
- Training loss per epoch
- Original vs Reconstructed images

##📈 Results (Research Work)
- BLEU Scores:
  - BLEU-1: 0.72
  - BLEU-2: 0.61
  - BLEU-3: 0.52
  - BLEU-4: 0.46

##🧪 Sample Output (VAE)
- Input: CIFAR-10 images
- Output: Reconstructed images using latent space

##📌 Future Work
- Improve caption quality using larger datasets (MS COCO)
- Optimize transformer architecture
- Enhance VAE with better reconstruction loss (e.g., BCE)
- Combine VAE with captioning for generative tasks

##👩‍💻 Authors
- Diya Nanda<br>
- Hasika<br>
- Himanshi Mittal

##📜 License
This project is for academic and research purposes.

##⭐ Acknowledgements
- Flickr8k Dataset
- PyTorch Community
