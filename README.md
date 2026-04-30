# 📌 Image Captioning using CNN and Transformers + VAE Implementation

## 📄 Research Overview

This project is based on our research paper titled:
**“Image Captioning using CNN and Transformers”**

The work focuses on generating **contextually meaningful captions from images** by combining:

* Convolutional Neural Networks (CNNs) for feature extraction
* Transformer-based architecture for sequence generation

We use VGG16 as the encoder and a Transformer decoder with attention mechanisms to generate captions. The model is trained on the Flickr8k dataset, achieving promising results in generating fluent and coherent captions.

---

## 💡 Key Contributions

* Hybrid CNN + Transformer architecture
* Multi-head attention for improved context understanding
* Image feature extraction using VGG16
* Evaluation using BLEU score metrics
* Additional implementation of a Variational Autoencoder (VAE) for image reconstruction

---

## 🧠 Code Overview

### 📷 Image Captioning (Research Work)

* CNN (VGG16) for feature extraction
* Transformer-based encoder-decoder
* Multi-head attention mechanism
* Caption generation pipeline

### 🔁 Variational Autoencoder (VAE)

Implemented using PyTorch for:

* Image compression
* Latent representation learning
* Image reconstruction (CIFAR-10 dataset)

---

## 🏗️ VAE Architecture

### Encoder

* Convolutional layers for feature extraction
* Fully connected layers for latent representation
* Outputs:

  * Mean (μ)
  * Log variance (log σ²)

### Sampling (Reparameterization Trick)

```
z = μ + ε * σ
```

### Decoder

* Fully connected layers → reshape
* Transposed convolutions → reconstruction

### Loss Function

* Reconstruction Loss (MSE)
* KL Divergence

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib

---

## 📂 Dataset

### Research Work

* **Flickr8k Dataset**

  * 8000 images
  * Each image has 5 captions

### VAE

* **CIFAR-10 Dataset**

  * 32×32 RGB images
  * 10 classes

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/himanshimittal/image-captioning-cnn-transformer
cd image-captioning-cnn-transformer
```

### 2. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 3. Run the VAE model

```bash
python vae.py
```

---

## 📊 Output

* Training loss per epoch
* Original vs reconstructed images

---

## 📈 Results (Research Work)

* BLEU-1: 0.72
* BLEU-2: 0.61
* BLEU-3: 0.52
* BLEU-4: 0.46

---

## 🧪 Sample Output (VAE)

* Input: CIFAR-10 images
* Output: Reconstructed images from latent space

---

## 📌 Future Work

* Improve caption quality using larger datasets (e.g., MS COCO)
* Optimize Transformer architecture
* Enhance VAE using better reconstruction loss (e.g., BCE)
* Explore integration of VAE with captioning

---

## 👩‍💻 Authors

* Diya Nanda
* Hasika
* Himanshi Mittal

---

## 📜 License

This project is intended for academic and research purposes.

---

## ⭐ Acknowledgements

* Flickr8k Dataset
* PyTorch Community
