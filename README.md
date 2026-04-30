# Variational Autoencoder (VAE) for Image Reconstruction + Deep Learning Research

## 📌 Overview

This repository contains an implementation of a **Variational Autoencoder (VAE)** built using PyTorch for image reconstruction on the CIFAR-10 dataset, along with conceptual grounding inspired by research in **computer vision and deep learning architectures**.

Additionally, this work is aligned with research on **image understanding and generation**, including concepts explored in CNN-based feature extraction and Transformer-based modeling.

---

## 🚀 Features

* Variational Autoencoder (VAE) implementation from scratch
* Encoder-Decoder architecture using convolutional layers
* Latent space representation learning
* Image reconstruction visualization
* Training on CIFAR-10 dataset
* GPU support (CUDA if available)

---

## 🧠 Model Architecture

### 🔹 Encoder

* Convolutional layers for feature extraction
* Fully connected layers to generate:

  * Mean (`μ`)
  * Log variance (`log σ²`)

### 🔹 Latent Space

* Reparameterization trick:

  ```
  z = μ + σ * ε
  ```

* Enables stochastic sampling while keeping backpropagation valid

### 🔹 Decoder

* Fully connected + transposed convolution layers
* Reconstructs image from latent vector

---

## 📉 Loss Function

The VAE uses a combination of:

1. **Reconstruction Loss (MSE/BCE)**
2. **KL Divergence Loss**

```
Loss = Reconstruction Loss + KL Divergence
```

* Reconstruction ensures output resembles input
* KL divergence regularizes latent space

---

## 📊 Dataset

* **CIFAR-10**

  * 60,000 images (32×32 RGB)
  * 10 classes

> Note: When running on Kaggle, dataset is loaded from local input path instead of downloading.

---

## 🖼️ Results

### Reconstruction Visualization

The model reconstructs input images after encoding and decoding:

* Top row → Original images
* Bottom row → Reconstructed images

```python
def plot_reconstruction(vae, data_loader):
    vae.eval()
    with torch.no_grad():
        x, _ = next(iter(data_loader))
        x = x.to(device)
        reconstructed, _, _ = vae(x)

        x = x.cpu()
        reconstructed = reconstructed.cpu()

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            axes[0, i].imshow(x[i].permute(1, 2, 0))
            axes[0, i].axis('off')

            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
            axes[1, i].axis('off')

        plt.show()
```

---

## ⚙️ Training Details

| Parameter     | Value |
| ------------- | ----- |
| Epochs        | 50    |
| Batch Size    | 32    |
| Learning Rate | 0.001 |
| Optimizer     | Adam  |
| Latent Dim    | 64    |

---

## 📚 Research Connection

This implementation connects with research in:

### 🔹 Computer Vision + NLP Integration

As explored in the research paper:

> *"Image Captioning using CNN and Transformers"*

Key insights:

* CNNs (e.g., VGG16) extract visual features
* Transformers handle sequence generation
* Attention mechanisms improve contextual understanding

### 🔹 Relevance to this Project

* VAE learns **compressed latent representations**
* Similar to feature extraction in CNNs
* Can be extended for:

  * Image generation
  * Feature learning for captioning systems
  * Multimodal AI systems

---

## 🔮 Future Improvements

* Replace MSE with Binary Cross Entropy (better for images)
* Use deeper convolutional architecture
* Visualize latent space using PCA / t-SNE
* Extend to Conditional VAE (CVAE)
* Integrate with caption generation models

---

## 🛠️ How to Run

### 1. Install dependencies

```bash
pip install torch torchvision matplotlib
```

### 2. Run training

```bash
python train.py
```

### 3. Visualize results

```bash
python visualize.py
```
---

## 👩‍💻 Author

**Himanshi**

---

## ⭐ Acknowledgements

* CIFAR-10 Dataset
* PyTorch
* Research inspiration from CNN + Transformer architectures for image understanding

---

## 📌 Conclusion

This project demonstrates how generative models like VAE can:

* Learn meaningful latent representations
* Reconstruct images effectively
* Serve as a foundation for advanced tasks like image generation and multimodal AI
