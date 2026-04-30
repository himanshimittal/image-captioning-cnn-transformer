# Image Captioning using CNN and Transformer Architecture

## 📄 Research Overview

This project focuses on the problem of automatic image captioning, which lies at the intersection of computer vision and natural language processing. The objective is to generate meaningful textual descriptions for images by understanding visual content and modeling sequential language.

The proposed approach combines a Convolutional Neural Network (CNN) for feature extraction with a Transformer-based architecture for sequence generation. A pre-trained VGG16 model is used to extract image features, while the Transformer decoder generates context-aware captions using attention mechanisms.

---

## 💻 Code Overview

The code implements an end-to-end image captioning pipeline:

* Image feature extraction using a pre-trained VGG16 model
* Text preprocessing including tokenization and sequence padding
* Transformer-based decoder for caption generation
* Training using paired image-caption data from the Flickr8k dataset
* Caption prediction for new input images

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash id="y3v7zz"
pip install tensorflow keras numpy matplotlib
```

### 2. Prepare Dataset

* Download and place the Flickr8k dataset in the project directory
* Ensure image paths and caption files are correctly configured

### 3. Train the Model

```bash id="fq6qoe"
python train.py
```

### 4. Generate Captions

```bash id="l2z4w2"
python predict.py
```

---

## 📊 Results

The model is capable of generating relevant and coherent captions for input images by capturing both visual features and contextual relationships in text.

**Example:**

* Ground Truth:
  A dog is running through a grassy field

* Generated Caption:
  A dog runs across the grass

Evaluation is performed using BLEU scores to measure the quality of generated captions.

---

## 📌 Notes

### Limitations

* Performance is limited by the size of the Flickr8k dataset
* Captions may be generic for complex scenes
* Training requires moderate computational resources

### Future Work

* Use larger datasets such as MS COCO
* Improve caption diversity and accuracy
* Incorporate advanced architectures such as Vision Transformers
* Apply beam search for better sequence generation

---
