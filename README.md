# 📰 Fake News Detection Using CNN

## 📌 Project Overview

Fake News Detection Using CNN is a Deep Learning project that classifies news articles as **Real** or **Fake** using a Convolutional Neural Network (CNN). The model is trained on a labeled dataset of real and fake news articles and deployed with an interactive **Streamlit** web application.

The application allows users to paste a news article and instantly receive a prediction along with the model's confidence score.

---

## 🚀 Features

* Detects whether a news article is **Real** or **Fake**
* Deep Learning model built using **TensorFlow/Keras**
* CNN architecture for text classification
* Text preprocessing with tokenization and padding
* Interactive Streamlit web interface
* Displays prediction confidence
* Saved model for future predictions

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Matplotlib
* Seaborn
* Pickle

---

## 📂 Project Structure

```
Fake-News-Detection-Using-CNN/

│── app.py
│── FakeNewsDetectionByCNN.ipynb
│── fake_news_cnn_model.keras
│── tokenizer.pkl
│── Fake.csv
│── True.csv
│── requirements.txt
│── README.md
```

---

## 📊 Dataset

The project uses the Fake and True News dataset containing thousands of labeled news articles.

* Fake.csv
* True.csv

Each article is assigned a label:

* Fake News → 0
* Real News → 1

---

## 🔄 Workflow

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Text Preprocessing
5. Tokenization
6. Sequence Padding
7. Train-Test Split
8. CNN Model Building
9. Model Training
10. Model Evaluation
11. Save Model
12. Deploy with Streamlit

---

## 🧠 CNN Architecture

* Embedding Layer
* Conv1D Layer
* MaxPooling1D Layer
* Dropout Layer
* GlobalMaxPooling1D Layer
* Dense Layer
* Output Layer (Sigmoid)

---

## 📈 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/shadowblade17/Fake-News-Detection.git
```

Move to the project folder:

```bash
cd Fake-News-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 💻 Application Interface

The application provides:

* Text input area
* Predict button
* Prediction result
* Confidence score
* Real/Fake indication

---

## 📌 Future Improvements

* Fine-tune the CNN model
* Use BERT or RoBERTa for improved accuracy
* Integrate real-time news APIs
* Support multiple languages
* Add explainable AI visualizations

---

## 👨‍💻 Author

**Shivam Chauhan**

* MCA Graduate
* Aspiring Data Scientist / Machine Learning Engineer

GitHub: https://github.com/shadowblade17

LinkedIn: Add your LinkedIn profile here.

---

## 📄 License

This project is developed for educational and learning purposes.
