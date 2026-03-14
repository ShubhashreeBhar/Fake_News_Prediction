# 📰 Fake News Detection using Machine Learning
Fake news has become a major issue in the digital world. This project builds a **Machine Learning model that detects whether a news article is REAL or FAKE** using Natural Language Processing (NLP).

The system processes textual news data, converts it into numerical features, and then uses **Logistic Regression** to classify the news.

This project demonstrates the use of **text preprocessing, TF-IDF vectorization, and supervised learning** for fake news classification.

---
## 🚀 Features

* Text preprocessing using NLP techniques
* Stopword removal
* Stemming using Porter Stemmer
* Feature extraction using TF-IDF
* Machine Learning model using Logistic Regression
* Training and testing evaluation
* Fake news prediction system

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* NLTK
* Scikit-learn

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading

The dataset is loaded using **Pandas**.

```python
dataset = pd.read_csv('news.csv.zip')
```

---

### 2️⃣ Data Preprocessing

Text data is cleaned using:

* Removing special characters
* Converting text to lowercase
* Tokenization
* Removing stopwords
* Stemming

Example preprocessing:

```python
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)
```

---

### 3️⃣ Feature Extraction

Text is converted into numerical features using **TF-IDF Vectorization**.

```python
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
```

TF-IDF helps identify important words in the document.

---

### 4️⃣ Train Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
```

* **80% Training data**
* **20% Testing data**

---

### 5️⃣ Model Training

A **Logistic Regression** model is used for classification.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

### 6️⃣ Model Evaluation

Accuracy on training data:

```
Training Accuracy: 90.78%
```

Accuracy on testing data:

```
Testing Accuracy: 83.34%
```

This indicates the model generalizes reasonably well.

---

## 🔮 Prediction System

The model can classify new news articles as **REAL or FAKE**.

Example:

```python
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("The news is real")
else:
    print("The news is fake")
```

---

## 📊 Model Performance

| Metric            | Score  |
| ----------------- | ------ |
| Training Accuracy | 90.78% |
| Testing Accuracy  | 83.34% |


---

