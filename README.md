Emotion Detection from Text
This project focuses on building various machine learning and deep learning models to detect emotions from textual data.
Data Loading and Initial Exploration
The dataset train.txt containing 'text' and 'emotion' columns was loaded using pandas. Initial checks revealed:


Dataset shape: (16000, 2)
No missing values.
One duplicated row, which was not explicitly handled in the provided notebook but is a good practice to address.
Distribution of emotions was visualized using a count plot, showing varying frequencies for 'sadness', 'anger', 'love', 'surprise', 'fear', and 'joy'.


Text Preprocessing
To prepare the text data for modeling, the following preprocessing steps were applied:

Lowercasing: All text was converted to lowercase.
Punctuation Removal: Punctuation marks were removed from the text.
Number Removal: Digits were removed from the text.
Stopword Removal: Common English stopwords were removed using nltk.
Lemmatization: Words were reduced to their base forms using spaCy to standardize vocabulary.


Feature Extraction and Data Splitting
Label Encoding: The categorical 'emotion' labels were converted into numerical representations using LabelEncoder.
Word Count Analysis: A new feature 'no of words' was created to count the number of words in each text, and its distribution was visualized across different emotions.
TF-IDF Vectorization: Text data was transformed into numerical feature vectors using TfidfVectorizer with ngram_range=(1,2).
Train-Test Split: The dataset was split into training and testing sets with an 80/20 ratio and random_state=42.


Model Training and Evaluation
Several machine learning and deep learning models were trained and evaluated:

Traditional Machine Learning Models (using TF-IDF features)
Logistic Regression:
Accuracy: ~0.84
A classification report and confusion matrix were generated to assess performance.
Multinomial Naive Bayes:
Accuracy: ~0.67
A classification report and confusion matrix were generated.
Linear Support Vector Machine (SVM):
Accuracy: ~0.89
A classification report and confusion matrix were generated. SVM showed the highest accuracy among traditional models.
Deep Learning Models (using Word Embeddings)
For deep learning models, text data was tokenized, padded, and then fed into neural networks.

Artificial Neural Network (ANN):
A sequential model with two dense layers and dropout was trained on TF-IDF features.
Achieved a validation accuracy of ~0.82-0.83 after 20 epochs.
Simple Recurrent Neural Network (RNN):
An embedding layer followed by two SimpleRNN layers and dense layers.
Achieved a validation accuracy of ~0.70 after 20 epochs.
Bidirectional RNN (BiRNN):
An embedding layer followed by a Bidirectional SimpleRNN layer and dense layers.
Achieved a validation accuracy of ~0.76 after 20 epochs.
Long Short-Term Memory (LSTM) Network:
An embedding layer followed by an LSTM layer and dense layers.
Achieved a validation accuracy of ~0.86 after 20 epochs, making it the best performing deep learning model presented.
Overall Best Performing Model
The Linear Support Vector Machine (SVM) and LSTM Network demonstrated the best performance among the tested models, with SVM achieving an accuracy of approximately 0.89 on the test set and LSTM achieving a validation accuracy of approximately 0.86.
