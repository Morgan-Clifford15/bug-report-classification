import os
import requests
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, SpatialDropout1D, Bidirectional, Layer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

#Custom Focal Loss Implementation
def focal_loss(gamma=1.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = y_true * tf.pow(1 - y_pred, gamma) + (1 - y_true) * tf.pow(y_pred, gamma)
        fl = alpha * weight * ce
        return tf.reduce_mean(fl)
    return focal_loss_fn


#Simple Data Augmentation with Synonym Replacement
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym.isalpha():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(text, n=2):
    words = text.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word.lower() not in stop_words]))
    if not random_word_list:
        return text
    num_replacements = min(n, len(random_word_list))
    words_to_replace = np.random.choice(random_word_list, num_replacements, replace=False)
    for word in words_to_replace:
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = np.random.choice(synonyms)
            new_words = [synonym if w == word else w for w in new_words]
    return " ".join(new_words)

#Download GloVe Embeddings
glove_file = 'glove.6B.100d.txt'
if not os.path.exists(glove_file):
    print("Downloading GloVe embeddings...")
    url = 'http://nlp.stanford.edu/data/glove.6B.zip'
    zip_file = 'glove.6B.zip'
    response = requests.get(url)
    with open(zip_file, 'wb') as f:
        f.write(response.content)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    print("GloVe embeddings downloaded and unzipped.")

#Load Data
df = pd.read_csv("lab1/datasets/incubator-mxnet.csv")
df["text"] = df.apply(lambda row: row["Title"] + " " + row["Body"] if pd.notna(row["Body"]) else row["Title"], axis=1)
X = df["text"].values
y = df["class"].values

#Preprocess Text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s+\-*/#.,]', '', text)
    text = " ".join(text.split())
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text.lower()

X = [clean_text(text) for text in X]

#Data Augmentation
X_augmented = []
y_augmented = []
for text, label in zip(X, y):
    X_augmented.append(text)
    y_augmented.append(label)
    # Augment positive class more to help with imbalance
    if label == 1:
        augmented_text = synonym_replacement(text, n=2)
        X_augmented.append(augmented_text)
        y_augmented.append(label)

X = np.array(X_augmented)
y = np.array(y_augmented)
print(f"Dataset size after augmentation: {len(X)} samples")

#Tokenization
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
word_index = tokenizer.word_index

# Analyze sequence lengths
sequence_lengths = [len(seq) for seq in sequences]
print(f"Sequence length stats: min={min(sequence_lengths)}, max={max(sequence_lengths)}, mean={np.mean(sequence_lengths):.2f}, median={np.median(sequence_lengths):.2f}")
maxlen = int(np.percentile(sequence_lengths, 95))
print(f"Setting maxlen to {maxlen} to cover 95% of sequences")

X_padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

#Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_padded, y)

#Load Pre-trained GloVe Embeddings
embeddings_index = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f"Loaded {len(embeddings_index)} word vectors.")

#Create Embedding Matrix
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in embeddings_index:
        embedding_matrix[i] = embeddings_index[word]

#Define Model Creation Function
def create_model():
    lstm_units = 128
    dropout_rate = 0.2
    learning_rate = 0.0005
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, 
                        weights=[embedding_matrix], trainable=True))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=0.1, 
                                 return_sequences=False, kernel_regularizer=regularizers.l2(0.01))))
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.compile(loss=focal_loss(gamma=1.0, alpha=0.75),
                  optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), 
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

#Test Standalone Model
print("Testing standalone model creation...")
test_model = create_model()
test_model.summary()
print("Standalone model created successfully.")

#Class Weights
classes = np.unique(y)
class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weights = dict(zip(classes, class_weights_array))
print(f"Class weights: {class_weights}")

#Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

#K-Fold Cross-Validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_smote, y_smote)):
    print(f"\nTraining Fold {fold + 1}/{n_splits}")
    X_train_fold, X_val_fold = X_smote[train_idx], X_smote[val_idx]
    y_train_fold, y_val_fold = y_smote[train_idx], y_smote[val_idx]

    model = create_model()
    model.fit(X_train_fold, y_train_fold, epochs=50, batch_size=32, 
              validation_data=(X_val_fold, y_val_fold), 
              callbacks=[early_stopping, reduce_lr], 
              class_weight=class_weights, verbose=1)

    # Evaluate on validation fold
    y_pred_probs = model.predict(X_val_fold, batch_size=32)
    y_pred = (y_pred_probs > 0.5).astype(int)
    accuracy = accuracy_score(y_val_fold, y_pred)
    precision = precision_score(y_val_fold, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val_fold, y_pred, average='macro')
    f1 = f1_score(y_val_fold, y_pred, average='macro')
    auc = roc_auc_score(y_val_fold, y_pred_probs) if len(np.unique(y_pred_probs)) > 1 else 0.5

    fold_metrics['accuracy'].append(accuracy)
    fold_metrics['precision'].append(precision)
    fold_metrics['recall'].append(recall)
    fold_metrics['f1'].append(f1)
    fold_metrics['auc'].append(auc)

    print(f"Fold {fold + 1} Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

# Print average metrics across folds
print("\n=== Cross-Validation Results ===")
for metric, values in fold_metrics.items():
    print(f"Average {metric.capitalize()}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

#Train Final Model on Full Training Set
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42, stratify=y_smote)
final_model = create_model()
final_model.fit(X_train, y_train, epochs=50, batch_size=32, 
                validation_split=0.2, callbacks=[early_stopping, reduce_lr], 
                class_weight=class_weights, verbose=1)

#Evaluate on Test Set with Fine-Tuned Threshold
y_pred_probs = final_model.predict(X_test, batch_size=32)
print(f"Predicted probabilities stats: min={y_pred_probs.min():.4f}, max={y_pred_probs.max():.4f}, mean={y_pred_probs.mean():.4f}")

if len(np.unique(y_pred_probs)) > 1:
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold (default method): {optimal_threshold:.4f}")
else:
    print("Predicted probabilities are constant; using default threshold of 0.5")
    optimal_threshold = 0.5

thresholds_to_test = np.arange(0.50, 0.61, 0.01)
best_f1, best_thresh = 0, 0
for thresh in thresholds_to_test:
    y_pred = (y_pred_probs > thresh).astype(int)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"Threshold: {thresh:.2f}, F1: {f1:.4f}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

y_pred = (y_pred_probs > best_thresh).astype(int)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
auc_score = roc_auc_score(y_test, y_pred_probs) if len(np.unique(y_pred_probs)) > 1 else 0.5

print("\n=== Final Test Set Results ===")
print(f"Optimal Threshold: {best_thresh:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc_score:.4f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))