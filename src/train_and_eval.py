# src/train_and_eval.py
# %% [markdown]
# Tweet Emotion Detector - full pipeline

# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from utils import clean_tweet

# %% config
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'emotion_dataset.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42

# %% load data
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# expected columns: try to guess common variations
possible_text_cols = ['text', 'tweet', 'content']
possible_label_cols = ['label', 'emotion', 'sentiment']
text_col = next((c for c in possible_text_cols if c in df.columns), None)
label_col = next((c for c in possible_label_cols if c in df.columns), None)

if text_col is None or label_col is None:
    raise Exception(f"Couldn't find expected columns. Found: {df.columns.tolist()}")

df = df[[text_col, label_col]].dropna()
df.columns = ['text', 'label']
print("Sample label counts:")
print(df['label'].value_counts())

# Optionally reduce labels to the four required categories if dataset has more:
# We'll map similar labels to {happy, sad, angry, neutral} if present.
label_map = {}
# Try an automatic mapping based on keywords
for lbl in df['label'].unique():
    l = lbl.lower()
    if 'joy' in l or 'happy' in l or 'happiness' in l or 'love' in l:
        label_map[lbl] = 'happy'
    elif 'sad' in l or 'sadness' in l or 'depression' in l or 'gloom' in l:
        label_map[lbl] = 'sad'
    elif 'anger' in l or 'angry' in l or 'annoy' in l or 'rage' in l:
        label_map[lbl] = 'angry'
    elif 'neutral' in l or 'no emotion' in l or 'others' in l:
        label_map[lbl] = 'neutral'
    else:
        # default: map uncommon labels to neutral
        label_map[lbl] = 'neutral'

df['label'] = df['label'].map(label_map)

print("Remapped label distribution:")
print(df['label'].value_counts())

# %% preprocess
df['clean_text'] = df['text'].astype(str).apply(lambda x: clean_tweet(x))
df = df[df['clean_text'].str.strip() != ""]  # drop empty after cleaning
X = df['clean_text'].values
y = df['label'].values

# %% split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE)

# %% vectorize
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# %% train models
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
nb = MultinomialNB()

print("Training Logistic Regression...")
lr.fit(X_train_vec, y_train)
print("Training MultinomialNB...")
nb.fit(X_train_vec, y_train)

# %% evaluate helper
def evaluate_model(model, X_test_vec, y_test, name="model"):
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds, labels=['happy','sad','angry','neutral'])
    return cm, acc, preds

cm_lr, acc_lr, preds_lr = evaluate_model(lr, X_test_vec, y_test, name='Logistic Regression')
cm_nb, acc_nb, preds_nb = evaluate_model(nb, X_test_vec, y_test, name='MultinomialNB')

# %% choose best model by accuracy
best_model = lr if acc_lr >= acc_nb else nb
best_name = 'Logistic Regression' if best_model is lr else 'MultinomialNB'
print("Selected best model:", best_name)

# %% plot confusion matrix heatmap for best model
cm = cm_lr if best_model is lr else cm_nb
labels = ['happy','sad','angry','neutral']
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(outpath, dpi=150)
print("Saved confusion matrix to", outpath)

# %% save model and vectorizer
joblib.dump({'model': best_model, 'vectorizer': vectorizer}, os.path.join(OUTPUT_DIR, 'model.joblib'))
print("Saved model and vectorizer.")

# %% inspect some misclassifications (optional)
def show_misclassified(X_test, y_test, preds, n=10):
    dfm = pd.DataFrame({'text': X_test, 'true': y_test, 'pred': preds})
    mis = dfm[dfm['true'] != dfm['pred']].sample(frac=1, random_state=RANDOM_STATE).head(n)
    return mis

mis = show_misclassified(X_test, y_test, preds_lr if best_model is lr else preds_nb, n=10)
print("Example misclassifications:")
print(mis[['text','true','pred']].to_string(index=False))

