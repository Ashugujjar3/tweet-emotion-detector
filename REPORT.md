```markdown
# Tweet Emotion Detector — Run Report

## Date:
YYYY-MM-DD

## Dataset
- Filename: `data/emotion_dataset.csv`
- # rows: 
- Column names used: text = `...`, label = `...`
- Label mapping performed (if any): e.g., joy -> happy, sadness -> sad

## Environment
- Python version: 
- Packages (from `pip freeze`): 

## Model(s) run
- Vectorizer: `TfidfVectorizer` (params: max_features=..., ngram_range=...)
- Classifiers: LogisticRegression / MultinomialNB
- Random seed: 42

## Results (fill after run)
### Logistic Regression
- Test accuracy: 
- Precision (macro):
- Recall (macro):
- F1 (macro):

### MultinomialNB
- Test accuracy: 
- Precision (macro):
- Recall (macro):
- F1 (macro):

### Selected best model
- Model name:
- Reason for selection:

## Confusion matrix
- File: `outputs/confusion_matrix.png`

## Classification report
- File: `outputs/classification_report.csv` (or paste key rows here)

## Example misclassifications (paste 5)
| text | true | pred |
|------|------|------|
| ...  | ...  | ...  |

## Observations & notes
- Common confusions:
- Data quality issues:
- Ideas to improve (e.g., more data, transformer embeddings, data augmentation):

## Next steps
- (e.g.) Run GridSearchCV for LR `C` param
- Try transformer embeddings (HuggingFace)
- Add a simple web UI (Streamlit)
