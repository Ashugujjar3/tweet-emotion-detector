from sklearn.metrics import classification_report
import pandas as pd
cr = classification_report(y_test, preds, output_dict=True)
pd.DataFrame(cr).transpose().to_csv(os.path.join(OUTPUT_DIR, 'classification_report.csv'))

# Save misclassified examples
dfm = pd.DataFrame({'text': X_test, 'true': y_test, 'pred': preds})
mis = dfm[dfm['true'] != dfm['pred']]
mis.to_csv(os.path.join(OUTPUT_DIR, 'misclassified_examples.csv'), index=False)
