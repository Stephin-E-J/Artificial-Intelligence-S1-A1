import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/Stephin-E-J/Artificial-Intelligence-S1-A1/refs/heads/main/Chart%20Details/creditrisk.csv"
data = pd.read_csv(url)

# Create a Credit_Risk column
def credit_risk(row):
    if row['Credit_History'] == 'Good' and row['Income'] >= 10000 and row['Loan_Amount'] < 400000:
        return 2   # Low risk
    elif row['Credit_History'] == 'Average' or (10000 <= row['Income'] < 60000) or (400000 <= row['Loan_Amount'] <= 700000):
        return 1   # Medium risk
    else:
        return 0   # High risk

data['Credit_Risk'] = data.apply(credit_risk, axis=1)

for col in ['Employment_status', 'Purpose', 'Credit_History']:
    data[col] = data[col].astype('category').cat.codes

data = data.drop(columns=['ID', 'Date', 'Name'])


X = data.drop(columns=['Credit_Risk'])
y = data['Credit_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Rule-Based AI Models
class RuleBasedCreditAI:
    def __init__(self, name):
        self.name = name

    def predict(self, row):
        try:
            if row['Income'] > 60000 and row['Credit_History'] == 2:  # 'Good'
                return 2
            elif row['Income'] > 40000 and row['Loan_Amount'] < 500000:
                return 1
            else:
                return 0
        except:
            return 1  

    def predict_dataframe(self, X):
        return X.apply(self.predict, axis=1)

class AdvancedRuleBasedCreditAI(RuleBasedCreditAI):
    def predict(self, row):
        try:
            if row['Credit_History'] == 2 and row['Loan_Amount'] < 400000:
                if row['Income'] >= 70000:
                    return 2
                elif row['Income'] >= 50000:
                    return 1
                else:
                    return 0
            elif row['Credit_History'] == 1:
                if row['Income'] >= 50000 and row['Loan_Amount'] < 600000:
                    return 1
                else:
                    return 0
            else:
                return 0
        except:
            return 1 

# Instantiate models and predict
model1 = RuleBasedCreditAI("Basic Model")
model2 = AdvancedRuleBasedCreditAI("Advanced Model")

y_pred_1 = model1.predict_dataframe(X_test)
y_pred_2 = model2.predict_dataframe(X_test)

# Evaluation
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    return {
        "Model": model_name,
        "Accuracy": round(accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2)
    }

results = [
    evaluate_model(y_test, y_pred_1, model1.name),
    evaluate_model(y_test, y_pred_2, model2.name)
]

results_df = pd.DataFrame(results)

plt.figure(figsize=(10, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
basic_scores = [results_df.loc[0, metric] for metric in metrics]
advanced_scores = [results_df.loc[1, metric] for metric in metrics]

x = range(len(metrics))
bar_width = 0.35

plt.bar([i - bar_width/2 for i in x], basic_scores, bar_width, label='Basic Model', color='blue', alpha=0.7)
plt.bar([i + bar_width/2 for i in x], advanced_scores, bar_width, label='Advanced Model', color='red', alpha=0.7)

plt.xlabel('Metrics')
plt.ylabel('Score (%)')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

cm1 = confusion_matrix(y_test, y_pred_1)
cm2 = confusion_matrix(y_test, y_pred_2)

ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=['High', 'Medium', 'Low']).plot(ax=ax1, cmap='Blues')
ax1.set_title("Basic Model - Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=['High', 'Medium', 'Low']).plot(ax=ax2, cmap='Reds')
ax2.set_title("Advanced Model - Confusion Matrix")

plt.tight_layout()
plt.show()

# ROC-AUC Curves
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
y_pred_1_bin = label_binarize(y_pred_1, classes=classes)
y_pred_2_bin = label_binarize(y_pred_2, classes=classes)

roc_auc_1 = roc_auc_score(y_test_bin, y_pred_1_bin, average='macro', multi_class='ovr')
roc_auc_2 = roc_auc_score(y_test_bin, y_pred_2_bin, average='macro', multi_class='ovr')


plt.figure(figsize=(8, 6))
class_idx = len(classes) - 1  

RocCurveDisplay.from_predictions(
    y_test_bin[:, class_idx],
    y_pred_1_bin[:, class_idx],
    name=f"Basic Model (AUC = {roc_auc_1:.3f})",
    color="blue"
)
RocCurveDisplay.from_predictions(
    y_test_bin[:, class_idx],
    y_pred_2_bin[:, class_idx],
    name=f"Advanced Model (AUC = {roc_auc_2:.3f})",
    color="red"
)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Low Risk Classification')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx, 'Model']
best_accuracy = results_df.loc[best_model_idx, 'Accuracy']