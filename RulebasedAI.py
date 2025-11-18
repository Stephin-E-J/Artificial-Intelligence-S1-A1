import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


# Load dataset

url = "https://raw.githubusercontent.com/Stephin-E-J/Artificial-Intelligence-S1-A1/refs/heads/main/Chart%20Details/creditrisk.csv"
data = pd.read_csv(url)



# Create synthetic Credit_Risk column

def assign_credit_risk(row):
    if row['Credit_History'] == 'Good' and row['Income'] >= 10000 and row['Loan_Amount'] < 400000:
        return 2   # Low risk
    elif row['Credit_History'] == 'Average' or (10000 <= row['Income'] < 60000) or (400000 <= row['Loan_Amount'] <= 700000):
        return 1   # Medium risk
    else:
        return 0   # High risk

data['Credit_Risk'] = data.apply(assign_credit_risk, axis=1)

print("Synthetic target column 'Credit_Risk' created successfully!\n")
print(data[['Income', 'Loan_Amount', 'Credit_History', 'Credit_Risk']].head())


# Encode categorical variables

for col in ['Employment_status', 'Purpose', 'Credit_History']:
    data[col] = data[col].astype('category').cat.codes

# Drop non-numeric/non-predictive fields
data = data.drop(columns=['ID', 'Date', 'Name'])


# Split data

X = data.drop(columns=['Credit_Risk'])
y = data['Credit_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Rule-Based AI Models
    
class   RuleBasedCreditAI:
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
            return 0

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
            return 0


# Instantiate models and predict

model1 = RuleBasedCreditAI("Basic Rule-Based Credit Model")
model2 = AdvancedRuleBasedCreditAI("Advanced Rule-Based Credit Model")

y_pred_1 = model1.predict_dataframe(X_test)
y_pred_2 = model2.predict_dataframe(X_test)


# Evaluation

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='macro') * 100
    recall = recall_score(y_true, y_pred, average='macro') * 100
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    return {
        "Model": model_name,
        "Accuracy (%)": round(accuracy, 2),
        "Precision (%)": round(precision, 2),
        "Recall (%)": round(recall, 2),
        "F1 Score (%)": round(f1, 2)
    }

results = [
    evaluate_model(y_test, y_pred_1, model1.name),
    evaluate_model(y_test, y_pred_2, model2.name)
]

results_df = pd.DataFrame(results)
print("\n📊 Performance Comparison of Rule-Based Credit Risk AI Models:\n")
print(results_df)


# Visualization: Performance

metrics = ['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1 Score (%)']
x = range(len(metrics))
plt.figure(figsize=(8,6))
for i, row in results_df.iterrows():
    plt.bar([p + i*0.3 for p in x], row[metrics], width=0.3, label=row['Model'])

plt.xticks([p + 0.3/2 for p in x], metrics)
plt.ylabel("Percentage (%)")
plt.title("Rule-Based Credit Risk Models Comparison")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Confusion Matrices

cm1 = confusion_matrix(y_test, y_pred_1)
cm2 = confusion_matrix(y_test, y_pred_2)

plt.figure(figsize=(6,5))
ConfusionMatrixDisplay(confusion_matrix=cm1).plot(cmap='Blues')
plt.title("Confusion Matrix - Basic Model")
plt.show()

plt.figure(figsize=(6,5))
ConfusionMatrixDisplay(confusion_matrix=cm2).plot(cmap='Greens')
plt.title("Confusion Matrix - Advanced Model")
plt.show()


# ROC-AUC Curves

classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)
y_pred_1_bin = label_binarize(y_pred_1, classes=classes)
y_pred_2_bin = label_binarize(y_pred_2, classes=classes)

roc_auc_1 = roc_auc_score(y_test_bin, y_pred_1_bin, average='macro', multi_class='ovr')
roc_auc_2 = roc_auc_score(y_test_bin, y_pred_2_bin, average='macro', multi_class='ovr')

print(f"\nROC-AUC Score (Macro OVR) - {model1.name}: {roc_auc_1:.2f}")
print(f"ROC-AUC Score (Macro OVR) - {model2.name}: {roc_auc_2:.2f}")

class_index = len(classes) - 1
RocCurveDisplay.from_predictions(
    y_test_bin[:, class_index],
    y_pred_1_bin[:, class_index],
    name=f"{model1.name}",
    color="darkorange"
)
RocCurveDisplay.from_predictions(
    y_test_bin[:, class_index],
    y_pred_2_bin[:, class_index],
    name=f"{model2.name}",
    color="green"
)
plt.title(f"ROC Curve (Class: {classes[class_index]})")
plt.show()
