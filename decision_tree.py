pip install scikit-learn matplotlib
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

# Sample dataset (You can replace this with your dataset)
data = {
    'Age': [25, 45, 35, 50, 23, 51, 36, 40, 45, 22],
    'Income': [50000, 64000, 58000, 60000, 52000, 68000, 54000, 61000, 62000, 59000],
    'Student': [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'Credit_rating': [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    'Buys_computer': [0, 1, 1, 1, 0, 1, 0, 1, 1, 0]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Age', 'Income', 'Student', 'Credit_rating']]  # Features
y = df['Buys_computer']  # Target variable

# Split dataset into training set and test set (80% training and 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create Decision Tree Classifier object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Step 3: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
