from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score
import numpy as np

# Load the dataset
data = load_iris()
X, y = data.data, data.target

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(max_depth=3)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(decision_tree_classifier, X, y, cv=5)

# Fit the classifier on the entire dataset
decision_tree_classifier.fit(X, y)

# Generate and display the rules of the Decision Tree classifier
tree_rules = export_text(decision_tree_classifier, feature_names=data.feature_names)
print("Decision Tree Rules:\n", tree_rules)

# Print the cross-validation scores
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
