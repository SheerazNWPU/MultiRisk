from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
import os
import shutil
import sys
import datetime
import time
from collections import Counter

import config
import numpy as np
import pandas as pd
import rules_process
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import config



cfg = config.Configuration(1)
cfg.tvt_selection = 15

# Create a decision tree classifier
clf = DecisionTreeClassifier()

rule_out_info = open(cfg.get_parent_path() + "decision_tree_rules_info.txt", "w")
rule_out_info.write(
        "### {} ###".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
)
df = pd.read_csv(cfg.get_raw_data_path())
pairs = df.values
print("- # of data: {}.".format(len(pairs)))
id_2_pair_info = dict()
for elem in pairs:
    id_2_pair_info[elem[0]] = elem  # <id, info>

# Train data information
train_data_info = []
get_feature_name = pd.read_csv(cfg.get_parent_path() + "train.csv")
feature_names = get_feature_name.columns.tolist()[2:]
print(feature_names) 
train_info = pd.read_csv(cfg.get_parent_path() + "train.csv").values
train_ids = train_info[:, 0]  ######wang delete
# train_ids = pairs[:, 0].astype(str)   ###wang add
msg = "- # of train data: {}".format(len(train_ids))
print(msg)
rule_out_info.write(msg + "\n")
for train_id in train_ids:
    train_data_info.append(id_2_pair_info.get(train_id))
train_data_info_df = pd.DataFrame(train_data_info, columns=df.columns)
# train_data_info_df.to_csv('hello.csv', index=False)
train_data_X = np.array(train_data_info)[:, 2].reshape(-1, 7).astype(np.float32)
#print(train_data_X)  
train_labels = np.array(train_data_info)[:, 1].reshape(-1, 7).astype(np.float32)
#print(feature_names) 
#classNames = 
class_names = ['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC' ]

# Train the classifier on your training data
X_train = ...  # Your training features
y_train = ...  # Your training labels
clf.fit(train_data_X, train_labels)

# Extract one-sided decision tree rules for each class
n_classes = len(clf.classes_)
print("The number of class for this data are: ")
print(n_classes)
#class_names =clf.classes_
#print(class_Names)
tree = clf.tree_
#class_index = classifier.classes_


#def extract_rules(tree, feature_names, class_index, threshold=0.5, path=[]):
#    if tree.value[0, class_index] >= threshold:
#        rule = " AND ".join([f"{feature_names[i]} <= {tree.threshold[i]}" for i in path])
#        return [rule]
#    
#    rules = []
#    if tree.children_left[class_index] != _tree.TREE_LEAF:
#        path.append(tree.feature[class_index])
#        rules += extract_rules(tree, feature_names, class_index, threshold, path)
#        path.pop()
#        path.append(tree.feature[class_index] + 1)
#       rules += extract_rules(tree, feature_names, class_index, threshold, path)
#        path.pop()
#    
#    return rules
print(clf.feature_importances_)
def extract_decision_rules(tree, feature_names, class_names, node=0, depth=0, rules=None):
    if rules is None:
        rules = []
    print(tree.feature)
    if tree.feature[node] != _tree.TREE_UNDEFINED:
        print(node)
        feature = feature_names[tree.feature[node]]
        threshold = tree.threshold[node]
        impurity = tree.impurity[node]
        if impurity > 0:  # Adjust the impurity threshold as needed
            rules.append((depth, feature, "<=", threshold))
            extract_decision_rules(tree, feature_names, class_names, tree.children_left[node], depth + 1, rules)
            rules.append((depth, feature, ">", threshold))
            extract_decision_rules(tree, feature_names, class_names, tree.children_right[node], depth + 1, rules)
        else:
            class_id = np.argmax(tree.value[node])
            class_name = class_names[class_id]
            rules.append((depth, class_name))
    else:
        class_id = np.argmax(tree.value[node])
        class_name = class_names[class_id]
        rules.append((depth, class_name))

    return rules

# Set the impurity threshold
impurity_threshold = 0.1

# Extract decision rules from the trained decision tree
decision_rules = extract_decision_rules(clf.tree_, feature_names, class_names)
#print(decision_rules)
# Save decision rules to a text file
with open("decision_rules.txt", "w") as file:
    line = ""
    for rule in decision_rules:
        if len(rule) == 4:
            
            depth, feature, operator, threshold = rule
            line += f"{feature}{operator}{threshold:.2f}|{depth}|and|"
        else:
            #print(rule)
            depth, class_name = rule
            class_index = class_names.index(class_name)
            line = line[:-3]
            line += f":{class_name}({class_index})\n"
            file.write(line)
            line = ""

# Count the number of rules
num_rules = len(decision_rules)
print(f"Number of rules: {num_rules}")
# -- deduplicate rules;
cleaned_rules = []
#cleaned_rules = rules_process.clean_rules(one_sided_homogeneous_rules, True)
print(cleaned_rules) 
#for class_index in range(n_classes):
#   rules = extract_rules(tree, feature_names, class_names, node=0)
#    cleaned_rules.append(rules)

# Clean the rules by removing redundancies and simplifying them
#def clean_rules(rules):
#    cleaned = []
#    for rule in rules:
#        # Add your cleaning logic here
#        cleaned.append(rule)
#    return cleaned

#cleaned_rules = [clean_rules(rules) for rules in cleaned_rules]

# Print the cleaned one-sided homogeneous decision tree rules for each class
#for class_index, rules in enumerate(cleaned_rules):
#    print(f"Rules for Class {class_index}:")
#    for rule in rules:
#        print(rule)
#    print()

def generate_rules():
    # global match_tree_split_threshold, unmatch_tree_split_threshold, match_rule_threshold, unmatch_rule_threshold
    # All data information
    rule_out_info = open(cfg.get_parent_path() + "decision_tree_rules_info.txt", "w")
    rule_out_info.write(
        "### {} ###".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    df = pd.read_csv(cfg.get_raw_data_path())
    pairs = df.values
    print("- # of data: {}.".format(len(pairs)))
    id_2_pair_info = dict()
    for elem in pairs:
        id_2_pair_info[elem[0]] = elem  # <id, info>

    # Train data information
    train_data_info = []
    train_info = pd.read_csv(cfg.get_parent_path() + "train.csv").values
    print(train_info) 
    train_ids = train_info[:, 0]  ######wang delete
    # train_ids = pairs[:, 0].astype(str)   ###wang add
    msg = "- # of train data: {}".format(len(train_ids))
    print(msg)
    rule_out_info.write(msg + "\n")
    for train_id in train_ids:
        train_data_info.append(id_2_pair_info.get(train_id))
    train_data_info_df = pd.DataFrame(train_data_info, columns=df.columns)
    # train_data_info_df.to_csv('hello.csv', index=False)
    train_labels = np.array(train_data_info)[:, 1]
    #print(train_labels) 
    label_counts = Counter(train_labels)
    train_impurity = 1.0
    for k, v in label_counts.items():
        train_impurity -= (1.0 * v / len(train_labels)) ** 2
    msg = "- class size of train data: {}, impurity of train data: {}".format(
        label_counts, train_impurity
    )
    print(msg)
    rule_out_info.write(msg + "\n")

    # Set threshold according to the impurity of training data.
    # match_tree_split_threshold = 0.1
    # unmatch_tree_split_threshold = 0.01
    # match_tree_split_threshold = 0.3
    # unmatch_tree_split_threshold = 0.015
    # match_rule_threshold = match_tree_split_threshold
    # unmatch_rule_threshold = unmatch_tree_split_threshold

    # Print configuration
    msg = ""
    msg += "- max_tree_depth: {}\n".format(max_tree_depth)
    # msg += '- min_samples_leaf: {}\n'.format(min_samples_leaf)
    msg += "- match_tree_split_threshold: {}\n".format(match_tree_split_threshold)
    msg += "- unmatch_tree_split_threshold: {}\n".format(unmatch_tree_split_threshold)
    msg += "- match_rule_threshold: {}\n".format(match_rule_threshold)
    msg += "- unmatch_rule_threshold: {}\n".format(unmatch_rule_threshold)
    print(msg)
    rule_out_info.write(msg)

    # Generate rules using decision tree.
    ratree_ = ratree(train_data_info_df)
    ratree_.createfeaturetrees()

    # Read rules and clean them.
    read_rules = rules_process.read_rules(cfg.get_raw_decision_tree_rules_path())
    # -- select low impurity rules;
    cleaned_rules = rules_process.select_rules_based_on_threshold(
        read_rules, match_rule_threshold, unmatch_rule_threshold
    )
    # -- deduplicate rules;
    cleaned_rules = rules_process.clean_rules(cleaned_rules, True)
    # multiple processes version.
    cleaned_rules = rules_process.clean_rules_mt(cleaned_rules, 1)
    msg = "- (Before cleaning) # of rules: {}, Classes: {}".format(
        len(read_rules), Counter([elem.infer_class for elem in read_rules])
    )
    print(msg)
    rule_out_info.write(msg + "\n")
    msg = "- (After cleaning) # of rules: {}, Classes: {}".format(
        len(cleaned_rules), Counter([elem.infer_class for elem in cleaned_rules])
    )
    print(msg)
    rule_out_info.write(msg + "\n")

    # Store cleaned rules.
    clean_rule_text = [elem.original_description for elem in cleaned_rules]
    rules_process.save_rules(cfg.get_decision_tree_rules_path(), clean_rule_text)

    # Output readable cleaned rules according to their classes.
    match_rules = []
    unmatch_rules = []
    for elem in cleaned_rules:
        if elem.infer_class == "M":
            match_rules.append(elem.readable_description)
        else:
            unmatch_rules.append(elem.readable_description)
    rule_out_info.write("\n--- Match rules ---\n")
    for elem in match_rules:
        rule_out_info.write(elem + "\n")
    rule_out_info.write("\n--- Unmatch rules ---\n")
    for elem in unmatch_rules:
        rule_out_info.write(elem + "\n")
    rule_out_info.write(
        "### The End (%s). ###" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    rule_out_info.flush()
    rule_out_info.close()
