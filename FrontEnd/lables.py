from sklearn.preprocessing import MultiLabelBinarizer

labels = [["melayu","jawa","english","non_labels"]]
mlb = MultiLabelBinarizer()
mlb.fit(labels)
MultiLabelBinarizer(classes=None, sparse_output=False)
print(mlb.classes_)
print(mlb.transform([["melayu"]]))