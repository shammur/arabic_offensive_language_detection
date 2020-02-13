

# News Categorization for English

To train the model, we annotated XY amount of data consists of 15 categories. The textual contents are collected from the following sources:
* AA
* AB

## Training the models
To train the model we conducted several experiments consisting of different machine learning algorithms and different feature representations: i) SVM and ii) BERT.

### Classification Results

### SVM
For the training the classifier with SVM, we used TF-IDF feature feature representations. The reason to choose SVM with TF-IDF is their simplicity and execution time while having comparable performance.

To run the classification model Please use python version 3.7, install dependencies and then use the following ```bin/SVM_with_BagOfWords_classifier.py``` script.
```
pip install -r requirements.txt

python bin/SVM_with_BagOfWords_classifier.py -c models/sm_news_en_trn_svm_svm.config -d data/sample_data.csv -o results/sample_data_classified_label.csv
```

