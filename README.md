

# Arabic Offensive Language Detection model from social media posts/comments.
This is a release includes model for offensive language detection for Arabic social media posts, trained using comments from different online platforms and tweets.
The model use a traditional SVM designed using character ngrams. The motivation for using Support Vector model is to handle the skewneess present in the dataset (see Table 1, for more details). The model is evaluated using:
* 5-fold cross validation for evaluating in-domain data performance
* Official dev set for OSACT Offensive Language detection competition
* Other available dataset :


To train the model, we annotated ~5000 amount of data consists of 2 categories: offensive (OFF) and not offensive (NOT_OFF).
The contents are collected from the following sources:
* Twitter
* Youtube
* Facebook

The annotation of the collected dataset is obtained using Amazon Mechanical Turk (AMT). To ensure the quality of the annotation and language proficiency, we utilized two different evaluation criteria of the annotator. For more details, check the below paper:

Will be available in the proceedings of LREC2020:

```
@inproceedings{shammur2020offensive,
  title={A Multi-Platform Arabic News Comment Dataset for Offensive Language Detection},
  author={Chowdhury, Shammur Absar  and Mubarak, Hamdy and Abdelali, Ahmed and Jung, Soon-gyo and Jansen, Bernard J and Salminen, Joni},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC'20)},
  year={2020}
}
```
In addition to the dataset mentioned in the above paper, we also added 948 data points along with 7000 annotated tweet from the training data of OSACT.

## Training the models

### SVM
For the training the classifier with SVM, we used TF-IDF representations for character ngrams (1,8). The reason to choose SVM with TF-IDF is their simplicity and execution time while having comparable performance for such dataset nature.

## Data Format
### Input data format
The input file should have the following fields, including
`<Input ID>\t<Text>\t<Class_Label>`
however when the model is not used to evaluate the performance, `<Class_Label>` is optional field.

### Output data format
The output of the file will include the following fields

* While running the model just for prediction:
`<id>\t<text>\t<class_label>`
* Output of the model when reference label is mentioned
`<id>\t<text>\t<class_label>\t<predicted_class_label>`
here predicted_class_label is the output of the model (OFF/NOT_OFF)

## Predicting using the models
To run the classification model please use python version 3.7, install dependencies

To install the requirements:
```
pip install -r requirements.txt
```

The model can be used in two ways, either using batch of data or single data points. Even though for single datapoint the batch processing script can be used, we suggest to use the example provided in `run_ar_offensive_language_detection_models_for_single_text.ipynb`

For batch classification of data:

```
python bin/prediction_model.py -c models/ar_offensive_detection_svm.config -d sample_data/test_instances.tsv -o results/test_instances_batch.tsv
```
For evaluation of batch with reference label, just add
the following flag to `prediction_model.py`

```
  --eval yes
```

The results of the model on the given dataset will be printed in the i/o
Example:
```
python bin/prediction_model.py.py -c models/ar_offensive_detection_svm.config -d sample_data/dataset_format_with_ref_labels.tsv -o results/dataset_with_reflab.tsv --eval yes &> logs/result_dataset_with_reflab.log
```

### Classification Results

As mentioned earlier, the performance of the model is tested using 1) 5-fold CV on training data 2) official dev set for OSACT (LREC2020) along with other 3? out of domain data including:


Test Data Sets           | F1 OFF   | Macro F1 | Weighted F1
-------------------------| :------: | :------: | :------:
5-fold CV                | **0.73** | **0.83** | **0.88**
OSACT_2020_Shared_dev_set| **0.74** | **0.84** | **0.91**
