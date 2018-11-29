# Simple application of SVM

==Thanks to Faruto enhancement toolkit==

== Thanks to http://www.ilovematlab.cn/thread-48175-1-1.html ==

*Install libsvm and Faruto toolkit before running the program*

---

The main task is to design a binary classifier based on SVM model. Both the training data and the data to be predicted are vectors with six characteristic dimensions, with 44,610 and 18,880 data respectively. We need to judge the labels that each data should belong to, namely, 1 or -1, according to the feature dimension.

 The labels of the data to be predicted are not given, so we cannot directly evaluate the accuracy of the model. Based on the traditional SVM training model and mature data processing methods, I chose to indirectly judge the accuracy of the model by dividing the training data into training set and test set, which involved the classic methods of normalization, dimensionality reduction processing and cross-validation.

The optimal c and g obtained were **0.7071** and **5.6569**, respectively, and the highest cross-validation accuracy reached 76.42% and the overall accuracy reached **75.14%** (3757/5000). There are 2,955 total support vectors.