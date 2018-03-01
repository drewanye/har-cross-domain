# Cross-domain Wearable Human Activity Recognition via Domain-adversarial Convolutional and Recurrent Neural Networks

## Introduction
Human Activity Recognition (HAR) is a hot research field in pervasive and mobile computing. Human activity recognition (HAR) can be affected by various factors, e.g., different users and sensor positions. Therefore, the activity recognition model trained in one scenario yields inaccurate results in another one. The information from the target scenario is required to adjust the original model trained in one scenario to adapt the target scenario. Cross-domain human activity recognition refers to transfer information from source domain to target domain to increase the accuracy of activity recognition on target domain. Existing methods mainly focus on supervised cross-domain human activity recognition, which requires a large labeled data on target domain. But annotation is a time-consuming and expensive work. Previous unsupervised cross-domain human activity recognition used unlabeled data in target domain to adjust the parameters in the model that had been trained in source domain. But the separate method cannot produce effective result.
To address aforementioned problems, we propose a deep unsupervised domain adaptation architecture based on domain-adversarial training for wearable activity recognition across domains, which simultaneously trains source label classification and domain classification in an adversarial way. We conduct domain transferring experiments across different users and sensor positions on two public real-world datasets.

## Model
![architecture](https://github.com/drewanye/har-cross-domain/blob/master/diagrams/architecuture.jpg)
The goal of the proposed algorithm is to learn a robust activity classifier on data from source domain, which generalizes to target domain without requiring sufficient activity labels required for standard supervised learning algorithms. Our model is trained such that the representations of multi-channel raw sensor from source domain are similar to those from the target domain. The learned domain-invariant representations allow a classifier trained on data from source domain to annotate labels for target domain.
Our novel architecture is that the feature extraction layers are shared by two classifiers trained simultaneously. Figure 1 shows the architecture of our model. One classifier is trained to correctly predict the activity labels on source domain while another classifier is a binary classifier that incorrectly predicts the domain labels of the input to learn domain-invariant representations.

## The structure of the project
Requirements: TensorFlow 1.2.1 or above, Python 2.7
<li> baselines: baseline methods
<li> data_utils.py: utils of prepocessing data
<li> flip_gradient.py: gradient reverse layer
<li> model.py: model file
<li> preprocess_REALDISP_data.py: preprocessing REALDISP data
<li> preprocess_RealWorld_data.py: preprocessing RealWorld data

## Experiments
We evaluate our approach on two datasets: REALDISP and RealWorld. Both datasets contain activities that are collected from more than 10 users and multiple sensor positions. Figure 2 illustrates the sensor positions of the REALDISP and RealWorld datasets.
![sensor_positions](https://github.com/drewanye/har-cross-domain/blob/master/diagrams/sensor_positions.jpg)
We conduct experiments on two scenarios: cross-subject activity recognition and cross-position activity recognition.
### Cross-subject activity recognition
Cross-subject activity recognition refers to inferring a knowledge from labeled user data (source domain) to new user data with none or few labels (target domain) to recognize activity of the new user. For both datasets, we examine our work by leave-one-subject-out.
### Cross-position activity recognition
Cross-position activity recognition refers to inferring knowledge from labeled data collected in one body part to another with none or few labels to recognize activity of this position.
### Baseline methods
**Source Only**: Source Only is that the model trained only on source domain is applied directly to annotate labels for target domain.
**DANN**:	DANN is a CNN based domain adversarial approach.
### Experimental Results
![results](https://github.com/drewanye/har-cross-domain/blob/master/diagrams/results.jpg)