# War Through the Lens of AI: Image Analysis of the Russia-Ukraine War in Spanish News
Computer Vision Project: Building an image Classification Model to detect war in News broadcasts. Subsequent inference with the obtained results is included. 

## Overview
The representation of war in media significantly impacts public perception and political discourse. This study aims to analyse the evolution of visual reporting on the Russia-Ukraine war in Spanish news broadcasts. We investigate how the depiction of the war on three channels, Antena 3, La Sexta, and Telecinco, changed from December 2022 to April 2024, focusing on the evolution of war coverage and on-the-ground war imagery. 

To achieve this, we use a subset of over 10,000 manually labelled screenshots from news broadcasts covering the Russia-Ukraine war, distinguishing between war-related and non-war-related content. We use a pre-trained ResNet50 network to build a binary classification model capable of accurately classifying if an image is war-related. Using this model, we track how the imagery of the war evolved over time, finding that as the war progressed, the proportion of war-related imagery in news broadcasts decreased, as well as war coverage overall. This trend is consistent across all three channels. Furthermore, the fluctuations in war images do not strongly correlate with actual events and military actions, suggesting a divergence between media representation and reality.

## Methodology
- [Data Collection](#data-collection)
- [Data Labeling](#data-labelling)
- [Model Training](#model-training)
- [Inferences](#inferences)

### Data Collection
- Collected data from nightly news broadcasts of Antena 3, La Sexta, and Telecinco.
- Each video is approximately 30 minutes long, spanning from December 2022 to April 2024.
- Corresponding transcripts were also collected.

### Data Labeling
- Created a subset of over 10,000 images.
- Manually labeled images into three categories: Military, Physical Damage from the War, and Not War.
- Performed Human Level Performance test.
- The labelled dataset is published on Kaggle: [The Russia-Ukraine War Images on Spanish News](https://www.kaggle.com/datasets/viktoriiayuzkiv/the-russia-ukraine-war-images-in-spanish-news) 

### Model Training
- Experimented with various classification models.
- Selected a pre-trained ResNet50 model for its superior performance.
- Fine-tuned the model by adjusting hyperparameters and conducting error analysis.

### Inferences
- Applied the trained model to generate predictions for the entire dataset.
- Performed a descriptive analysis of the visual representation trends.
- Compared trends with real events to assess alignment between news reporting and reality.


## Collaborators
- Angelo Di Gianvito
- Oliver Gatland
- Viktoriia Yuzkiv
