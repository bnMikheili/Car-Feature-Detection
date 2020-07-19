# Car-Feature-Detection
## Data preparation for training
For this project I've chosen tasks that would be done in the same manner. Those tasks are prediction of several features of the car: `color`, `number of doors` and `category`.

To train models on those tasks first step was to identify images that show the car exterior. To solve this problem I used `pretrained SSD model with MobileNet backbone`. I chose this model because of it's inference speed and the best performace on wide objects (such as car) compared to it's rivals. Using this model I extracted images of car exterior for each car ID.

As long as there are several images for each car, I decided that making the data `image oriented` would be the best approach for the solution, so I've transformed it in such way that each image of the car shares it's features. Model get's an image as input and predicts those car features for the image.

For each above mentioned tasks I cleared the data. Because of the huge amount of labels, most of which apperad several times in the dataset, I decided to drop those labels and give more attention to normal ones. Even after doing so, the dataset was still quite imbalanced, so I had to balance it by taking samples for each label with restriction on maximum number. 

`To take a look a the workflow of preprocessing check FLOW.ipynb`

## Modelling
When it came to choosing a model, it was obvious that a simple architecture with several CNN layers wouldn't be able to afford learning the complexity of the task. So I chose to take the `RESNET18` architecture. Because of it's huge amount of weight parameters (>11M), not to waste time on training it from scratch I decided to make `Transfer Learning` using it's pretrained weights, so I used it as a feature extractor and expanded it with pyramid architecture.

## Evaluation
At the beginning of the project, the first I knew I had to do was to extract the data that I wouldn't touch until the model evaluation. For that reason I extracted the test set at the very beginning of the preprocessing. Even before transforming the data to image oriented one, I kept the car ID's that I only used for final evaluation of the models.

As long as the models were trained on getting an image as input and predicting the features, I had to unite those predictions, as each car has several images. I summed up the ouputs of images of the same car and applied softmax function on it.

As the dataset labels were quite imbalanced and I had to balance them for the training, I assumed that evaluating model performance on data with balanced distribution would lead to a mistake, so for the final evaluation I took car ID's from test set with the original data distribution. All this may have caused the fact, that a dummy model predicting only the label of the largest set could have better overall prediction accuracy (it happened so on door number prediction where that type of model would have 96% accuracy), but of course, that type of performace is worthless. I've measured the performace for each class by checking the precision/recall for each of them and it performs quite well most of the times. `You can take a better look in EVALUATION.ipynb` . I've also added code in the same file giving the user ability to test the performance on his/her own input.

### `Color prediction performace ( 0.76 F1 score weighted average )`
The color prediction model has performed quite well overally, but what's more important, it has got quite good results on almost every class. Except the green color, for every class the minimum of F1 score was 0.68

### `Door number prediction performace ( 0.94 F1 score weighted average )`
As the 4/5 door cars represented more than 96% one of the main goals was not to let to learn only the distribution of the dataset. Finally it can be counted as a success, because the F1 score was 0.37 on 2/3 door cars while maintaining 0.96 F1 score (0.98 Precision and 0.94 Recall) on 4/5 doored cars.

### `Category prediction performace ( 0.6 F1 score weighted average)`
Because of variety of car categories and the similarities among them I can say that this task was the hardest one mentioned here. Because of the similarities of Hatchback and Coupe with Sedan the model wasn't able to differentiate them as separate instances, but overall result is still positive, as the model managed to successfully differentiate two biggest classes, Jeep and Sedan (0.83 and 0.77 F1 scores respectively, both of them with precision higher than 0.83), and still had a positive performance on a small class of Microbus with 0.46 F1 score. 