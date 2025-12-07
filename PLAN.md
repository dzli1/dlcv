**pre pipeline
**
1. 

Dataset, scoping, and design:
- classify art that is western in 10-12 westerns styles, with focus on art from 1400 to 1970
- find a dataset with full attributes including medium, etc
- new prop dataset that includes visual attributes of each painting.

2. 
dataset:
- use existing datasets of wikiart, artemis, and semart

maybe make a wikiart/museum scraper to find additional art to use (maybe as a test set)


split: split, create a few cool looking graphs of distribution

**training:**

1. preprocessresize normalize augmet
2. encode labels of movmement and artist, should be one multitask thing that does both, which is slightly more complex,
the tasks are highly related so this should be okay

3. train the model
- confusion matrix, compare architecures, parameter and accuracy graphs, hyperparameter tuning
- regularization, shared plus separate classification

4. model stats, visualization

**the above is the main project**

**side project 1**
do smth with nlp

gpt get visual features of each painting, the try on the test set

fine tuning, style attributes, multi label classifier for style

fine tune. qualitative review.  (use the same dataset)

**side project2**

use a bit of the same dataset

generate AI art data. google ai, google colab, everything etc and all combine together.
make based on description

train data classify, generalization testing, error and other stuff
visualization, quantitative scores, 

**final steps**
clean the code base
save a few models
document results and writ the report, make the video


write the report 












