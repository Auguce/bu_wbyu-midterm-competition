# Final Algorithm Description and Performance Improvements  
  
## Introduction  
  
This document outlines the final algorithm implemented for predicting Amazon product review scores using machine learning techniques. The goal was to accurately classify reviews into one of the five possible scores (1 to 5 stars). The dataset provided included various features such as textual data (reviews and summaries), user and product IDs, timestamps, and helpfulness metrics.  

## Algorithm Overview  
  
The algorithm employs a **hierarchical classification approach**[^1] using Support Vector Machines (SVM) with a linear kernel (`LinearSVC`). The model is structured into three layers:  
  
1. **First Layer**: Binary classification to distinguish between 5-star reviews and non-5-star reviews.  
2. **Second Layer**: For non-5-star reviews, a binary classifier differentiates between middle scores (3 and 4 stars) and low scores (1 and 2 stars).  
3. **Third Layer**:  
   - **Model A**: Distinguishes between 3-star and 4-star reviews among the middle scores.  
   - **Model B**: Distinguishes between 1-star and 2-star reviews among the low scores.  
  
## Feature Engineering  
  
### Text Preprocessing  
  
- **Lowercasing**: Converted all text to lowercase to ensure uniformity.  
- **Removal of Non-alphabetic Characters**: Stripped out any character that is not a letter.  
- **Stop Words Removal**: Removed common English stop words using NLTK's stopwords corpus.  
  
### Textual Features  
  
- **TF-IDF Vectors**: Created TF-IDF features for both the review text and summary using `TfidfVectorizer` with bigrams (`ngram_range=(1,2)`) and increased `max_features` to capture more information (2000 for summaries and 5000 for reviews).  
  
### Sentiment Analysis Features  
  
- **VADER Sentiment Scores**: Used the VADER sentiment analyzer to extract:  
  - **Compound Score**: Overall sentiment score normalized between -1 (most extreme negative) and +1 (most extreme positive).  
  - **Positive, Neutral, Negative Scores**: Proportions of the text that fall into each sentiment category.  
- **Sentiment Word Counts**: Counted the number of positive and negative words in the review text based on predefined sets of positive and negative words.  
  
### Additional Features  
  
- **Word Count**: Number of words in each review.  
- **Text Lengths**: Character counts of the review text and summary.  
- **Helpfulness Metrics**: Calculated the helpfulness ratio and included both the numerator and denominator.  
- **Temporal Features**: Extracted the year, month, and day of the week from the timestamp.  
- **User and Product Encoding**: Transformed user and product IDs into numerical labels using `LabelEncoder`[^5].  
  
## Model Training and Optimization  
  
### First Layer Classifier  
  
- **Objective**: Separate 5-star reviews from the rest.  
- **Approach**:  
  - Used `LinearSVC` with hyperparameter tuning via `GridSearchCV`[^2].  
  - Balanced class weights to handle class imbalance using the `class_weight` parameter[^3].  
  - Adjusted the regularization parameter `C` to `[1, 2, 5, 10]`.  
- **Outcome**: Achieved high accuracy in distinguishing 5-star reviews.  
  
### Second Layer Classifier  
  
- **Objective**: Distinguish between middle scores (3 and 4 stars) and low scores (1 and 2 stars) among non-5-star reviews.  
- **Approach**:  
  - Implemented another `LinearSVC` with hyperparameter tuning.  
  - Used `C` values `[0.5, 1, 2]`.  
  - Balanced class weights to mitigate class imbalance.  
- **Outcome**: Effectively separated middle scores from low scores.  
  
### Third Layer Classifiers  
  
#### Model A (3 vs. 4 Stars)  
  
- **Objective**: Differentiate between 3-star and 4-star reviews.  
- **Challenges**:  
  - Initial models tended to misclassify 3-star reviews as 4-star due to similarities in sentiment.  
- **Approach**:  
  - Added new sentiment features to better capture subtle differences.  
  - Adjusted class weights to `{3.0: 2, 4.0: 1.5}`[^3] to give more importance to 3-star reviews.  
  - Increased `C` values to `[0.5, 1.0, 1.5]` for flexibility in decision boundaries.  
- **Outcome**: Improved the recall for 3-star reviews and achieved a better balance between precision and recall for both classes.  
  
#### Model B (1 vs. 2 Stars)  
  
- **Objective**: Differentiate between 1-star and 2-star reviews.  
- **Approach**:  
  - Similar to previous models, used `LinearSVC` with hyperparameter tuning.  
  - Explored `C` values `[0.01, 0.05, 0.1]` suitable for smaller datasets.  
- **Outcome**: Adequately distinguished between the two lowest ratings.  
  
## Performance Evaluation  
  
- **Accuracy**: Achieved an overall testing accuracy of approximately **64.1%**, improving upon previous models.  
- **Kaggle Submission**: The improved model achieved a Kaggle score of **0.64341**, surpassing previous records.  
- **Classification Report**: Notable improvements in the precision and recall for 3-star and 4-star reviews.  
- **Confusion Matrix**: Showed a better distribution of correct predictions across all classes, especially reducing the misclassification of 3-star reviews as 4-star.  
  
## Special Tricks and Observations  
  
### Hierarchical Modeling[^1]  
  
- **Why Hierarchical?**: Direct multi-class classification was less effective due to class imbalances and similarities between adjacent classes.  
- **Benefit**: Breaking down the problem into smaller, more manageable binary classifications improved overall accuracy.  
  
### Enhanced Sentiment Features  
  
- **Observation**: Basic sentiment scores were insufficient to distinguish between 3-star and 4-star reviews.  
- **Action**:  
  - Introduced sentiment components (`pos`, `neu`, `neg`) alongside the compound score.  
  - Counted specific positive and negative words to capture more nuanced sentiment differences.  
- **Result**: These features provided the model with more detailed sentiment information, aiding in differentiating reviews with subtle sentiment differences.  
  
### Adjusting Class Weights[^3]  
  
- **Observation**: Imbalanced classes led to biased models favoring the majority class.  
- **Action**: Customized class weights in the third-layer models to give higher importance to underrepresented classes (e.g., 3-star reviews).  
- **Result**: Improved recall for minority classes without significantly sacrificing precision.  
  
### Feature Scaling and Transformation[^4]  
  
- **StandardScaler with Sparse Matrices[^4]**: Used `with_mean=False` to accommodate sparse TF-IDF matrices.  
- **Label Encoding[^5]**: Converted categorical variables like user and product IDs into numerical formats suitable for SVMs.  
  
### Increased TF-IDF Features  
  
- **Observation**: Limiting `max_features` in TF-IDF vectorization constrained the model's ability to capture important terms.  
- **Action**: Increased `max_features` for both summary and text TF-IDF vectors to include more significant words and bigrams.  
- **Result**: Allowed the model to utilize a richer vocabulary, improving text feature representation.  
  
## Assumptions  
  
- **Data Quality**: Assumed that the provided data was clean except for necessary preprocessing steps (e.g., text cleaning).  
- **Model Interpretability**: Prioritized improving model performance over interpretability, given the use of SVMs with high-dimensional data.  
- **Computational Resources**: Assumed sufficient computational resources to handle increased feature sizes and longer training times due to added complexity.  
  
## Conclusion  
  
By employing a **hierarchical classification strategy**[^1], enhancing sentiment analysis features, adjusting class weights[^3], and carefully tuning hyperparameters using `GridSearchCV`[^2], the final model achieved significant improvements in accuracy and balanced performance across all classes. The focused effort on distinguishing between closely related classes (especially 3-star and 4-star reviews) was critical in enhancing the model's predictive capabilities.  
  
---  
  
**Note**: The Python script accompanying this writeup reproduces the best Kaggle submission exactly, ensuring consistency and reproducibility of results.  
  
---  