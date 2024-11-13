# Amazon-Review-Sentiment-Analysis
Oct 27th 2023

This project analyzes Amazon food reviews to determine if a review is positive or negative, providing insights into customer sentiment. By leveraging text analysis and linear regression, the model predicts the likelihood of a review being positive, helping users gauge product feedback at a glance.

## Project Overview

The main goal of this project is to:

- **Categorize Amazon food reviews** as positive or negative based on commonly used sentiment words.
- **Identify customer sentiment** for each product, offering a quick summary of how well a product is received.
- **Use machine learning** (specifically a linear regression model) to predict the confidence of a review being positive.

## Data

- The data was sourced from a large dataset of Amazon food reviews.
- Initial dataset issues led to a switch from a `.tsv` to a `.csv` format.
- For computational simplicity, the data was reduced to **1,000 rows** for processing in Google Colab.
- Each product in the dataset was assigned a unique identifier (variables from `a` to `p7`).

## Methodology

1. **Data Preprocessing**:
   - Reduced dataset to 1,000 rows.
   - Assigned unique variables to each product for easier reference.

2. **Sentiment Word Lists**:
   - Created a list of **positive** words (e.g., 'Good', 'Great', 'Amazing') and **negative** words (e.g., 'Bad', 'Worst', 'Disgust').
   - Counted occurrences of these words in each review.

3. **Bag of Words Implementation**:
   - Iterated over the reviews to count instances of positive and negative words.
   - Stored word counts in `product_var_count` and `product_var_occurrences` dictionaries.
   - Calculated the sentiment score for each product based on word counts.

4. **TF-IDF Vectorization and Linear Regression**:
   - Used `TfidfVectorizer` to transform review text into vectorized features.
   - Applied a linear regression model to establish a relationship between the vectorized features and sentiment scores.
   - Tested the model’s prediction confidence on a sample review outside the training set.

5. **Ethical Considerations**:
   - Due to resource limitations, the model only uses a subset of the dataset. This limits the model’s applicability across all product reviews but still provides useful insights for the subset analyzed.

## Model Evaluation

The model was evaluated on a sample sentence: 

> “This cereal has no artificial sweeteners, is high in fiber, but has a great taste (A hard combination to find)”

The prediction indicates a positive sentiment with reasonable confidence, although the model’s accuracy is still limited due to the reduced dataset size.

## Results

- **Word Frequency and Sentiment Analysis**:
  - After assigning unique identifiers to each product and categorizing common words into positive and negative lists, the model could effectively gauge sentiment based on the frequency of these words in each review.
  - For each product, a sentiment score was generated based on the count of positive and negative words in its reviews. This process provided a numerical sentiment summary for each product, giving an overall indication of customer satisfaction.

- **TF-IDF Vectorization and Linear Regression Model**:
  - Using `TfidfVectorizer`, the text of reviews was transformed into feature vectors, enabling a deeper analysis of word significance within the context of each review.
  - A linear regression model was then applied to establish a relationship between the text features and sentiment scores, allowing the model to predict the likelihood of a review being positive.
  - The model was tested on an unseen review: *“This cereal has no artificial sweeteners, is high in fiber, but has a great taste (A hard combination to find).”* Given that this review was not part of the training set, it provided an interesting case to observe the model’s prediction capability, which indicated a positive sentiment for this review.

- **Dataset Limitations and Ethical Considerations**:
  - Due to computational limitations, the dataset was shortened to 1,000 rows, which limited the scope of analysis. This reduction, while necessary for processing in Google Colab, also meant that not all products in the full dataset were analyzed.
  - Although reducing the dataset did not bias the results for the products included, it did impact the overall generalizability of the model. Future iterations could benefit from expanding the dataset to improve model accuracy and robustness.

- **Model Usefulness and Practical Limitations**:
  - The model demonstrates potential in predicting the sentiment of reviews without requiring users to read each one, making it a valuable tool for quickly assessing customer opinions.
  - However, the model requires further validation, particularly around the accuracy of its confidence intervals. Additionally, due to limited training data, its effectiveness may vary across different types of reviews and products.

## Discussion

- **Performance**:
  - The model successfully categorizes reviews as positive or negative based on keyword frequency and linear regression predictions. However, its predictive confidence would likely improve with a larger dataset and more advanced validation techniques.
  
- **Potential Improvements**:
  - Expanding the dataset to include all available reviews would increase the model’s representativeness and robustness.
  - Automating testing and validation on a broader range of reviews could help in refining accuracy, especially for predicting confidence intervals.
  
- **Conclusion**:
  - Overall, the model shows promise as a preliminary tool for sentiment analysis of Amazon reviews, with applications in quickly summarizing customer feedback. With additional data and validation, it could serve as a practical solution for sentiment analysis across various products and categories.

## Future Improvements

To enhance the model, future iterations could:
- Expand the dataset to include all reviews for better coverage and more accurate predictions.
- Develop methods for validating the accuracy of the confidence intervals used in the predictions.
- Automate testing on a larger set of sample reviews.

## Installation and Usage

### Prerequisites

- Python 3.x
- Required libraries: `pandas`, `sklearn`, `numpy`

### Running the Project

1. **Download the Notebook**:
   - Download `Amazon-Review-Sentiment-Analysis.ipynb` from this repository.

2. **Install Dependencies**:
   - Open a terminal and run:
   ```bash
   pip install pandas sklearn numpy
