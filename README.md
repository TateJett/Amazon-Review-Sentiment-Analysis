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

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/Amazon-Review-Sentiment-Analysis.git
   cd Amazon-Review-Sentiment-Analysis
