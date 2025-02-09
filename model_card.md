# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

*   **Model Name:** Census Income Prediction Model
*   **Model Version:** 1.0
*   **Developed By:**  Merlyn Patil
*   **Model Type:** RandomForestClassifier (scikit-learn) 
*   **Input:** A tabular dataset with features related to individuals, including age, workclass, education, marital status, occupation, relationship, race, sex, capital gains/losses, hours per week, and native country.
*   **Output:** A binary classification prediction indicating whether an individual's income is above or below $50,000 per year ("<=50K" or ">50K").
*   **Size:** [Insert file size of the saved model file]
*   **Date Created:** 2024-11-15 (Adjust to the actual date)
*   **License:** [Specify the license under which the model is released.  e.g., MIT License, Apache 2.0, or None if it's proprietary]

## Intended Use

*   **Primary Use Case:** The model is intended to predict whether an individual's income exceeds $50,000 based on their demographic and employment characteristics.
*   **Intended Users:** Data scientists, researchers, policy analysts, and potentially individuals interested in understanding income predictors.
*   **Out-of-Scope Uses:** The model should **not** be used for making decisions that have legal or significant financial implications for individuals.  It is **not** a substitute for professional financial advice. It should **not** be used to discriminate against individuals based on protected characteristics.

## Training Data

*   **Dataset Name:** Adult Census Income Dataset (from `census.csv` in the provided repository).
*   **Source:** Extracted from the 1994 U.S. Census Bureau database.  (You might want to add more details or link directly to the data source if possible).
*   **Size:** [Number of rows in your training set after the split]
*   **Preprocessing:**
    *   Categorical features were one-hot encoded using scikit-learn's `OneHotEncoder`.
    *   The target variable ('salary') was binarized using scikit-learn's `LabelBinarizer`.
*   **Splitting:** The data was split into training (80%) and testing (20%) sets using `train_test_split` with `random_state=42` for reproducibility.

## Evaluation Data

*   **Dataset:** The 20% holdout test set from the Adult Census Income Dataset.
*   **Size:** [Number of rows in your test set after the split]
*   **Purpose:** Used to evaluate the model's generalization performance on unseen data.
*   **Slices:**  The model's performance was evaluated on slices of the data based on categorical features (workclass, education, etc.) to identify potential biases.

## Metrics

*   **Primary Metrics:**
    *   **Precision:** [Report the precision on the test set]
    *   **Recall:** [Report the recall on the test set]
    *   **F1-Score:** [Report the F1-score on the test set]
*   **Performance on Slices:**
    *   The model's precision, recall, and F1-score were calculated for each unique value within each categorical feature. (The results are saved in `slice_output.txt`). This revealed [Summarize any notable performance differences across slices. For example: "The model performed worse on individuals with 'Self-emp-not-inc' workclass."]

## Ethical Considerations

*   **Bias:** The model may reflect biases present in the training data. For example, historical income disparities related to race, sex, or occupation may be learned by the model. Careful monitoring and fairness analysis are necessary.
*   **Fairness:** The model should not be used in a way that unfairly discriminates against any individual or group.  Consider evaluating fairness metrics (e.g., demographic parity, equal opportunity) to assess potential disparities in performance across different demographic groups.
*   **Privacy:** The model uses demographic and employment data, which, in some cases, could be considered sensitive.  Ensure compliance with all applicable privacy regulations.
*   **Transparency:** While Random Forests are generally less interpretable than linear models, efforts should be made to understand the model's decision-making process (e.g., using feature importance analysis).
*   **Explainability:** While this model predicts income based on the provided features, it does not explain *why* those relationships exist.  Misinterpreting correlation as causation could lead to harmful conclusions.

## Caveats and Recommendations

*   **Data Limitations:** The model is trained on data from 1994.  Income distributions and demographic patterns may have changed significantly since then, which could affect the model's accuracy and fairness.
*   **Generalization:** The model's performance may vary on different populations or datasets.
*   **Monitoring:** The model's performance should be continuously monitored to detect any degradation in accuracy or emergence of biases.
*   **Regular Retraining:** The model should be retrained periodically with updated data to maintain its accuracy and relevance.
*   **Thresholding:** The default classification threshold (0.5) may not be optimal. Consider adjusting the threshold based on the desired balance between precision and recall, or using a probability calibration technique.
*   **Further Research:** Further research is needed to assess the model's fairness and potential for discriminatory impact. Fairness-aware machine learning techniques should be explored.