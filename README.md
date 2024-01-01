# Corporate Credit Rating Forecast using Machine Learning Methods

> - **The Problem** Corpoarte credit ratings, issued by credit rating agencies like Standard and Poor's and Moody's, express the agency's opinion about the ability of a company to meet its debt obligations. Each agency applies its own methodology to measure creditworthiness and this assessment is an expensive and complicated process. Usually, the agencies take time to provide new ratings and update older ones. This causes delays in decision-making process for investors who use these ratings to assess their credit risk. 
> - **Solution** One solution to address delays would be to use the historical financial information of a company to build a predictive quantitative model capable of forecasting the credit rating that a company will receive. I employed machine learning techniques, creating classification models that quickly forecast credit ratings. 
> - **Applications** The insights gained can aid financial analysts, investors, and companies in making more informed and quick decisions related to credit risk. The classification methods used here can also be used to forecast other ratings like ESG Ratings.
> - **Skills** Explored classification methods like XGBoost, RandomForest and techniques to address imbalance in datasets.
[SMOTE Analysis]({% post_url 2023-12-26-smote %}) Also delved into Financial Ratios gaining knowledge on understanding a company's fiscal strength.


## Dataset

The dataset is obtained from Kaggle [Corporate Credit Rating](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/data).
This dataset comprises 2029 credit ratings assigned by major agencies to prominent US firms traded on NYSE or Nasdaq. The ratings span the period from 2010 to 2016. Each entry encompasses 30 features, with 25 being financial indicators. 

[Exploratory Data Analysis and Data Preparation](dataset-description)

## Methods

The following machine learning models were implemented

1. Logistic Regression
2. KNN
3. Naive Bayes
4. SVM
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. MLP

