# Corporate Credit Rating Dataset: Exploratory Data Analysis and Data Preparation

The dataset `ratings_df` consists of 2029 entries (rows) and 31 columns. Each entry represents a big US firm traded on NYSE or Nasdaq. The ratings span the period from 2010 to 2016. The dataset has 593 unique US firms, as seen from `ratings_df.Name.value_counts()`.    
## Credit Ratings

The target variable is the `Rating` column, representing the credit rating assigned by agencies. Credit ratings categorize a company's ability to repay debt. Taking a closer look at the list of agencies and their different ratings using `ratings_df['Rating Agency Name'].value_counts()` and `ratings_df.groupby('Rating Agency Name')['Rating'].unique()`:

|     Rating Agency Name             |                     Ratings             |      Counts      |
|------------------------------------|-----------------------------------------|------------------|
| DBRS                               |                              [AA, BBB]  |        3         |
| Egan-Jones Ratings Company         |  [A, BBB, AA, B, CCC, BB, AAA, CC, C]   |      603         |
| Fitch Ratings                      |          [BBB, A, AA, CC, BB, B, CCC]   |      100         |
| Moody's Investors Service          |      [A, BBB, BB, B, AAA, CCC, C, AA]   |      579         |
| Standard & Poor's Ratings Services |  [BBB, A, BB, AA, B, D, AAA, CCC, CC]   |      744         |

The dataset shows an imbalance in credit ratings, with varying frequencies for each rating category as it is evident from `ratings_df.Rating.value_counts()`

|Ratings| Count|
|------|-------|
| BBB  |   671 |
| BB   |  490  |
| A    |  398  |
| B    |  302  |
| AA   |   89  |
| CCC  |   64  |
| AAA  |    7  |
| CC   |   5   |
| C    |   2   |
| D    |   1   |

In addition, we are working with ratings from different agencies. One way to address this is to simplify and merge the ratings labels according to the following table from [Investopedia: Corporate Credit Ratings](https://www.investopedia.com/terms/c/corporate-credit-rating.asp)

| Moody's     | Standard & Poor's |  Fitch            |   Grade      | Risk         |
|-------------|-------------------|-------------------|--------------|--------------|
| Aaa         | AAA               | AAA               | Investment   | Lowest Risk  |
| Aa          | AA                | AA                | Investment   | Low Risk     |
| A           | A                 | A                 | Investment   | Low Risk     |
| Baa         | BBB               | BBB               | Investment   | Medium Risk  |
| Ba, B       | BB, B             | BB, B             | Junk         | High Risk    |
| Caa/Ca      | CCC/CC/C          | CCC/CC/C          | Junk         | Highest Risk |
| C           | D                 | D                 | Junk         | In Default   |

Instead of 10 different rating categories, we have now 6 categories. Using a dictionary for the mapping of new ratings and old ratings and `ratings_df['Rating'].map(rating_dict)` we have

|(New) Rating    | Counts | (Old) Ratings   |
|----------------|--------|-----------------|
|'Lowest Risk'   |  7     |'AAA'            |
|'Low Risk'      |  487   | 'AA', 'A        |
|'Medium Risk'   |  671   |'BBB'            |
|'High Risk'     |  792   |'BB', 'B'        |
|'Highest Risk'  |  71    |'CCC' 'CC', 'C'  |
|'In Default'    |  1     |'D'              |

The rows with the Ratings: `'Lowest Risk` and `'In Default` are dropped from the dataset given their small value counts. 

The categorical variable `Rating` was converted into numerical labels using the `LabelEncoder` from scikit-learn's preprocessing module, assigning a distinct integer code to each unique label.

Despite these enhancements, the dataset remains unbalanced. To tackle this, SMOTE Analysis was applied to generate synthetic instances for the minority classes, addressing the imbalance automatically through the imbalance library.
Now, to transform the categorical variable `Rating` into numerical labels, the `LabelEncoder` from the scikit-learn's preprocessing module is utilized where each unique label is assigned a unique integer code. 

Although improved, our dataset still remains unbalanced.  To tackle this, [SMOTE Analysis](smote.md) was applied to generate synthetic instances for the minority classes using the `SMOTE` function from the `imblearn.over_sampling`.  

## Input Features:
The other columns in the dataset are the input features related to financial indicators and information about the company. 

The 5 features with the company information such as `Name`, `Symbol` (for trading), `Rating Agency Name`, `Date`, and `Sector` provide context and additional details for analysis but their inclusion in the model may not be necessary for the specific task of credit rating prediction.  However, it is important to include the `Sector` feature in our model is crucial for capturing industry-specific nuances that significantly influence a company's financial performance and risk profile. Different sectors exhibit distinct economic characteristics and respond differently to market conditions. By incorporating the `Sector` variable, we aim to enhance the granularity of our analysis, ensuring that the machine learning model discerns sector-specific trends and challenges.

In order to integrate the categorical data `Sector` it into the model, `LabelEncoder` is employed to represent it numerically.

The dataset includes 25 financial indicators that can be categorized into different groups. These financial indicators collectively provide a comprehensive view of a company's financial health and performance, contributing to the evaluation of its creditworthiness.

### Liquidity Measurement Ratios:
These ratios provide insights into a company's short-term financial health and ability to meet its immediate obligations.
    1. `currentRatio`: Indicates the company's ability to cover short-term liabilities with short-term assets.
    2. `quickRatio`: Measures the company's ability to cover immediate liabilities without relying on inventory.
    3. `cashRatio`: Reflects the proportion of cash and cash equivalents to current liabilities.
    4. `daysOfSalesOutstanding`: Measures the average number of days it takes for a company to collect payment after a sale.

### Profitability Indicator Ratios:
These ratios evaluate a company's ability to generate profits relative to its revenue and investments.
    5. `netProfitMargin`: Represents the percentage of profit relative to total revenue.
    6. `pretaxProfitMargin`: Measures profitability before taxes are considered.
    7. `grossProfitMargin`: Indicates the percentage of revenue retained after deducting the cost of goods sold.
    8. `operatingProfitMargin`: Reflects the company's profitability from its core operations.
    9. `returnOnAssets`: Gauges how efficiently a company utilizes its assets to generate earnings.
    10. `returnOnEquity`: Measures the return generated on shareholders' equity.
    11. `returnOnCapitalEmployed`: Assesses the efficiency of capital utilization in generating profits.
    12. `ebitPerRevenue`: Measures earnings before interest and taxes relative to revenue.

### Debt Ratios: 
These ratios assess the company's leverage and debt management.
    13. `debtEquityRatio`: Measures the proportion of debt relative to equity.
    14. `debtRatio` : Represents the percentage of a company's assets financed by debt.

### Operating Performance Ratios:
These ratios focus on operational efficiency and effectiveness.
    15. `assetTurnover`: Evaluates how efficiently a company utilizes its assets to generate sales revenue.
    16. `fixedAssetTurnover` : Measures the efficiency of generating sales from fixed assets.
    17. `payablesTurnover`: Measures the efficiency of a company's payment of its liabilities.

### Cash Flow Indicator Ratios:
These ratios delve into a company's cash flow dynamics, providing insights into its financial sustainability. 
    18. `operatingCashFlowPerShare`: Reflects the cash generated by core business operations per share.
    19. `freeCashFlowPerShare`: Measures the amount of cash available to shareholders after covering operational expenses and capital expenditures.
    20. `cashPerShare`: Represents the amount of cash available per outstanding share.
    21. `operatingCashFlowSalesRatio`: Evaluates the percentage of sales revenue converted into cash from operating activities.
    22. `freeCashFlowOperatingCashFlowRatio`: Measures the efficiency of converting operating cash flow into free cash flow.
    23. `effectiveTaxRate`: Reflects the company's tax efficiency.
    24. `companyEquityMultiplier`: Indicates the multiplier effect on equity due to debt
    25. `enterpriseValueMultiple`: Evaluates a company's overall value relative to its earnings.

# Descriptive Statistics

The function `data.describe()` gives statistical descriptions like `mean`, `min`, `max`, `percentiles` of the numerical financial indicators. Comparison of the mean to the median and examining the range between percentiles, there seems to be an indication of the presence of outliers. 

Features exhibit different scales, as evident from the magnitude of mean and standard deviation values. To ensure equal contribution from all features for machine learning algorithms, feature scaling is performed. The Min-Max scaling technique is applied to normalize the numerical values representing financial indicators.  For each column, the `MinMaxScaler` function from `sklearn.preprocessing` is used to transform the values into a standardized range between 0 and 1. The values are then multiplied by 1000, to amplify the scaled values. 

Additionally, a logarithmic transformation is applied to each value using the `np.log10` function, with a small constant (0.01) added to avoid issues with zero values. This dual transformation approach aims to normalize and potentially enhance the interpretability of the financial indicators in the dataset.

This work in exploring and preparing our dataset, sets the stage for the next phase: deploying different machine learning models to forecast credit ratings.