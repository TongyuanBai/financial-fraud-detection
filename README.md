# Financial fraud detection for listed companies

Code of my undergraduate thesis (2022), *Financial fraud detection and prediction of listed companies using machine learning algorithms* (基于机器学习的上市公司财务造假识别与预警模型研究). Annual financial statements of Chinese listed companies are used to find, industry by industry, the indicators related to fraudulent reporting and to classify company-years as fraudulent or not.

The work was published as:

Z. Zhao and **T. Bai**. Financial fraud detection and prediction in listed companies using SMOTE and machine learning algorithms. *Entropy* 24(8): 1157, 2022. [doi:10.3390/e24081157](https://doi.org/10.3390/e24081157)

## Data

The data are those of problem A of the 9th TipDM Cup Data Mining Challenge (第九届“泰迪杯”数据挖掘挑战赛, 2021), which were distributed to participants. They are not included in this repository.

- 附件1: industry of each listed company (19 industries)
- 附件2: financial data, 18,060 company-years with 363 fields, including the fraud label `FLAG`
- 附件3: description of the fields in 附件2

About 1 % of the company-years are labelled as fraudulent.

## Method

1. **Missing values.** Fields unrelated to fraud (report metadata such as publication dates, report type, fiscal period and currency) are removed. Indicators with at least 70 % missing values are dropped, those with 40–70 % missing are filled with 0, and the rest with the column mean.
2. **Industries.** The data are split by industry and standardised within each industry. Five small industries are merged into one group.
3. **Feature selection.** Logistic regression (absolute coefficients), random forest, GBDT, decision tree and XGBoost (feature importances) each rank the indicators. The top indicators of each model are compared, and those that appear in the lists of four models are kept.
4. **Classification.** 70/30 train/test split; the minority class of the training set is oversampled. Logistic regression, SVM, random forest and XGBoost are tuned by grid search and evaluated by accuracy, precision, recall, F1 and the ROC curve (AUC).
5. **Ensembles.** Soft-voting classifiers combine XGBoost with logistic regression, SVM and random forest.

## Files

| File | Content |
|---|---|
| `python/final.py` | Main script: missing values, one file per industry, standardisation, feature selection, the four classifiers with grid search, voting ensembles and ROC curves (manufacturing industry) |
| `python/code.py` | Missing-value step on its own |
| `R/process.R` | Class and industry frequency plots; split by industry, standardisation, one CSV file per industry |
| `python/select_models.py` | Feature selection on one industry file written by `process.R` |
| `python/classification_models.py` | Model tuning and evaluation on one industry file (earlier version of the modelling part of `final.py`) |

## Running the code

The scripts are kept as written in 2022. They were run section by section in Spyder and RStudio, read and write files under hard-coded paths (`E:/essay/data/`, `F:/essay/data/`), and have comments in Chinese. To run them, put the competition files in a folder and change the paths.

- `final.py` expects `data1.csv`: 附件2 with the industry of each company from 附件1 in a column `category`, without the report-metadata fields. It writes `data_clean.csv` and one Excel file per industry, then models `制造业.xlsx` (manufacturing).
- `process.R` reads `data_clean.csv` and writes `data_<industry>.csv`; `select_models.py` and `classification_models.py` read one of these (`data_make.csv` is manufacturing). For another industry, change the file name.

Python packages: see `requirements.txt` (the code uses the scikit-learn API of 2022; a few arguments, such as `min_impurity_split`, were removed in later versions). R packages: `ggplot2`.

## License

MIT
