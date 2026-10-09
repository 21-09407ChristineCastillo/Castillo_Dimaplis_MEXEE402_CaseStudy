 # MexEE 402: Data Preprocessing Case Study

MexEE Elective 2: Data Science and Machine Learning
Batangas State University, Alangilan Campus
1st Semester, AY 2026-2027

## Members

| Name | Student Number | Section |
|---|---|---|
| Castillo, Christine Dianne | 21-09407 | MEXE 4102 |
| Dimapilis, Mc Leejoe | 23-02583 | MEXE 4102 |

## Notebook links

| Chapter | Member 1 | Member 2 |
|---|---|---|
| Ch1_2_3 | [link]() | [https://colab.research.google.com/drive/1KR4wFae2T9xVuIQD_hr9WJcaqnnJdC-4?usp=sharing]() |
| Ch4 | [link]() | [https://colab.research.google.com/drive/1n8PfhDED1Yz8d6APzQODnhUXOUn01RwY?usp=sharing]() |
| Ch5 | [link]() | [https://colab.research.google.com/drive/1FIg8lu38MseYfH1usoyNGD5gIlcr9cga?usp=sharing]() |
| Ch6 | [link]() | [https://colab.research.google.com/drive/1VR2PMQ6qlUqgihkg-Nz5vKtyXbXM0lAU?usp=sharing]() |
| Ch7 | [link]() | [https://colab.research.google.com/drive/13Xp5sOS-tX-JmpvMxTiqIYjftWLlz14T?usp=sharing]() |
| Ch8 | [link]() | [https://colab.research.google.com/drive/1lWk-th_oToAuS1IjTf2R2cDpAogB-SJ_?usp=sharing]() |
| Ch9 | [link]() | [https://colab.research.google.com/drive/1r4d2STRgBLZS51Udq4fFFZkhnZlCsuCC?usp=sharing]() |

## What we learned

Chapter 1_2_3
- In these chapters, we learned how to load, understand, and clean data using Python and Pandas. We learned how to handle missing values, remove duplicates, and get rid of unnecessary data. We also realized that even if we think our code is correct, small mistakes like forgetting to import Pandas or upload the CSV file can cause errors. It taught us to double-check our code and make sure everything is in the right order.
  
Chapter 4
- This chapter taught us that raw data usually isn't ready for a model, and that we can build better inputs by combining columns (like Lemonade per Degree), grouping values into bins, or converting categories to numbers. The key idea we took away is that the encoding has to match the category: Little/Medium/Lots has an order, so it gets 0, 1, 2, but Sunny/Cloudy/Rainy has no order, so it needs one-hot. What surprised us is that we first thought everything in the notebook was correct, but when we had it checked, we found a lot of minor errors.

Chapter 5
- This chapter showed us that a model can favor a feature just because its numbers are bigger, like Grades (0 to 100) overpowering Study Hours (0 to 20). StandardScaler sets the mean to 0 and the standard deviation to 1, while MinMaxScaler squeezes everything into 0 to 1. We were surprised again that we trusted the notebook at first, then found minor errors when we double-checked.

Chapter 6
- In this chapter, we learned how to identify and handle outliers using methods like Z-score and IQR. We also learned that outliers can affect our data and results. What surprised us was that the Z-score did not detect 100 as an outlier, while the IQR method did, showing us that different methods can give different results.

Chapter 7
- In this chapter, we learned about feature selection and how to choose the most useful features for a prediction. We learned about correlation and the three main methods: filter, wrapper, and embedded methods. We also learned that removing unnecessary features can make the model simpler and improve its performance. This can make the model simpler, faster, and more accurate by removing unnecessary or irrelevant features. It taught us that removing the data that does not help so the model can focus on what really matters.

Chapter 8
- This chapter taught us to treat preprocessing like a conveyor belt, where each step (imputing, then scaling) runs automatically and in the same order every time. We also understood why fit_transform is used on the training data but only transform on the test data, so the test data doesn't influence what the preprocessor learns. Once more, we were surprised that something we assumed was right had small mistakes.

Chapter 9
- This chapter put everything together on the Titanic data: filling missing values (median for numbers, "missing" for categories), log-transforming the skewed Fare, binning Age, and one-hot encoding the categories. The plots helped us confirm the cleaning worked, like seeing that the missing values were gone. We were surprised that checking our work showed minor errors we would have missed, and also that we could add so many types of graphs to the same notebook, like histograms, count plots, a box plot, and a correlation heatmap, each showing something different about the data.

  
## Errors we found
Chapter 1_2_3 Errors

Mistake 1: Duplicate check can never find duplicates
- Where: Removing Redundancies, df.duplicated() cell
- Original: df.duplicated().sum() → 0, then drop_duplicates() → 0
- Why it's wrong: Rank is unique for every row, so no two full rows can match. The check always returns 0, whatever the data contains.
- Correct: df.duplicated(subset=['Name', 'Platform', 'Year', 'Genre', 'Publisher']).sum() (or drop Rank before checking)

Mistake 2: Mean imputation gives a fractional year
- Where: Imputation markdown vs. the fillna cell
- Original: Year (numeric) → mean imputation, df['Year'].fillna(df['Year'].mean())
- Why it's wrong: The mean fills the 271 missing years with 2006.406443, which isn't a valid year, and it hides that the dates are unknown.
- Correct: df['Year'] = df['Year'].fillna(df['Year'].median()) (or use the mode, or leave as NaN and use astype('Int64')`)

Mistake 3: Real best-sellers removed as "outliers"
- Where: Noisy Data markdown vs. the Global_Sales <= 40 cell
- Original: "Treat games with Global Sales > 40M as outliers and remove them"
- Why it's wrong: This deletes Wii Sports (82.74M) and Super Mario Bros. (40.24M). The head() output now starts at index 2. These are genuine values, not errors, and the 40M cutoff is arbitrary.
- Correct: Keep them, or use a justified rule (e.g. IQR or a log-transform) and say why.

Mistake 4: Deletion step removes nothing
- Where: Deletion cell vs. the Imputation cell
- Original: df = df[df['Publisher'].notna()]`
- Why it's wrong: Publisher was already filled with the mode in the cell above, so there are no missing values left to delete. Imputation and deletion are alternatives, not sequential steps.
- Correct: Choose one strategy per column, e.g. delete the 58 rows with df.dropna(subset=['Publisher']) instead of imputing.
  
Chapter 4 Errors 
Mistake 1: Ordinal encoding values don’t match the code              
- Cell / Line: Markdown under “Ordinal Encoding”, vs. the OrdinalEncoder cell output       
- Original: Little → 1, Medium → 2, Lots → 3       
- Why it’s wrong: OrdinalEncoder starts counting at 0, and the output shows 0.0, 1.0, 2.0.          
- Correct version: Little → 0, Medium → 1, Lots → 2          

Mistake 2: One-hot example doesn’t match the column order              
- Cell / Line: Markdown under “One-hot Encoding”, vs. the pd.get_dummies output     
- Original: Sunny → [1,0,0], Cloudy → [0,1,0], Rainy → [0,0,1]    
- Why it’s wrong: get_dummies orders the columns alphabetically (Cloudy, Rainy, Sunny), not by order of appearance.           
- Correct version: Cloudy → [1,0,0], Rainy → [0,1,0], Sunny → [0,0,1]

Mistake 3: Output shows True/False, but the text says 1/0             
- Cell / Line: “Assigns 1 = True, 0 = False”, vs. the get_dummies cell             
- Original: df_encoded = pd.get_dummies(df_2, columns=['Weather'])               
- Why it’s wrong: pandas 2.x returns booleans (True/False), not the 1/0 integers the text describes.            
- Correct version: df_encoded = pd.get_dummies(df_2, columns=['Weather'], dtype=int)                  

Mistake 4: “very hot” is never used               
- Cell / Line: “define bins and labels” cell           
- Original: bins = [70, 75, 85, 95, 100]                 
- Why it’s wrong: the highest temperature, 95, falls in (85, 95], which is “hot”, so no row is ever “very hot”. df.head() only shows 5 of 7 rows, which hides this.            
- Correct version: bins = [70, 75, 85, 90, 100] (now 91 and 95 become “very hot”)

<br>

Chapter 5 Errors   
Mistake 1: The output is called a “dataset”, but it’s a NumPy array
- Cell / Line: Text under the StandardScaler output, “The outcome is a new dataset where the scales...”  
- Original: scaled_data = scaler.fit_transform(df)  
- Why it’s wrong: fit_transform returns a NumPy array, not a DataFrame, so the column names are lost and the printout has no headers.  
- Correct version: scaled_df = pd.DataFrame(scaled_data, columns=df.columns), then show scaled_df as the last line of the cell.  

Mistake 2: scaler is overwritte
- Cell / Line: “normalize the data” cell  
- Original: scaler = MinMaxScaler()  
- Why it’s wrong: the same name was already used for StandardScaler, so the first fitted scaler is lost. Running cells out of order can silently use the wrong one.  
- Correct version: standard_scaler = StandardScaler() in the first cell, and minmax_scaler = MinMaxScaler() with normalized_data = minmax_scaler.fit_transform(df_2) in the second.  

Mistake 3: Scaling is said to help with feature importance
- Cell / Line: 4th bullet under “Data Normalization”  
- Original: Helps when unsure about the relative importance of features.  
- Why it’s wrong: scaling says nothing about feature importance. It only puts features on a comparable range so that larger numbers don’t dominate.  
- Correct version: Useful when features are measured on very different scales.

Chapter 6 Errors
Mistake 1: Z-score output contradicts the text
Where: Z-score “find outliers” cell vs. the markdown below it
Original: Output is Outliers: [], but the markdown says “the number 100 is a clear outlier”
Why it’s wrong: The z-score of 100 is 2.615, which is inside [-3, 3], so the method does not flag it. The notebook then claims it did.
Correct: State that the z-score method missed the outlier here, and show the IQR method catching it. Or use a threshold of 2 (np.abs(z_scores) > 2).
Mistake 2: Z-score threshold of 3 is impossible with this data
Where: Z-score markdown (“Values outside [-3, 3] are usually considered outliers”) vs. the 8-value example
Original: Uses > 3 on an array of only 8 numbers
Why it’s wrong: With n = 8, the largest possible z-score is √(n−1) ≈ 2.65 (stats.zscore uses the population std), so nothing can ever exceed 3, even with an extreme value. The rule of 3 only works on larger samples.
Correct: Use a larger dataset for the demo, or explain that z-scores are unreliable on small samples. The outlier also inflates the mean and std, which masks itself.
Mistake 3: The IQR steps don’t match the code
Where: IQR markdown vs. the data.quantile() cell
Original: Q1 = median of lower half, Q3 = median of upper half
Why it’s wrong: By that method, Q1 = 12 and Q3 = 21.5, giving IQR = 9.5. The code uses pandas’ linear interpolation, which gives Q3 = 21.25 and IQR = 9.25. The text and output disagree.
Correct: Say that quantile() uses linear interpolation, so results can differ slightly from the hand method. Or change the description to match the code.

Chapter 7 Errors
Mistake 1: “Zero correlation” example contradicts the data
Where: Correlation markdown (“Zero: assignments completed ↔ grades unrelated”) vs. the first df.corr() output
Original: Assignments completed and grades are described as unrelated
Why it’s wrong: In the example data, Assignments Completed has a 0.9565 correlation with Final Grade, which is strongly positive. It’s actually the highest of the three features.
Correct: Pick a different example for zero correlation (e.g. shoe size ↔ grades), or change the data so the text matches.
Mistake 2: Filter keeps features by raw correlation, not absolute value
Where: Filter Methods markdown (“drop those with low correlation”) vs. correlations[correlations > 0.5]
Original: correlations[correlations > 0.5]
Why it’s wrong: A feature with a strong negative correlation (e.g. -0.9) is just as predictive but would be dropped. The first example in this same chapter had Extracurricular Activities at -0.79.
Correct: correlations[correlations.abs() > 0.5]
Mistake 3: The target is included in the selected features
Where: The relevant_features output
Original: final grade 1.000000 appears in the list of relevant features
Why it’s wrong: The target’s correlation with itself is always 1.0, so it passes every threshold. It isn’t a feature.
Correct: correlations.drop('final grade') before filtering.
Mistake 4: study hours and assignments completed are identical columns
Where: data_2 definition
Original: Both columns are [10, 9, 8, 7, 10, 9, 8]
Why it’s wrong: They’re perfectly duplicated (correlation 1.0), so one adds no information. The filter keeps both, which defeats the purpose of feature selection. It also makes the later methods arbitrary: RFECV keeps assignments completed and Lasso keeps study hours.
Correct: Change the data so the columns differ, or add a step that removes highly correlated features.

Chapter 8 Errors  
Mistake 1: “clean, scaled data ready for ML models” is false  
- Cell / Line: Bullet under the fit_transform cell, “Result → ...”  
- Original: Result → clean, scaled data ready for ML models.  
- Why it’s wrong: the original code drops every column except Age and Fare, and Sex and Embarked are still text.  
- Correct version: Result → Age and Fare are imputed and scaled; the other columns are passed through unchanged (Sex and Embarked still need encoding).  

Mistake 2: remainder is not set  
- Cell / Line: ColumnTransformer cell   
- Original: ColumnTransformer(transformers=[('age_fare', pipeline, ['Age', 'Fare'])])  
- Why it’s wrong: the default is remainder='drop', so Pclass, Sex, SibSp, Parch, and Embarked are silently removed.   
- Correct version: ColumnTransformer(transformers=[('age_fare', pipeline, ['Age', 'Fare'])], remainder='passthrough')  

Mistake 3: “Pipeline can be reused on new data” is misleading  
- Cell / Line: Last bullet under the ColumnTransformer explanation
- Original: Pipeline can be reused on new data for consistent preprocessing.
- Why it’s wrong: fit_transform re-learns the mean and std, so new data needs transform.
- Correct version: The fitted preprocessor can be reused on new data with transform() (not fit_transform).

Mistake 4: No train/test split (data leakage)  
- Cell / Line: Last code cell, fit_transform(X) on the full data
- Original: X_transformed = preprocessor.fit_transform(X)
- Why it’s wrong: the mean and std are learned from rows the model will later be tested on.
- Correct version:  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
X_train_transformed = preprocessor.fit_transform(X_train)  
X_test_transformed = preprocessor.transform(X_test)

<br>
  
Chapter 9 Errors  
Mistake 1: The “Before discretization” plot is drawn after binning  
- Cell / Line: Histogram cell with label='Before discretization'
- Original: plt.hist(data['Age'].dropna(), alpha=0.5, label='Before discretization')
- Why it’s wrong: data['Age'] was already converted by pd.cut, so the plot shows Adult/Elderly/Child bars, not ages.
- Correct version: plt.hist(raw_data['Age'].dropna(), alpha=0.5, label='Before discretization'), or move the plot above the pd.cut cell.

Mistake 2: The “After discretization” plot uses the wrong column  
- Cell / Line: Histogram cell with label='After discretization'
- Original: plt.hist(titanic_preprocessed[:,2], alpha=0.5, label='After discretization')
- Why it’s wrong: column 2 is car__Embarked_C, a one-hot column. Age is column 0, and it is scaled, not discretized.
- Correct version: data['Age'].value_counts().plot(kind='bar')

Mistake 3: Step 1 says Google Drive, but the code uploads a file  
- Cell / Line: Step 1 heading and cell
- Original: uploaded = files.upload()
- Why it’s wrong: the heading says “Connect Google Colab to your Google Drive”, but no Drive mount happens.
- Correct version: rename the heading to “Upload the Dataset”, or use from google.colab import drive; drive.mount('/content/drive').

Mistake 4: Typo in the transformer name  
- Cell / Line: ColumnTransformer cell
- Original: ('car', categorical_transformer, categorical_features)
- Why it’s wrong: it should be cat for categorical. The typo also shows in the output as car__Embarked_C.
- Correct version: ('cat', categorical_transformer, categorical_features)

Mistake 5: The log transform on Fare is never done  
- Cell / Line: Intro list, item 2
- Original: Data Transformation – apply log transform to Fare (skewed).
- Why it’s wrong: the code only applies StandardScaler to Fare, so no log transform happens.
- Correct version: data['Fare_log'] = np.log1p(data['Fare']) (needs import numpy as np; use log1p because the minimum Fare is 0)

Mistake 6: PassengerId is never dropped  
- Cell / Line: Intro list, item 3
- Original: Data Reduction – drop irrelevant features like PassengerId.
- Why it’s wrong: no code does this, and PassengerId still appears in the heatmap, where it is meaningless.
- Correct version: data = data.drop('PassengerId', axis=1)

Mistake 7: “Dataset is now cleaned, transformed, and ready” is false  
- Cell / Line: Text after the binning cell
- Original: Dataset is now cleaned, transformed, and ready for visualization
- Why it’s wrong: preprocessor.fit_transform(data) never modifies data. The binned Age still has 177 NaN values, and Cabin (687 missing) and Embarked (2 missing) are untouched.
- Correct version: The preprocessed array is stored in titanic_preprocessed. The Age column in data is binned, and its 177 missing values stay NaN.

Mistake 8: Histogram and KDE on binned (categorical) Age  
- Cell / Line: Age distribution plot
- Original: sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True)
- Why it’s wrong: bins=30 and kde=True mean nothing for 3 categories. The KDE curve peaks near 1150, higher than any bar, and the 177 NaN ages are silently skipped.
- Correct version: sns.countplot(x='Age', hue='Survived', data=data)

Mistake 9: The text says Sex has missing values  
- Cell / Line: Step 3 bullets, Data Cleaning
- Original: Embarked, Sex → replace with "missing"
- Why it’s wrong: Sex has 891 non-null values, so only Embarked (2 missing) needs it.
- Correct version: Embarked → replace with "missing"

Mistake 10: The Data Quality Report has no “before”  
- Cell / Line: Data Quality Report cell and text
- Original: compare the number of missing values before and after preprocessing
- Why it’s wrong: only the after count is printed, so nothing can be compared.
- Correct version: add print(data.isnull().sum()) before preprocessing.


## Note on AI tools

AI Use Disclosure
- We used an AI tool (Claude) in this activity, for two things:

1. Finding mistakes in the notebooks. We used the prompt below to have the AI review each notebook for silent mistakes. We then checked the findings against our own notebooks. At first we thought everything was correct, but the review showed many minor errors.
2. Improving our wording. We used the AI to make our sentences clearer and shorter.

Prompt used for finding mistakes:
- You are a strict reviewer of a data programming notebook. The notebook runs without errors, so do NOT look for crashes. Find the SILENT mistakes: things that run fine but are wrong.

Check for:
1. Logic errors: wrong formula, wrong column, wrong axis, wrong operator, off-by-one, wrong filter condition
2. Data handling: NaN/duplicates ignored, wrong dtype, wrong merge/join, wrong groupby or aggregation, data leakage (fit on test data), wrong train/test split
3. Stats/ML: wrong metric, evaluating on training data, wrong interpretation of results, missing random_state
4. Text vs code mismatch: comments, markdown, or conclusions that say something different from what the code or output actually shows
5. Plots: wrong title, labels, axis, units, or plot type for the data
6. Numbers in the written explanation that don't match the actual output

For EACH mistake:
- Cell / Line:
- Original (exact copy):
- Why it's wrong (1-2 sentences):
- Correct version:

Rules:
- Only report real errors. Put uncertain ones under "Possible issues" with your doubt.
- Compare every written claim against the actual output shown.
- End with a summary table: # | Cell | Type.


## References

McKinney, W. (2021). Python for Data Analysis, 3rd ed. O'Reilly.
VanderPlas, J. Python Data Science Handbook.
Any other page or article you used.
