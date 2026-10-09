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

Chapter 4 Errors 

- Mistake 1: Ordinal encoding values don’t match the code              
Cell / Line: Markdown under “Ordinal Encoding” (page 3), vs. the OrdinalEncoder cell output       
Original: Little → 1, Medium → 2, Lots → 3       
Why it’s wrong: OrdinalEncoder starts counting at 0, and the output shows 0.0, 1.0, 2.0.          
Correct version: Little → 0, Medium → 1, Lots → 2          

- Mistake 2: One-hot example doesn’t match the column order              
Cell / Line: Markdown under “One-hot Encoding” (page 3), vs. the pd.get_dummies output     
Original: Sunny → [1,0,0], Cloudy → [0,1,0], Rainy → [0,0,1]    
Why it’s wrong: get_dummies orders the columns alphabetically (Cloudy, Rainy, Sunny), not by order of appearance.           
Correct version: Cloudy → [1,0,0], Rainy → [0,1,0], Sunny → [0,0,1]

- Mistake 3: Output shows True/False, but the text says 1/0             
Cell / Line: “Assigns 1 = True, 0 = False” (page 3), vs. the get_dummies cell             
Original: df_encoded = pd.get_dummies(df_2, columns=['Weather'])               
Why it’s wrong: pandas 2.x returns booleans (True/False), not the 1/0 integers the text describes.            
Correct version: df_encoded = pd.get_dummies(df_2, columns=['Weather'], dtype=int)                  

- Mistake 4: “very hot” is never used               
Cell / Line: “define bins and labels” cell (page 2)            
Original: bins = [70, 75, 85, 95, 100]                 
Why it’s wrong: the highest temperature, 95, falls in (85, 95], which is “hot”, so no row is ever “very hot”. df.head() only shows 5 of 7 rows, which hides this.            
Correct version: bins = [70, 75, 85, 90, 100] (now 91 and 95 become “very hot”)              

Chapter 5 Errors 
- Mistake 1: The output is called a "dataset", but it's a NumPy array
Original: scaled_data = scaler.fit_transform(df) with the text "The outcome is a new dataset where the scales..."
Why it's wrong: fit_transform returns a NumPy array, not a DataFrame, so the column names are lost and the printout has no headers.
Correct: scaled_df = pd.DataFrame(scaled_data, columns=df.columns), then show scaled_df as the last line of the cell.

- Mistake 2: scaler is overwritten
Original: scaler = MinMaxScaler()
Why it's wrong: the same name was already used for StandardScaler, so the first fitted scaler is lost. Running cells out of order can silently use the wrong one.
Correct: standard_scaler = StandardScaler() in the first cell, and minmax_scaler = MinMaxScaler() with normalized_data = minmax_scaler.fit_transform(df_2) in the second.

- Mistake 3: "Helps when unsure about the relative importance of features"
Why it's wrong: scaling doesn't tell you anything about feature importance. It only puts features on a comparable range so that scale doesn't dominate. This one is a wrong concept in the text.
Correct: "Useful when features are measured on very different scales."

Chapter 8 Errors 

- Mistake 1: "clean, scaled data ready for ML models" is false
Original: Result → clean, scaled data ready for ML models.
Why it's wrong: the original code drops every column except Age and Fare, and Sex and Embarked are still text.
Correct: Result → Age and Fare are imputed and scaled; the other columns are passed through unchanged (Sex and Embarked still need encoding).

- Mistake 2: remainder is not set
Original: ColumnTransformer(transformers=[('age_fare', pipeline, ['Age', 'Fare'])])
Why it's wrong: the default is remainder='drop', so Pclass, Sex, SibSp, Parch, and Embarked are silently removed.
Correct: ColumnTransformer(transformers=[('age_fare', pipeline, ['Age', 'Fare'])], remainder='passthrough')

- Mistake 3: "Pipeline can be reused on new data" is misleading
Original: Pipeline can be reused on new data for consistent preprocessing.
Why it's wrong: fit_transform re-learns the mean and std, so new data needs transform.
Correct: The fitted preprocessor can be reused on new data with transform() (not fit_transform).

- Mistake 4: No train/test split (data leakage)
Original: X_transformed = preprocessor.fit_transform(X)
Why it's wrong: the mean and std are learned from rows the model will later be tested on.
Correct: python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


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
