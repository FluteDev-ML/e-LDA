# Enhanced-Topic-Modeling
This Enhanced Topic Modeling used the strenght of wrod2vec embedding. 
Here some following steps to run the code

1. You should have a CSV file that be a input for the the Model
2. Make sure that the column name of a CSV file should be "descriptiin" that contain the textuall data. You can change the name of the column but accordingly you have to chang the name in the code also.
3. Database should be in the same location where the code is running or you have to pass the location of the database.
4. the visualization is done using pyLDAvis that store in .html file and PCA ploting
5. the code enhances the coharence score of the LDA topic and improves the interpretability.

Important: This code can also deal with the dataset where some of the column has NaN values

NOTE: 
1.Make sure the location of the file is correct.
2. Need previous knowledge of LDA to understand the OUTPUT of eLDA and its corresponding visualization.
