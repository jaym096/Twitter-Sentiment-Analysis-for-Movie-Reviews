# Twitter-Sentiment-Analysis-for-Movie-Reviews

**Tools and Technology Used:** Python, NLTK, Sk-Learn

Used movie related twitter data using **‘tweepy’** API to summarizes user’s sentiment towards a particular movie. Trained the model 
against rt-polarity dataset using classification algorithms: **Gaussian Naïve Bayes** and **Support Vector Machine**. Classified 
sentiments into positive, negative, and neutral, performing visual analysis using word cloud, bar graph and pie chart 

### STEPS TO RUN THE CODE:
	1. The Data folder contains all the Datasets required for the Project.
	2. The twitter data to be generated needs security credentials.
	3. Replace security credentials in config.json file present in Data folder.
	4. To Run the Movie Review Classifiers type on the console,
		python sentiment_analysis.py NAME ALGO OUTPUT
			positional arguments:
			  NAME     Name of the movie for which you want to get the analysis
			  ALGO     Classification Algorithm to be used (gnb svm)
			  OUTPUT   Graphical representation of the analysis (Bargraph(bar) Wordcloud(WC) Piechart(pie))

		example: to run the code for movie andhadhun using SVM and get result in piechart:
			 python sentiment_analysis.py andhadhun svm pie
