# Twitter-Sentiment-Analysis-for-Movie-Reviews
Uses tweet to identify user's general sentiment towards a particular movie

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

		example: to run the code for movie andhadhun using SVM and get result as piechart:
			 python sentiment_analysis.py andhadhun svm pie
