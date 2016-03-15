Running the Web Scrapper:
In the folder Pythoncodes_MLProject1, you will get a folder called final. It contains the subfolder final and inside it go to
the spiders subfolder. It contains the python file called preet which contains the python code for our crawler.
To run this code, make sure you have scrapy package available in python installed and its path set.
Go and open cmd and give the following command line:
scrapy crawl --nolog --output=final.csv preet 
and it will run an d you will find final.csv file formed in side your spiders folder.


Running Closed Form method and Ridge Regression method for predicting target:
Put the OnlineNewsPopularity.csv and final.csv in the same folder as Closed-form.py
Run the file Closed-form.py. For both part of the project, the program runs 10 times. It print and plot the training errors and test errors of both closed-form model and ridge-regression model. It also prints the best penalty size for each run after evaluated by cross validation. 



Running Gradient Descent method for predicting target:
Put the OnlineNewsPopularity.csv and final.csv in the same folder as Closed-form.py
Run the file gradientDescent.py. Then you will see that there is promo text asking you which part of data you want test. If you want test part 1 data, just enter 1 , otherwise, eneter 2. Then you will get all the pictures, printing results.


