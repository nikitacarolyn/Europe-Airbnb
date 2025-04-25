**Introduction:**
To analyse the dataset, we use inferential tools and machine learning methods like hypothesis, simple linear regression, random forest etc. 
Firstly, we start by the formulation of hypothesis based on the business questions that we are targeting. In order to do this, it is critical to identify the independent and dependent variables along with the target that the analysis is aiming towards. 
The dataset consists of 19 variables that have relevant and significant impact on Airbnb listings specifically in Europe for the selected dataset. The identification and classification of these variables based on my business question are as follows:

**Independent variables:**
1. City: It refers to the 9 European cities - Amsterdam, Athens, Barcelona, Berlin, Budapest, Lisbon, Paris, Rome, Vienna. It is a highly important variable as it drives the demand and experiences of tourists. Cities with higher demand have more highly prices Airbnbs. 
2. Day: Demand and price of Airbnbs depend on the day of stay – weekdays and weekends. Typically weekends and holidays are more expensive since the demand is high. 
3. Room Type: Depending on the number of occupants, there are different room types that can be booked – Entire apartment, Private Room, and Shared apartment. 
4. Shared Room: In case of solo travellers, one might find it more economical to share a room. This variable is a Boolean value – True or False.
5. Private Room: If the Airbnb has private rooms available for booking. 
6. Person Capacity: The maximum capacity the Airbnb can hold. Hosts can charge extra in exceptional cases.
7. Superhost: If the Airbnb host is a Superhost or not based on the reviews.
8. Multiple Rooms: If the Airbnb has multiple room (2-4) rooms. 
9. Business: If the Airbnb has any business offers (here, 4 or more)
10. Cleanliness Rating: Rating given by the hospitality management or previous occupants of the Airbnb based on the cleanliness and hygiene.
11. Guest Satisfaction: The satisfaction score given by guests who stayed at the Airbnb. 
12. Bedrooms: Number of bedrooms in an Airbnb. 
13. City Centre (km): Distance of the Airbnb from the city centre. Price highly depends on this distance. 
14. Metro Distance (km): Distance of the Airbnb from public transport. This is a connivence variable. 
15. Attraction Index and Normalised Attraction Index: An index to determine the proximity of the Airbnb from sights or attractions in the city.
16. Restaurant Index and Normalised Restaurant Index: An index to determine the proximity of the Airbnb from restaurants in the city.

**Dependent variables:** 
Price: The price of different Airbnb listings is influenced by several different factors given in the dataset. It is a good target variable for analysis as it has significance in business insights, market analysis as well as demand forecasting. 

**Data Analysis:**
We conduct a pilot analysis in order to take a decision regarding our hypothesis based on our business question using regression models. 
To do this, we use two statistical tests – Random Forest and Linear Regression.
To analyse Price of Airbnbs we checked for outliers, by visualising it using a boxplot.
We modelled the data using Linear regression and Random forest regression and on comparing the two regression models - we can conclude that the Random Forest Regression model is a better fit than the linear regression model as the R square value is higher and the error values are lower in comparison. This model answers the business question - **The independent variables do significantly affect the dependent variable.**
- We then plot the line of best fit for both the models for a better visual representation.
The Random Forest regression model has a better line of best fit and the data points are less scattered. Therefore, there are not that many outliers in the model but this could also be owing to the models insensitivity to outliers.
The data points of the Linear Regression model do not fall on the line of best fit. Majority of the values are clustered at the bottom of the line and are not spread. There outliers are evident in the model and hence the model will have to deal with them effectively. Owing to all this, the model has a low R square value which does not help in accurately answering our business question that the independent variables have a significant effect on the target variable – Price.

**Conclusions and Recommendations:**
Now from the analysis of Europe Airbnb dataset using a number of different processes, tools, and models, we can make some conclusions: 
1. The dataset is limited to European countries for Airbnb data and therefore is biased – excluding patterns and considerations of people living in Europe
2. Data points like whether there is a café in the Airbnb is not mentioned in the dataset which brings up the question of availability of food like complimentary meals, etc which can affect the choice of a guest to book the Airbnb.
3. Europe is a continent with many different countries – 9 in the dataset, having many different languages. Language can be a barrier for tourists from other countries, therefore the language spoken at the Airbnb can be an important metric to be mentioned in the data set
4. The mention of availability of different functions in the Airbnb can be crucial in a guest decision of which Airbnb do book at. Features like the kitchen facilities, whether there is a fridge, provision of hot water and good ventilation, etc
5. Price being the target variable is crucial and it is dependent on all the other variables. Now, whether the price for additional facilities is charged separately or is considered under the same price can skew the guests decision in booking the Airbnb.
6. Collection of feedback from the guests should be encouraged in order to go over and implement the recommended changes and suggestions given by them for future improvement.
Referral codes for discount on the stay in the Airbnb can also be implemented in order to attract more guests. 

