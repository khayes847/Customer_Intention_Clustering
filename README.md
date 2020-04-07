# Customer_Intentions_Clustering

## Created By:

* __Kyle Hayes__

## Project Details:

Raw customer data is often times unsuitable for supervised machine learning, because they come in an unlabeled and uncategorized form. Despite this, we can still use these datasets to classify customer categories using clustering techniques. The purpose of this project is to use clustering techniques to classify unsupervised data related to online customer shopping, to label the categories based on how each category differs from the others, and to perform classification on the labelled data using machine learning algorithms.

- **Business Understanding**:
 Like their physical counterparts, online stores have a vested interest in obtaining data related to customer demographics, as well as how they interact with the store's online infrastructure. Google Analytics allows businesses to collect a wide variety of data regarding individual visits to online retailers, including the location of the visitors, which pages they spend time on, and whether they purchase items at the end of their visits. Although this data is unsupervised, if online businesses are able to use this data to classify visitors based on their demographics and their actions, they will be able to better anticipate customer needs and maximize customer purchases.

- **Data Understanding**:
 I obtained the "Online Shoppers Purchasing Intention" dataset from the UCI Machine Learning Repository. The purpose of the original study was to predict whether an online shopper was likely or unlikely to make a purchase. The authors obtained from Google Analytics 12230 instances in which a separate potential customer entered a retail website. The dataset includes the following information about each instance:
    - Administrative: The number of account management pages the user visited (ex. login pages, password recovery pages, etc.). These are pages defined by one of the following urls: /?login, /?logout, /LoginRegister, /login', /passwordrecovery, /?ref, /?refer, /?returnurl, /customer, /emailwishlist, /omnicards.
    - Administrative_Duration: Total amount of time (in seconds) spent by the visitor on administrative pages.
    - Informational: The number of pages the user visited about the website, the store address, or store communications (ex. store locations, the "Contact Us" page, etc.). These pages are defined by one of the following urls: /Topic, /t-popup, /t, /contactus, /Catalog, /stores.
    - Informational_Duration: Total amount of time (in seconds) spent by the visitor on informational pages.
    - Product: The number of pages the user visited related to products (ex. product searches, the "shopping cart" page, etc.). These are pages defined by one of the following urls: /, /c, /urun, /search, /cart.
    - Product_Duration: Total amount of time (in seconds) spent by the visitor on product pages.
    - Bounce Rates: The average bounce rate for the pages visited. A page's bounce rate is defined as the overall percentage of users that enter a site using the page in question, and leave without interacting with the page.
    - Exit Rates: The average exit rate for the pages visited. A page's exit rate is defined as the percent of users that leave the website immediately after interacting with the page.
    - Page Value: The average page value for the pages visited. A page's page value is defined by the average value that the page produces (how much users spend on average after visiting the page).
    - Special Day: How close the visit was to a holiday for which users commonly purchase gifts (ex. Mother's Day, Valentine's Day, etc.). The authors normalized this variable to a [0, 1] range.
    - Month: The month in which the visit took place.
    - Operating System: The operating system (ex. Windows, IOS, etc.) used by the shopper.
    - Browser: The browsing system (ex. Chrome, Safari, etc.) used by the shopper.
    - Region: The geographic region from which the shopper initiated the session.
    - Traffic Type: The traffic source by which the shopper accessed the website (ex. direct, banner, etc.).
    - Visitor Type: Whether the shopper was a new visitor, a returning visitor, or "other". Note: the authors do not describe what "other" refers to.
    - Weekend: Boolean, describes whether the visit took place on a weekend.
    - Revenue: Boolean, describes whether the visit ended in a transaction.

Please note that for the variables "Operating System", "Browser", "Region", and "Traffic Type", the authors anonymized the data by substituting each categorical variable by a number. We will therefore be referring to each category within these variables by their number (ex. "Browser_1, etc.).

- **Data Preparation**:
 Prior to clustering the data, we will need to clean the data. This function will transform the data in the following ways:
    - It will transform each nominal categorical feature with an integer datatype to a string datatype.
    - For consistency, it will convert each integer or Boolean datatype to a float datatype.
    - For each categorical feature, it will combine categories with low numbers of datapoints into an "other" category.
    - It will create dummy variables for each categorical variable with more than two categories. In order to ensure multicollinearity, it will then remove the most common category.
    - In order to increase normality, it will preform a log transformation on each continuous feature. Since these often include zero values, it will first find the minimum non-zero value for each feature, and add 1/2 this number to each zero value.
    - In order to decrease outlier influence, it will cap each continuous feature's outlier values at their 99.97% quantile.
    - To prepare for clustering, it will standardize the data using the StandardScaler().

- **Modeling**:
 In order to first obtain labels for the unsupervised data, I will determine how our data preforms when clustered into a range of 2 to 10 clusters using the following clustering techniques:
    - K-Means without PCA
    - Agglomerative Hierarchical Clustering without PCA
    - K-Means with PCA (dimensions determined by 95% variance retention level)
    - Agglomerative Hierarchical Clustering with PCA (dimensions determined by 95% variance retention level)
    I will use as a metric the Silhouette Score.
    
    Then, I will use the clustering algorithm to obtain labels for the unclassified data. I will then use Random Forests Feature Importance to determine which features the algorithm is using to clssify the data, and explore these features. I will use the results of this analysis to create labels for each data cluster that reflect their respective categories.

- **Evaluation**:
 In order to evaluate how good each label is at predicting individual characteristics, I will train a logistic regression algorithm to classify each data point based on their category label. Prior to this, since the dataset is unbalanced, I will use SMOTE to create synthetic data points for the training dataset.
 
- **Deployment**:
  I will save and make available my final Keras model. In addition, I will include a diagram demonstrating the steps to productionizing this process.
  
## Files in Repository:

* __README.md__ - A summary of the project.

* __technical_notebook.ipynb__ - Step-by-step walk through of the modeling/optimization process with rationale explaining decisions made. 

* __data_prep.py__ - Gathers and prepares data for analysis.

* __functions.py__ - Provides general functions used in the technical notebook.

* __productionization_diagram.jpg__ - A diagram demonstrating the steps to productionizing this process.
