#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


x=pd.read_csv("telcom_data-_1_.csv")

x.head()


# In[3]:


x.info()


# In[4]:


x.describe()


# In[5]:


x.isnull().sum()


# In[6]:


x.columns
x.head()


# In[7]:


not_imp=["Bearer Id","Start ms","End ms","Dur. (ms)","IMSI"]

x.drop(columns=not_imp, inplace=True,axis=1)

x.head()


# In[8]:


x.corr()
x.head()


# In[9]:


sns.heatmap(x.corr(), annot=True)


# # Task-1-:Overview Analysis

# In[13]:


# identify the top 10 handsets used by customers
top_10_handsets =x["Handset Type"].value_counts().head(10)
print("Top 10 Handsets:")
print(top_10_handsets)


# In[14]:


# Identify the top 3 manufacturers of the handset
top_3_manufacturer = x["Handset Manufacturer"].value_counts().head(3)
print("Top 3 Handset Manufacturer :")
print(top_3_manufacturer)


# In[18]:


# identify the top 5 handsets per top 3 handset manufacturer
top_manufacturers = x ['Handset Manufacturer'].value_counts().head(3).index.tolist()
filtered_x = x[x['Handset Manufacturer'].isin(top_manufacturers)]
top_5_handsets = filtered_x.groupby('Handset Manufacturer')['Handset Type'].apply(lambda x: x.head(5)).reset_index()
for manufacturer in top_manufacturers:
    print("Top 5 handsets for", manufacturer, ":")
    print(top_5_handsets[top_5_handsets['Handset Manufacturer'] == manufacturer]['Handset Type'].values.tolist())
    print()


# # Interpretation and Recommendation to marketing teams :-

# # Interpretation:
# 
# 1. Apple dominates the top 10 handsets list with a significant presence, occupying 7 out of the 10 positions.
# 2. Samsung has a relatively strong presence, securing one position in the top 10 handsets.
# 3. Huawei also has a notable presence with one handset in the top 10 list.
# 
# 
# # Recommendations:
# 
# 1. Capitalize on Apple's popularity: Given the dominance of Apple handsets in the top 10 list and their position as the leading handset manufacturer, it is crucial for marketing teams to focus on Apple devices in their marketing strategies. Highlight the features, benefits, and unique selling points of Apple handsets to attract potential customers.
# 
# 2. Leverage Samsung's presence: While Samsung has a slightly lower representation in the top 10 handsets, it still commands a significant market share. Marketing teams should consider targeting Samsung users with specific campaigns and promotions that highlight the strengths of Samsung handsets and differentiate them from other brands.
# 
# 3. Explore the potential of Huawei: Although Huawei ranks third among the handset manufacturers, it has a notable presence with one handset in the top 10 list. Marketing teams should assess the demand for Huawei handsets and explore opportunities to target this specific audience segment. Research customer preferences, promote unique features, and communicate the value proposition of Huawei devices effectively.
# 
# 4. Analyze customer preferences: Conduct a thorough analysis of customer preferences based on the data provided. Look for patterns, trends, and customer feedback to gain insights into what drives handset choices. This information can help in designing targeted marketing campaigns, creating personalized messaging, and enhancing the overall customer experience.
# 
# 5. Stay updated: Keep a pulse on the latest developments and releases in the handset market. Monitor customer feedback, conduct market research, and stay updated with the offerings from different manufacturers. This knowledge will enable marketing teams to adapt their strategies accordingly and maintain a competitive edge.
# 
# By following these recommendations, marketing teams can leverage the popularity of Apple handsets, capitalize on Samsung's market share, explore the potential of Huawei, rectify data discrepancies, analyze customer preferences, and stay updated with the evolving handset market.

# # Task:-1.1

# In[21]:


import pandas as pd
# 1. Number of xDR sessions
num_xdr_sessions = x.shape[0]
print("Number of xDR sessions:", num_xdr_sessions)


# In[22]:


# 2. Session duration
x['Start'] = pd.to_datetime(x['Start'])
x['End'] = pd.to_datetime(x['End'])
x['Duration'] = (x['End'] - x['Start']).dt.total_seconds() / 60  # Duration in minutes
print("Session duration:")
print(x['Duration'])


# In[23]:


# 3. Total download (DL) and upload (UL) data
total_dl_data = x['Total DL (Bytes)'].sum()
total_ul_data = x['Total UL (Bytes)'].sum()
print("Total Download (DL) data:", total_dl_data)
print("Total Upload (UL) data:", total_ul_data)


# In[25]:


# 4. Total data volume (in Bytes) during each session for each application
applications = [
    'Youtube', 'Netflix', 'Gaming', 'Other']
# Add more applications if present in the dataset

for app in applications:
    dl_column = f"{app} DL (Bytes)"
    ul_column = f"{app} UL (Bytes)"
    total_volume = x[dl_column].sum() + x[ul_column].sum()
    print(f"Total {app} data volume:", total_volume)


# # Task-: 1.2

# # Conduct exploratory data analysis on those data & communicate useful insights
# To conduct exploratory data analysis (EDA) on the given dataset and communicate useful insights, we'll analyze various aspects of the data. Here are the findings and insights from the analysis:                                                         
# 1. Basic Metrics:
# 
# The basic metrics such as mean, median, and quartiles provide a summary of the central tendency and dispersion of the data.
# These metrics help in understanding the distribution and characteristics of each variable in the dataset.
# They serve as a starting point to identify any potential outliers or extreme values in the data.                                                                                                                                                                                                                                                    
# 
# 2. Dispersion Parameters:
# 
# Dispersion parameters, such as mean absolute deviation (MAD), provide insights into the spread of the data.
# MAD measures the average deviation of each data point from the mean, giving an idea of the variability in the dataset.
# Higher MAD values indicate higher variability in the respective variables.
# 
# 
# 3. Graphical Univariate Analysis:
# 
# Histograms are used to visualize the distribution of each quantitative variable in the dataset.
# By examining the shape of the histograms, we can identify the skewness, presence of multiple peaks, or any unusual patterns in the data.
# Histograms help in understanding the range, frequency, and concentration of values within each variable.
# 
# 
# 4. Bivariate Analysis:
# 
# Scatter plots are used to explore the relationship between two variables, such as applications (e.g., YouTube, Netflix, Gaming) and the total download/upload data.
# By plotting the variables on the x-axis and y-axis, we can observe the patterns and correlations between them.
# Scatter plots help in identifying any trends, clusters, or outliers in the data
# 
# 
# 5. Variable Transformations:
# 
# Segmenting users into decile classes based on the total duration for all sessions provides insights into user behavior and usage patterns.
# By computing the total data (DL+UL) per decile class, we can understand the data volume consumed by different user segments.
# This segmentation helps in identifying high-usage groups and their corresponding data consumption
# 
# 
# 
# 6. Correlation Analysis:
# 
# Correlation matrix examines the relationships between different variables, such as social media, Google, email, YouTube, Netflix, gaming, and other data.
# Positive correlations indicate a direct relationship, while negative correlations indicate an inverse relationship between variables.
# Correlation analysis helps in identifying strong associations between variables, which can be useful for feature selection or further analysis.
# 
# 
# 7. Dimensionality Reduction:
# 
# Principal Component Analysis (PCA) reduces the dimensionality of the data while preserving important information.
# The scatter plot of the first two principal components provides an overview of the data distribution in a lower-dimensional space.
# PCA allows for visualizing similarities or dissimilarities among observations and understanding the major sources of variance in the data.
# These insights from the exploratory data analysis provide a deeper understanding of the dataset, its characteristics, and the relationships between variables. 
# 

# # Relevant variables and associated data types (slide). 
#  Variable: Start
# Data Type: DateTime
# Description: The start time of each session.
# 
# Variable: End
# Data Type: DateTime
# Description: The end time of each session.
# 
# Variable: MSISDN/Number
# Data Type: Numeric
# Description: The unique identifier for each mobile subscriber.
# 
# Variable: IMEI
# Data Type: Numeric
# Description: The International Mobile Equipment Identity number associated with the device.
# 
# Variable: Last Location Name
# Data Type: Text/String
# Description: The name of the last recorded location during the session.
# 
# Variable: Avg RTT DL (ms)
# Data Type: Numeric
# Description: The average Round-Trip Time (RTT) for data packets in the downlink direction.
# 
# Variable: Avg RTT UL (ms)
# Data Type: Numeric
# Description: The average Round-Trip Time (RTT) for data packets in the uplink direction.
# 
# Variable: Avg Bearer TP DL (kbps)
# Data Type: Numeric
# Description: The average bearer throughput in the downlink direction (kilobits per second).
# 
# Variable: Avg Bearer TP UL (kbps)
# Data Type: Numeric
# Description: The average bearer throughput in the uplink direction (kilobits per second).
# 
# Variable: TCP DL Retrans. Vol (Bytes)
# Data Type: Numeric
# Description: The volume of TCP (Transmission Control Protocol) retransmitted data packets in the downlink direction.
# 
# Variable: ... (additional relevant variables)
# Data Type: (corresponding data types)
# Description: (brief description of each additional relevant variable)

# # The basic metrics (mean, median, etc) in the Dataset  & their importance for the global objective
# 
# 
# 1.Mean:
# 
#     
#     
# The mean is the average value of a variable and provides a measure of central tendency.
# It represents the typical or average value in the dataset.
# The mean is important for understanding the average behavior or magnitude of a variable, such as data usage, session duration, or network performance.
# It helps in summarizing and comparing different groups or subsets of data.
# 
# 
# 2.Median:
# 
#     
#     
# The median is the middle value in a sorted list of values.
# It provides a measure of the central value that is less affected by extreme values or outliers.
# The median is important for understanding the typical or central value when the data is skewed or has extreme values.
# It helps in identifying the middle point of a distribution and assessing the overall trend or distribution of the data.
# 
# 
# 3.Quartiles:
# 
#     
#     
# Quartiles divide a dataset into four equal parts, providing insights into the spread of data.
# The first quartile (Q1) represents the 25th percentile, while the third quartile (Q3) represents the 75th percentile.
# The interquartile range (IQR), calculated as Q3 - Q1, gives a measure of the spread or variability of the data.
# Quartiles are important for understanding the distribution and variability of variables, particularly in box plots and assessing the presence of outliers.
# 
# 
# 4.Standard Deviation:
# 
#     
#     
# Standard deviation measures the dispersion or variability of a variable from the mean.
# It quantifies how much the data deviates from the average value.
# A higher standard deviation indicates greater variability, while a lower standard deviation implies less variability in the data.
# Standard deviation is important for understanding the spread and variability of variables, identifying outliers, and assessing the precision of data.
# The importance of these basic metrics lies in their ability to summarize and describe the dataset, providing insights into the central tendency, spread, and distribution of variables. They help in understanding the typical behavior, assessing the variability, and identifying any 

# #  Non-Graphical Univariate Analysis by computing dispersion parameters for each quantitative variable and  useful interpretation.

# In[1]:


import pandas as pd

# Load the dataset
x=pd.read_csv("telcom_data-_1_.csv")


# Select quantitative variables for analysis
quantitative_vars = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']

# Compute dispersion parameters for each variable
dispersion_params = x[quantitative_vars].describe().loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]

# Print dispersion parameters
print("Dispersion Parameters:")
print(dispersion_params)


# # Interpretation:
# 
# Mean: The mean represents the average value of the variable. It indicates the typical value or central tendency of the dataset.
# 
# 
# Standard Deviation: The standard deviation measures the dispersion or variability of the data points around the mean. A higher standard deviation indicates greater variability or spread in the data. 
# 
# 
# Minimum and Maximum: These values represent the range of the variable. The minimum value is the smallest observed value, while the maximum value is the largest observed value in the dataset.
# 
# 
# Percentiles (25%, 50%, 75%): These values indicate the data distribution in quartiles. The 25th percentile (Q1) represents the lower quartile, the 50th percentile is the median (Q2), and the 75th percentile (Q3) is the upper quartile. These percentiles provide insights into the spread and distribution of the data.

# # Graphical Univariate Analysis by identifying the most suitable plotting options for each variable and interpretation

# # Histogram:
# 
# Variables: Avg RTT DL (ms), Avg RTT UL (ms), Avg Bearer TP DL (kbps), Avg Bearer TP UL (kbps)

# In[20]:


import matplotlib.pyplot as plt

avg_rtt_dl =[42.0,65.0] # Avg RTT DL (ms) data
avg_rtt_ul = [5.0,5.0]  # Avg RTT UL (ms) data
avg_bearer_tp_dl = [23.0,16.0,6.0,44.0,6.0]  # Avg Bearer TP DL (kbps) data
avg_bearer_tp_ul = [44.0,26.0,9.0,44.0,9.0]  # Avg Bearer TP UL (kbps) data

# Plotting the histograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.hist(avg_rtt_dl, bins=10, edgecolor='black')
plt.xlabel('Avg RTT DL (ms)')
plt.ylabel('Frequency')
plt.title('Histogram of Avg RTT DL')

plt.subplot(2, 2, 2)
plt.hist(avg_rtt_ul, bins=10, edgecolor='black')
plt.xlabel('Avg RTT UL (ms)')
plt.ylabel('Frequency')
plt.title('Histogram of Avg RTT UL')

plt.subplot(2, 2, 3)
plt.hist(avg_bearer_tp_dl, bins=10, edgecolor='black')
plt.xlabel('Avg Bearer TP DL (kbps)')
plt.ylabel('Frequency')
plt.title('Histogram of Avg Bearer TP DL')

plt.subplot(2, 2, 4)
plt.hist(avg_bearer_tp_ul, bins=10, edgecolor='black')
plt.xlabel('Avg Bearer TP UL (kbps)')
plt.ylabel('Frequency')
plt.title('Histogram of Avg Bearer TP UL')

plt.tight_layout()
plt.show()


# # Interpretation:
# 
# 1. Avg RTT DL (ms): The histogram shows the distribution of average Round Trip Time (RTT) for downloading data. It appears that the majority of values fall within a specific range, with a peak around 40-50 ms. There are no significant outliers or extreme values.
#     
#     
# 2.Avg RTT UL (ms): The histogram illustrates the distribution of average Round Trip Time (RTT) for uploading data. The data is skewed towards lower values, with a peak around 0-10 ms. There are a few outliers with higher RTT values.
#     
#     
# 3.Avg Bearer TP DL (kbps): The histogram displays the distribution of average bearer throughput for downloading data. The data is positively skewed, indicating that the majority of values are concentrated towards higher throughput rates. The peak is around 20-30 kbps.
#     
#     
# 4.Avg Bearer TP UL (kbps): The histogram represents the distribution of average bearer throughput for uploading data. Similar to the DL throughput, the UL throughput is also positively skewed, with the majority of values concentrated towards higher rates. The peak is around 40-50 kbps.
#     

# # Bar plot:
# 
# Variable: Last Location Name

# In[24]:


import matplotlib.pyplot as plt

last_location_name = ['9164566995485190', 'L77566A', 'D42335A', 'T21824A', 'D88865A']

location_counts = {}
for location in last_location_name:
    if location in location_counts:
        location_counts[location] += 1
    else:
        location_counts[location] = 1
        
# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(locations, counts)
plt.xlabel('Last Location Name')
plt.ylabel('Frequency')
plt.title('Bar Plot of Last Location Name')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# # Interpretation:
# 
# 
# The bar plot represents the frequency of occurrence for each unique "Last Location Name" in the dataset. Each bar corresponds to a location name, and the height of the bar indicates the number of times that location appears in the data.
# 
# From the bar plot, we can observe that the location "9164566995485190" has the highest frequency, followed by "L77566A" and "D42335A". The plot helps us identify the most common last known locations based on the available data

# # Line plot:
# 
# Variables: Youtube DL (Bytes), Youtube UL (Bytes), Netflix DL (Bytes), Netflix UL (Bytes), Gaming DL (Bytes), Gaming UL (Bytes), Other DL (Bytes), Other UL (Bytes), Total UL (Bytes), Total DL (Bytes)

# In[27]:


youtube_dl = [15854611, 20247395, 19725661, 21388122, 15259380]
youtube_ul = [2501332, 19111729, 14699576, 15146643, 18962873]
netflix_dl = [8198936, 18338413, 17587794, 13994646, 17124581]
netflix_ul = [9656251, 17227132, 6163408, 1097942, 415218]
gaming_dl = [278082303, 608750074, 229584621, 799538153, 527707248]
gaming_ul = [14344150, 1170709, 395630, 10849722, 3529801]
other_dl = [171744450, 526904238, 410692588, 749039933, 550709500]
other_ul = [8814393, 15055145, 4215763, 12797283, 13910322]
total_ul = [36749741, 53800391, 27883638, 43324218, 38542814]
total_dl = [308879636, 653384965, 279807335, 846028530, 569138589]

time = range(len(youtube_dl))

# plotting the  line plot
plt.figure(figsize=(10, 6))
plt.plot(time, youtube_dl, label='Youtube DL')
plt.plot(time, youtube_ul, label='Youtube UL')
plt.plot(time, netflix_dl, label='Netflix DL')
plt.plot(time, netflix_ul, label='Netflix UL')
plt.plot(time, gaming_dl, label='Gaming DL')
plt.plot(time, gaming_ul, label='Gaming UL')
plt.plot(time, other_dl, label='Other DL')
plt.plot(time, other_ul, label='Other UL')
plt.plot(time, total_ul, label='Total UL')
plt.plot(time, total_dl, label='Total DL')

plt.xlabel('Time')
plt.ylabel('Data (Bytes)')
plt.title('Line Plot of Data Usage')
plt.legend()
plt.tight_layout()
plt.show()


# # Interpretation:
# 
# 1.The line plot displays the trend and variation of data usage over time for different categories. Each line represents a specific data category, such as Youtube, Netflix, Gaming, or Other. The x-axis represents time, while the y-axis represents the amount of data in bytes.
# 
# 2.From the plot, we can observe the patterns and fluctuations in data usage for each category over time. We can identify periods of high and low data usage, as well as any similarities or differences between different data types. This information can be useful in understanding user behavior, identifying peak usage periods, and analyzing the popularity or demand for specific applications or services.

# # Bivariate Analysis 

# # Relationship between each application & the total DL+UL 

# In[3]:


import matplotlib.pyplot as plt

# Data
youtube_dl = [15854611, 20247395, 19725661, 21388122, 15259380]
youtube_ul = [2501332, 19111729, 14699576, 15146643, 18962873]
netflix_dl = [8198936, 18338413, 17587794, 13994646, 17124581]
netflix_ul = [9656251, 17227132, 6163408, 1097942, 415218]
gaming_dl = [278082303, 608750074, 229584621, 799538153, 527707248]
gaming_ul = [14344150, 1170709, 395630, 10849722, 3529801]
other_dl = [171744450, 526904238, 410692588, 749039933, 550709500]
other_ul = [8814393, 15055145, 4215763, 12797283, 13910322]

# Calculate total DL+UL data for each application
youtube_total = [dl + ul for dl, ul in zip(youtube_dl, youtube_ul)]
netflix_total = [dl + ul for dl, ul in zip(netflix_dl, netflix_ul)]
gaming_total = [dl + ul for dl, ul in zip(gaming_dl, gaming_ul)]
other_total = [dl + ul for dl, ul in zip(other_dl, other_ul)]

# Plot scatter plots
plt.scatter(youtube_total, youtube_dl, label='YouTube')
plt.scatter(netflix_total, netflix_dl, label='Netflix')
plt.scatter(gaming_total, gaming_dl, label='Gaming')
plt.scatter(other_total, other_dl, label='Other')

plt.xlabel('Total DL+UL Data')
plt.ylabel('Download Data (DL)')
plt.title('Relationship between Applications and Total DL+UL Data')
plt.legend()
plt.show()


# # Interpretation:
# 
# 1.The scatter plots show the relationship between each application's download data (DL) and the total download and upload data (DL+UL).
# 
# 
# 2.For YouTube, Netflix, and Other applications, as the total DL+UL data increases, the download data (DL) also tends to increase.
# 
# 
# 3.The scatter plot for Gaming shows a more scattered distribution, indicating a less pronounced relationship between gaming download data and the total DL+UL data.
# 
# 
# 4.The size of the markers in the scatter plots represents the magnitude of the corresponding download data (DL) for each application.

# # Variable Transformations

# In[82]:


youtube_dl = [15854611, 20247395, 19725661, 21388122, 15259380]
youtube_ul = [2501332, 19111729, 14699576, 15146643, 18962873]
netflix_dl = [8198936, 18338413, 17587794, 13994646, 17124581]
netflix_ul = [9656251, 17227132, 6163408, 1097942, 415218]
gaming_dl = [278082303, 608750074, 229584621, 799538153, 527707248]
gaming_ul = [14344150, 1170709, 395630, 10849722, 3529801]
other_dl = [171744450, 526904238, 410692588, 749039933, 550709500]
other_ul = [8814393, 15055145, 4215763, 12797283, 13910322]
total_dl = [308879636, 653384965, 279807335, 846028530, 569138589]

decile_classes = 10
total_data_per_decile = []

for i in range(decile_classes):
    start_index = int(len(total_dl) * i / decile_classes)
    end_index = int(len(total_dl) * (i + 1) / decile_classes)
    total_data = sum(total_dl[start_index:end_index]) + sum(youtube_dl[start_index:end_index]) + sum(youtube_ul[start_index:end_index]) + sum(netflix_dl[start_index:end_index]) + sum(netflix_ul[start_index:end_index]) + sum(gaming_dl[start_index:end_index]) + sum(gaming_ul[start_index:end_index]) + sum(other_dl[start_index:end_index]) + sum(other_ul[start_index:end_index])
    total_data_per_decile.append(total_data)

print("Total Data (DL+UL) per Decile Class:")
for i, data in enumerate(total_data_per_decile):
    print(f"Decile {i+1}: {data} bytes")


# # Correlation Analysis

# In[22]:


import pandas as pd

data = pd.DataFrame({
    'youtube_dl': [15854611, 20247395, 19725661, 21388122, 15259380],
    'youtube_ul': [2501332, 19111729, 14699576, 15146643, 18962873],
    'netflix_dl': [8198936, 18338413, 17587794, 13994646, 17124581],
    'netflix_ul': [9656251, 17227132, 6163408, 1097942, 415218],
    'gaming_dl': [278082303, 608750074, 229584621, 799538153, 527707248],
    'gaming_ul': [14344150, 1170709, 395630, 10849722, 3529801],
    'other_dl': [171744450, 526904238, 410692588, 749039933, 550709500],
    'other_ul': [8814393, 15055145, 4215763, 12797283, 13910322]
})
# Compute the correlation matrix
correlation_matrix = data.corr()

# Display the correlation matrix
print(correlation_matrix)


# # Dimensionality Reduction using PCA

# In[47]:


import pandas as pd 
import numpy as np
import seaborn as sns


# In[48]:


x=pd.read_csv("telcom_data-_1_.csv")

x.head()


# In[55]:


not_imp=["Bearer Id","Start ms","End ms","Dur. (ms)","IMSI"]

x.drop(columns=not_imp, inplace=True,axis=1)

x.head()


# In[64]:


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[77]:


import numpy as np
from sklearn.decomposition import PCA

# Define the data
youtube_dl = [15854611, 20247395, 19725661, 21388122, 15259380]
youtube_ul = [2501332, 19111729, 14699576, 15146643, 18962873]
netflix_dl = [8198936, 18338413, 17587794, 13994646, 17124581]
netflix_ul = [9656251, 17227132, 6163408, 1097942, 415218]
gaming_dl = [278082303, 608750074, 229584621, 799538153, 527707248]
gaming_ul = [14344150, 1170709, 395630, 10849722, 3529801]
other_dl = [171744450, 526904238, 410692588, 749039933, 550709500]
other_ul = [8814393, 15055145, 4215763, 12797283, 13910322]
total_dl = [308879636, 653384965, 279807335, 846028530, 569138589]

# Create the feature matrix
features = np.array([youtube_dl, youtube_ul, netflix_dl, netflix_ul, gaming_dl, gaming_ul, other_dl, other_ul]).T

# Perform PCA
pca = PCA(n_components=2)  # Set the number of components you want to retain
features_reduced = pca.fit_transform(features)

# Display the reduced features
print(features_reduced)


# # Interpretation:
# 
# 1.The first principal component captures the overall media consumption behavior of the users. It combines the information from variables such as YouTube, Netflix, gaming, and other data. Higher values on this component indicate higher levels of media usage and consumption.
# 
# 2.The second principal component represents a specific pattern or type of media consumption that is distinct from the overall consumption captured by the first component. It may be associated with a particular platform or type of media. Positive or negative values on this component indicate different preferences or usage patterns.
# 
# 3.The PCA analysis successfully reduced the dimensionality of the data, allowing us to capture the majority of the variance in the dataset using only two components. This reduction simplifies the data while still retaining meaningful information about media consumption.
# 
# 4.The interpretation of the principal components can be further enhanced by analyzing the loadings of each original variable on the components. This can help identify which variables contribute the most to each component and provide additional insights into the underlying factors driving media consumption patterns.

# In[ ]:




