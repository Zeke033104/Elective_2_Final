import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

df = pd.read_csv('laptop_price - dataset.csv')
df.head()

print("Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)

print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

print(df['Price (Euro)'].describe())

df['Price (Euro)'].plot(kind='hist', bins=30)
plt.title('Price Distribution')
plt.xlabel('Price (Euro)')
plt.show()

df['Company'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Laptop Count by Company')
plt.xlabel('Company')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

df['TypeName'].value_counts().plot(kind='bar', color='coral')
plt.title('Laptop Type Distribution')
plt.tight_layout()
plt.show()

df['RAM (GB)'].value_counts().sort_index().plot(kind='bar', color='mediumpurple')
plt.title('RAM Distribution')
plt.xlabel('RAM (GB)')
plt.tight_layout()
plt.show()

df['CPU_Company'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('CPU Company Share')
plt.ylabel('')
plt.show()

df[df['RAM (GB)'].isin([4,8,16,32])].boxplot(column='Price (Euro)', by='RAM (GB)')
plt.title('Price by RAM Size')
plt.suptitle('')
plt.show()


""""
Observations and Interpretation
Based on the exploratory data analysis conducted on the Laptop Price dataset, 
the following observations were noted:

1. The dataset is clean with no missing values or duplicates. 
All 15 columns across 1,275 entries are complete, meaning no imputation 
or row removal is necessary before proceeding to model training.

2. Lenovo and Dell are the most represented brands. 
The bar chart shows that Lenovo has the highest laptop count, 
followed by Dell and HP. This suggests the dataset leans toward 
mainstream consumer and business laptops rather than niche or premium brands.

3. 8GB RAM is the dominant configuration. The RAM distribution chart confirms that 
the majority of laptops ship with 8GB, with 4GB and 16GB as secondary options. 
The box plot further shows that higher RAM sizes correspond to notably higher 
price ranges, indicating RAM is a strong pricing factor.

4. Intel controls over 95% of the CPU market share in this dataset. 
The pie chart reveals that Intel CPUs appear in 95.2% of laptops, 
with AMD at 4.7% and Samsung at 0.1%. This imbalance means CPU company 
alone may not be a useful differentiating feature for the model.

5.Intel controls over 95% of the CPU market share in this dataset. 
The pie chart reveals that Intel CPUs appear in 95.2% of laptops, with 
AMD at 4.7% and Samsung at 0.1%. This imbalance means CPU company alone 
may not be a useful differentiating feature for the model.


"""
   
