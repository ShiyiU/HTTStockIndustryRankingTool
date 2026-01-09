import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

production_growth_rate_df = pd.read_csv("production_growth_rate.csv")
price_change_df = pd.read_csv("quarterly_price_change.csv")

print(production_growth_rate_df)
print(price_change_df)


production_growth_rate_df = production_growth_rate_df.rename(columns = {"company_name": "Ticker"})

merged_df = pd.merge(production_growth_rate_df, price_change_df, on =["Ticker", "year", "quarter"])

print(merged_df)

correlation = merged_df["production_growth_rate"].corr(merged_df["price_change_%"])
print(f"Correlation coefficient: {correlation}")

plt.figure(figsize = (10, 6))
sns.scatterplot(data = merged_df, x = "production_growth_rate", y = "price_change_%")
plt.title("Correlation between Production Growth Rate and Price Change")
plt.xlabel("Production Growth Rate")
plt.ylabel("Price Change")
plt.savefig(f"correlation_{merged_df['Ticker'].iloc[0]}.png")
plt.show()


