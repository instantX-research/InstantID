products_data = [
    {"product_name": "Product A", "cost_price": 5.5, "sell_price": 8.5, "inventory": 100},
    {"product_name": "Product B", "cost_price": 3.0, "sell_price": 6.0, "inventory": 150},
    {"product_name": "Product C", "cost_price": 2.5, "sell_price": 4.0, "inventory": 200},
]

profit = 0
i = 0
while i<len(products_data):
  cost_price = products_data[i]["cost_price"]
  sell_price = products_data[i]["sell_price"]
  inventory = products_data[i]["inventory"]
  each_product_profit = (sell_price - cost_price) * inventory
  profit += each_product_profit
  i+=1
total_profit = round(profit)
print("Total profit of the company is : ${}".format(total_profit))
