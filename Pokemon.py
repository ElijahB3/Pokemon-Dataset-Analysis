# Using Python
# Option 1: pydytuesday python library
## pip install pydytuesday
import pandas as pd # type: ignore
import numpy as np
import matplotlib as plt
import plotly_express as px
import scipy.stats
import plotly.io as pio
from scipy.optimize import curve_fit

pio.renderers.default = 'browser'

# Download files from the week, which you can then read in locally

# Option 2: Read directly from GitHub and assign to an object

pokemon_df = pd.read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2025/2025-04-01/pokemon_df.csv')

#Questions I wanna answer

#1 Does your attack correlate with your special attack?
#2 Does your weight cordinate with your defense? Some pokeon are tanks but their weight may help with that
#3 Speed vs Weight? The bigger they are the slower they should be right? I hope so
#4 Common duo typing?
#5 Talk about special pokemon (Outliers, sudo legendarys, legendarys)

#Lets just plot everything to start and see wuz poppin
for x_axis, y_axis in [
    ("height", "weight"),
    ("attack", "defense"),
    ("special_attack", "special_defense"),
]:

    px.scatter(pokemon_df, x = x_axis, y=y_axis,)

#We've found some correlation now I want a line of regression for each plot

#Answering 0 
copy_pkmn = pokemon_df.copy()
x = copy_pkmn['height']
y = copy_pkmn['weight']

m, b = np.polyfit(x,y,1)
copy_pkmn['height_weight_pred'] = m*x + b

fig = px.scatter(copy_pkmn, x="height", y="weight",title="Height vs Weight")
fig.add_traces(px.line(copy_pkmn, x="height", y="height_weight_pred").data)

#fig.show()

ans_height_weight = """The regression analysis shows a weak linear relationship between Pokémon height and weight.
Most observations are widely scattered around the regression line, with several extreme outliers.
This suggests that height alone is not a strong predictor of weight across Pokémon species."""

#print(copy_pkmn["weight_prec"].describe())

# Most of the values are far from the standard deviation with a std of 78, most of the data lies in in the 85% percentle of the graph
#Id say this does not give us insight to ask whether or not the weight and height have any relation 


#1 Does your attack correlate with your special attack?
x_atk = copy_pkmn['attack']
y_special_atk = copy_pkmn['special_attack']

m, b = np.polyfit(x_atk, y_special_atk, 1)
copy_pkmn['spec_pred'] = m * x_atk + b

fig2 = px.scatter(copy_pkmn, x = "attack", y = "special_attack", title = "#0 Attack vs Special Attack")
fig2.add_traces(px.line(copy_pkmn,x ="attack", y = "spec_pred").data)

#fig2.show()

#print(copy_pkmn['special_attack'].describe())

atk_special_ansr = """ The scatter plot and fitted regression line indicate a weak linear relationship between physical
attack and special attack stats. Most data points deviate substantially from the regression line,
suggesting that these attributes are largely independent and influenced by different design roles."""


#2 Does your weight correlate with your defense? Some pokemon are tanks but their weight may help with that
x_weight = copy_pkmn['weight']
y_defense = copy_pkmn['defense']

m,b = np.polyfit(x_weight, y_defense, 1)
copy_pkmn['def_pred'] = m * x_weight + b

fig3 = px.scatter(copy_pkmn, x = "weight", y = "defense", title = "Weight to Defense")
fig3.add_traces(px.line(copy_pkmn, x = "weight", y = "def_pred").data)

#fig3.show()

#print(copy_pkmn["def_pred"].describe())

#Defense correlates to your weight? Could be a better figure
x_defense = copy_pkmn['defense']
y_weight = copy_pkmn['weight']

m,b = np.polyfit(x_defense, y_weight, 1)
copy_pkmn['weight_pred'] = m * x_defense + b

fig4 = px.scatter(copy_pkmn, x = "defense", y = "weight", title = "#1 Defense to Weight")
fig4.add_traces(px.line(copy_pkmn, x = "defense", y = "weight_pred").data)

#fig4.show()

#print(copy_pkmn["weight_pred"].describe())

ans_weight_pred = """ The regression suggests a moderate positive relationship between Pokémon weight and defense.
Heavier Pokémon tend to exhibit higher defense values, although notable variability remains.
This pattern may reflect design choices where bulkier Pokémon are more defensively oriented. """

heavy_pkm = copy_pkmn[copy_pkmn["weight"] > 100].copy()
(heavy_pkm.head())

#Weve gotten our heavier pokemon now well check their typings into a barplot...
px.bar(heavy_pkm['type_1'])
#The top 3 Heaviest pokemons are type Rock, Water and Dragon, 
type_counts = heavy_pkm['type_1'].value_counts().reset_index()
type_counts.columns = ['type_1', 'count']

fig5 = px.pie(type_counts, names='type_1', values='count', title='Heavy Pokémon Type Distribution')

#fig5.show()

#Lets create a Bar chart to categorize the top 3 heaviest pokemon, and see the total counts
top3 = ['rock','water','dragon']
heaviest_types = heavy_pkm[heavy_pkm['type_1'].isin(top3)]

heavy_counts = heaviest_types['type_1'].value_counts().reset_index()
heavy_counts.columns = ['type_1', 'counts']

fig6 = px.bar(heavy_counts,x = 'type_1', y = 'counts', title = "Top 3 Heaviest Typing Counts")
#fig6.show()
#We've found a tie between Rock type pokemon and Water Type pokemon for who has heavier pokemon,
#We should check for Dual Typing to reduce our counts, we'll filter out those that have type 2

pure_typing = heaviest_types[heaviest_types['type_2'].isnull() | (heaviest_types['type_2'] == '')].copy()

pure_typing_counts = pure_typing['type_1'].value_counts().reset_index()
pure_typing_counts.columns = ['type_1', 'counts']

fig7=px.bar(pure_typing_counts, x = 'type_1', y= 'counts', title = "Top 3 Heaviest Pure Typing Pokemon")
#fig7.show()

#There are 7 pure water types where Rock and Dragon have 4, We can then conclude that the Water-type pokemon are the heaviest set.
#So well name them...
pure_water = pure_typing[pure_typing['type_1'] == 'water']

ans_heaviest = 'The Heaviest pokemon are in the Water type category\n' 
print((pure_water.sort_values(by="weight",ascending=False).head(7)))

ans_heaviest = """ Among Pokémon with weight greater than 100, Water-type Pokémon appear most frequently when
restricting the analysis to pure typings. This suggests that Water-type Pokémon are commonly
represented among heavier designs, although this result depends on the chosen weight threshold."""

#3 Speed vs Weight? The bigger they are the slower they should be right? I hope so...
# We can shorten our time using the heaviest pokemon set 
x_speed_pkmn = copy_pkmn['speed']
y_weight_pkmn = copy_pkmn['weight']

# Fit regression: weight = m * speed + b
m, b = np.polyfit(x_speed_pkmn, y_weight_pkmn, 1)
copy_pkmn['weight_pred_by_speed'] = m * x_speed_pkmn + b

fig8 = px.scatter(copy_pkmn, x="speed", y="weight", title="#2 Speed vs Weight")
fig8.add_traces(px.line(copy_pkmn, x="speed", y="weight_pred_by_speed").data)
#fig8.show()

ans_speed = """ The linear regression between speed and weight performs poorly, with many observations lying
far from the fitted line. This indicates that a simple linear model does not adequately capture
the relationship, and that speed and weight may not be strongly related. """

#Lets try making a polynomial regression model relationship...
coeffs = np.polyfit(x_speed_pkmn, y_weight_pkmn,2)
copy_pkmn['weight_pred_poly'] = coeffs[0]*x_speed_pkmn**2 + coeffs[1]*x_speed_pkmn + coeffs[2]

fig9 = px.scatter(copy_pkmn, x="speed",y="weight",title ="#3 Speed to Weight prediction (Quadratic Degree 2)")
fig9.add_traces(px.line(copy_pkmn,x = "speed", y = "weight_pred_poly").data)
#fig9.show()
ans_speed_poly = """ Applying a quadratic regression does not significantly improve model fit, as the predicted curve
still fails to capture the wide dispersion of the data. This suggests there is no clear functional
relationship between Pokémon speed and weight in this dataset. """

#4 Common duo typing?
#Well dive and see our most common duo typing, I believe pie and bar charts are more useful here

#First, I want to see the ratio of typing across the pokemon and target one type that has majority of pokemon in it
#Then well try to predict what the most common dual typing is found from that type

#Combine both types into a single series

common_type = pd.concat([copy_pkmn['type_1'],copy_pkmn['type_2']]).dropna()

common_type_count = common_type.value_counts().reset_index()
common_type_count.columns = ['type', 'counts']

fig10 = px.pie(common_type_count,names = "type", values = "counts", title = "Pokemon Types Distribution")
#fig10.show()

#Once again, water is the most common typing seen in pokemon, so well try to extract what usually goes with water

water_duo_type = copy_pkmn[(copy_pkmn["type_1"] == 'water') & (copy_pkmn['type_2'].notnull()) & (copy_pkmn['type_2'] != ' ')].copy()
#Made a copy of water pokemon and other typing set, lets check its good
#print(water_duo_type.head())
#OK, now well try to get the average typing through a bar chart and visualize it
second_typing_count = water_duo_type['type_2'].value_counts().reset_index()
second_typing_count.columns = ['type_2' ,'counts']

fig11 = px.bar(second_typing_count, x = "type_2",y = "counts", title= "#4 Water-Second Typing distribution")
#fig11.show()

ans_duo = """ Water is the most common Pokémon type in the dataset. Among dual-typed Water Pokémon, Ground
emerges as the most frequent secondary type. This suggests a common design pairing rather than
a performance-based relationship. """

#5 Talk about special pokemon (Outliers, sudo legendarys, legendarys)
#Lets try and pick some outlier pokemon which are exceptions amongst their typing.
#Lets pick Fairy, Steel and Bug Typing, as theyre smaller sets

unique_pkmn = pokemon_df[(pokemon_df['type_1'] == 'bug') | (pokemon_df['type_1'] == 'steel') | (pokemon_df['type_1'] == 'fairy') |
(pokemon_df['type_2'] == 'bug') | (pokemon_df['type_2'] == 'steel') | (pokemon_df['type_2'] == 'fairy')].copy()
unique_pkmn = unique_pkmn.drop(['url_icon','generation_id','color_1','color_2', 'base_experience','url_image' ],axis=1)
#Filtering...
#Well pick pokemon off their special attack, defense and speed

#Shuckle #1 Defense stype
shuckle = unique_pkmn[unique_pkmn['pokemon'] == "shuckle"]
#Mega Gardevoir #1 Special attacker
mega_gardevoir = unique_pkmn[unique_pkmn['pokemon'] == "gardevoir-mega"]
#Ninjask #1 Fastest pokemon
ninjask = unique_pkmn[unique_pkmn['pokemon'] == "ninjask"]

ans_unique = """ Within the selected typings (Bug, Steel, and Fairy), several Pokémon stand out as statistical
outliers. Mega Gardevoir exhibits exceptionally high special attack, Shuckle demonstrates extreme
defensive stats at the cost of speed, and Ninjask is a notable speed outlier with low defenses.
These examples highlight specialized stat distributions."""


#This concludes the analysis of the pokemon data sets in short
#0
print(ans_heaviest)
#1
print(ans_weight_pred)
#2
print(ans_speed)
#3
print(ans_speed_poly)
#4
print(ans_duo)
#5
print(ans_unique)

#July 2, 2025 - First Project Made,  but is just the beginning...

# print(copy_pkmn['height'].values)
# from LinearRegression import LinearRegression
# reg = LinearRegression()
# reg.fit(copy_pkmn['height'].values, copy_pkmn['weight'].values)
# predictions = reg.predict(copy_pkmn['height'].values)

# def mse(y_test, predictions):
#     return np.mean((y_test - predictions) ** 2)

# mse = mse(copy_pkmn['weight'].values,predictions)
# print(mse)

from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare your data: X should be 2D, y should be 1D
X = copy_pkmn['height'].values.reshape(-1, 1)  # shape (n_samples, 1)
y = copy_pkmn['weight'].values                 # shape (n_samples,)

# Create and fit the model
reg = LinearRegression()
reg.fit(X, y)

# Make predictions
predictions = reg.predict(X)

# Calculate mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

fig = px.scatter(copy_pkmn, x="height", y="weight",title="Height vs Weight Model").show()
fig.add_traces(px.line(copy_pkmn, x="height", y=predictions).data)

fig9.add_traces(px.line(copy_pkmn,x = "speed", y = "weight_pred_poly").data)

print("MSE:", mse(y, predictions))