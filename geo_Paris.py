import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression

# transactions
transactions_df = pd.read_csv('75.csv')
transactions_df = transactions_df[transactions_df.surface_reelle_bati.notnull()]
transactions_df['valeur_fonciere'] = transactions_df['valeur_fonciere'].astype(float)
transactions_df['surface_reelle_bati'] = transactions_df['surface_reelle_bati'].astype(float)
transactions_df['price_m2'] = transactions_df['valeur_fonciere'] / transactions_df['surface_reelle_bati']
transactions_df = transactions_df[transactions_df.surface_reelle_bati < 200]
transactions_df = transactions_df[(transactions_df.price_m2 > 9000) & (transactions_df.price_m2 < 25000)]
transactions = gpd.GeoDataFrame(transactions_df, geometry=gpd.points_from_xy(transactions_df.longitude, transactions_df.latitude))
transactions.crs = {'init': 'epsg:4326'}

arrond_price_m2 = transactions_df.groupby(['code_postal']).mean()

# arrondissement
arrondissements = gpd.read_file('./arrondissements/arrondissements.shp')
arrondissements['code_postal'] = arrondissements['c_arinsee'] - 100
arrondissements = arrondissements.merge(arrond_price_m2, on='code_postal')

# quartiers
quartiers = gpd.read_file('./quartier_paris/quartier_paris.shp', encoding='utf-8')
print(quartiers.head())
merge_quartier = gpd.sjoin(transactions, quartiers, how="inner", op='within')
print(merge_quartier.head())
groupby_quartier = merge_quartier.groupby('l_qu').mean()
quartiers = quartiers.merge(groupby_quartier, on='l_qu')
print(quartiers.head())


# metro
metro = gpd.read_file('./emplacement-des-gares-idf-data-generalisee/emplacement-des-gares-idf-data-generalisee.shp')
metro = metro[metro['mode']=='METRO']

# plot
fig1, ax1 = plt.subplots()
ax1 = arrondissements.plot(edgecolor='black', zorder=3, column='price_m2', legend=True, cmap='OrRd')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title('Moyenne par Arrondissement des Prix au m² (€/m²) - 2021')
# metro.plot(ax=ax)
# transactions.plot(ax=ax)

def find_plot_index(index):
    r = (index-1) // 4
    c = (index-1) % 4
    return [r, c]

fig2, axs = plt.subplots(5,4, squeeze=False)
for index, row in arrond_price_m2.iterrows():

    plot_index = find_plot_index(int(index-75000))

    temp_df = transactions_df[transactions_df.code_postal == index]
    temp_df.plot(x='surface_reelle_bati', y='valeur_fonciere', c='price_m2', vmin=7000, vmax=30000, kind='scatter', colorbar=False, cmap='OrRd', ax=axs[plot_index[0], plot_index[1]])
    axs[plot_index[0], plot_index[1]].annotate(str(int(index)), fontsize=9, xy=(0,1), xycoords='axes fraction', horizontalalignment='left', verticalalignment='top', xytext=(5,-5), textcoords='offset points')
    axs[plot_index[0], plot_index[1]].set_xlim(0, 100)
    axs[plot_index[0], plot_index[1]].set_ylim(0, 2000000)
    axs[plot_index[0], plot_index[1]].set_xlabel('')
    axs[plot_index[0], plot_index[1]].set_ylabel('')
    if plot_index in [[0,0], [1,0], [2,0], [3,0]]:
        axs[plot_index[0], plot_index[1]].set_xticklabels([])
    elif plot_index in [[4,1], [4,2], [4,3]]:
        axs[plot_index[0], plot_index[1]].set_yticklabels([])
    elif plot_index == [4,0]:
        pass
    else:
        axs[plot_index[0], plot_index[1]].set_xticklabels([])
        axs[plot_index[0], plot_index[1]].set_yticklabels([])
    
    # regression lineaire
    reg = LinearRegression(fit_intercept=False)
    reg.fit(temp_df.loc[:, 'surface_reelle_bati'].values.reshape(-1,1), temp_df.loc[:, 'valeur_fonciere'].values.reshape(-1,1))
    Y_pred = reg.predict(np.linspace(0, 100).reshape(-1,1))
    axs[plot_index[0], plot_index[1]].plot(np.linspace(0, 100), Y_pred, c='black')
    axs[plot_index[0], plot_index[1]].annotate(str(int(reg.coef_))+' €/m2', fontsize=7, xy=(0,0.8), xycoords='axes fraction', ha='left', va='top', xytext=(5,-5), textcoords='offset points')

im = fig2.axes[0].collections[0]
cbar = fig2.colorbar(im, ax=fig2.axes)
cbar.ax.set_ylabel('Prix par m² (€/m²)')
fig2.text(0.5, 0.04,'Surface (m²)', ha='center')
fig2.text(0.04, 0.5,'Prix de Vente (€)', va='center', rotation='vertical')
fig2.suptitle('Transactions Immobilières à Paris en 2021', fontsize=12)



ax3 = quartiers.plot(edgecolor='black', zorder=3, column='price_m2', legend=True, cmap='OrRd')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_title('Moyenne par Quartier des Prix au m² (€/m²) - 2021')

plt.show()

