import pandas as pd
import numpy as np
from random import sample
from statistics import mean
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.metrics import silhouette_score

data  = pd.read_table('C:/Users/Owner/Desktop/iems_308/datasets/Medicare_Provider_Util_Payment_PUF_CY2017.txt')
data.drop(['nppes_provider_first_name','nppes_provider_last_org_name','nppes_provider_mi','nppes_credentials','nppes_provider_gender','nppes_provider_street1','nppes_provider_street2','provider_type','hcpcs_description','hcpcs_drug_indicator','line_srvc_cnt','bene_unique_cnt','nppes_entity_code'],axis = 1,inplace = True)
data = data[data['nppes_provider_country'] == 'US']
data = data[data['nppes_provider_state'] != 'XX']
data = data[data['nppes_provider_state'] != 'AA']
data = data[data['nppes_provider_state'] != 'AE']
data = data[data['nppes_provider_state'] != 'AP']
data = data[data['nppes_provider_state'] != 'ZZ']
data = data[data['nppes_provider_state'] != 'AS']
data = data[data['nppes_provider_state'] != 'GU']
data = data[data['nppes_provider_state'] != 'MP']
data = data[data['nppes_provider_state'] != 'VI']
data.reset_index(inplace = True)
data.drop(['nppes_provider_country','index'],inplace = True, axis = 1)

states = data['nppes_provider_state'].unique()

true_d = {'npi' : np.zeros(len(states)*500),'nppes_provider_city': np.zeros(len(states)*500),'nppes_provider_zip' : np.zeros(len(states)*500),'nppes_provider_state' : np.zeros(len(states)*500), 'avg_medicare_pay_ratio': np.zeros(len(states)*500), 'payed_to_requested_ratio' : np.zeros(len(states)*500)}
true_d = pd.DataFrame(data = true_d)

counter = 0

for i in states:
    state = data[data['nppes_provider_state'] == i]
    npis = state['npi'].unique()
    npis = sample(list(npis),500)
    for j in npis:
        place = state[state['npi'] == j]
        true_d.iloc[counter,0] = j
        true_d.iloc[counter,1] = place.iloc[0,1]
        true_d.iloc[counter,2] = place.iloc[0,2]
        true_d.iloc[counter,3] = place.iloc[0,3]
        pay_ratios = np.zeros(place.shape[0])
        req_ratios = np.zeros(place.shape[0])
        for z in range(place.shape[0]):
          pay_ratios[z] = place.iloc[z,10] / place.iloc[z,8]
          req_ratios[z] = place.iloc[z,8] / place.iloc[z,9]
        true_d.iloc[counter,4] = mean(pay_ratios)
        true_d.iloc[counter,5] = mean(req_ratios)
        counter = counter + 1
        
fake_d = true_d.copy()        
scaler1 = preprocessing.Normalizer().fit(fake_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])
fake_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']] = scaler1.transform(fake_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])

#maxclusters = 30
#sse = []
#silh = []
#for nClusters in range(15,maxclusters):
#    kmeans_1 = cluster.KMeans(n_clusters = nClusters, random_state = 0).fit(true_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])
#    silhouette_avg = silhouette_score(true_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']], kmeans.labels_)
#    sse.append(kmeans.inertia_)
#    silh.append(silhouette_avg)
    
#plt.plot(range(15,maxclusters),sse)
#plt.plot(range(15,maxclusters),silh)

kmeans_1 = cluster.KMeans(n_clusters = 25, random_state = 0).fit(fake_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])

fake_d = pd.concat([fake_d,pd.DataFrame(kmeans_1.labels_,columns=["Cluster"])],axis=1)

h = []

maxclusters = 12

for j in range(25):
    sse = []
    test_d = fake_d[fake_d['Cluster'] != j]
    for nClusters in range(2,maxclusters):
        kmeans = cluster.KMeans(n_clusters = nClusters, random_state = 0, max_iter = 10).fit(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])
        silhouette_avg = silhouette_score(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']], kmeans.labels_)
        sse.append(silhouette_avg)
    h.append(max(sse))

h = pd.DataFrame(h,columns = ['score'])
h = h[h['score'] < .6]
k = np.asarray(h['score'].index)
fake_d = fake_d[fake_d['Cluster'].isin(k)]
fake_d.drop(['Cluster'],inplace = True,axis = 1)

test_d = {'npi' : np.zeros(len(states)*350),'nppes_provider_city': np.zeros(len(states)*350),'nppes_provider_zip' : np.zeros(len(states)*350),'nppes_provider_state' : np.zeros(len(states)*350), 'avg_medicare_pay_ratio': np.zeros(len(states)*350), 'payed_to_requested_ratio' : np.zeros(len(states)*350)}
test_d = pd.DataFrame(data = test_d)

counter = 0

for i in states:
    state = true_d[true_d['nppes_provider_state'] == i]
    states = fake_d[fake_d['nppes_provider_state'] == i]
    npis = states['npi'].unique()
    npis = sample(list(npis),350)
    for j in npis:
        place = state[state['npi'] == j]
        test_d.iloc[counter,0] = j
        test_d.iloc[counter,1] = place.iloc[0,1]
        test_d.iloc[counter,2] = place.iloc[0,2]
        test_d.iloc[counter,3] = place.iloc[0,3]
        test_d.iloc[counter,4] = place.iloc[0,4]
        test_d.iloc[counter,5] = place.iloc[0,5]
        counter = counter + 1


scaler2 = preprocessing.Normalizer().fit(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])
test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']] = scaler2.transform(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])



#maxclusters = 13
#sse = []
#silh = []
#for nClusters in range(2,maxclusters):
#    kmeans = cluster.KMeans(n_clusters = nClusters, random_state = 0).fit(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])
#    silhouette_avg = silhouette_score(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']], kmeans.labels_)
#    sse.append(kmeans.inertia_)
#    silh.append(silhouette_avg)
    
#plt.plot(range(2,maxclusters),sse)
#plt.plot(range(2,maxclusters),silh)

kmeans = cluster.KMeans(n_clusters = 4, random_state = 0).fit(test_d[['avg_medicare_pay_ratio','payed_to_requested_ratio']])

test_d = pd.concat([test_d,pd.DataFrame(kmeans.labels_,columns = ['Cluster'])],axis = 1)

clus_0 = test_d[test_d['Cluster'] == 0]
clus_1 = test_d[test_d['Cluster'] == 1]
clus_2 = test_d[test_d['Cluster'] == 2]
clus_3 = test_d[test_d['Cluster'] == 3]