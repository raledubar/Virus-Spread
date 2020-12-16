#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 00:12:04 2020

@author: raulduarte
"""

from __future__ import print_function
from __future__ import division# ya no tengo que porne len(g)/float(s)
import networkx as nx
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random as r
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep

from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

from ndlib.viz.bokeh.DiffusionPrevalence import DiffusionPrevalence

from ndlib.viz.bokeh.MultiPlot import MultiPlot

import time

import heapq 

import os

import operator
from collections import OrderedDict

import json 


start_time = time.time()

def model_IC(g, S, proba_list, steps):
    spread_simulations=[]
    for k in range(steps):
#        r.seed creo que solo en el THmodel para indiv threshol de cada nodo
        model = ep.IndependentCascadesModel(g)
        config = mc.Configuration()
        infected_nodes = S
        config.add_model_initial_configuration("Infected", infected_nodes)
        # Setting the edge parameters
        link=0
        for e in g.edges():
          config.add_edge_configuration("proba", e, proba_list[link] )
          link += 1
        model.set_initial_status(config)
#        iterations = []
        while True:
           iteration = model.iteration()
#           iterations.append(iteration)
           if iteration["iteration"] > 0 and (iteration["status_delta"][0]==0) and (iteration["status_delta"][1]==0) and (iteration["status_delta"][2]==0) : 
                break

#en caso de que se quiera ver la evolución de los infectados, sanos y recuperados            
#        ina = []
#        acti = []
#        remo = []
#        for i in range(iteration["iteration"]):
#          ina.append(iterations[i]["node_count"][0])
#          acti.append(iterations[i]["node_count"][1])
#          remo.append(iterations[i]["node_count"][2])
#en caso de que se quiera ver la evolución de los infectados, sanos y recuperados          
        spread_simulations.append(iteration["node_count"][2]-len(S)) 
        
        
    vexpected = np.mean(spread_simulations)
    #return iterations, ina, acti, remo, maxspread
    #return vexpected, spread_simulations
    return vexpected



files = os.listdir("creations_sch2/")

graph_list = []

#Mf1 = np.loadtxt("creations_sch1/creation_1.txt")
#Mnp1 = np.matrix(Mf1)
#g1 = nx.from_numpy_matrix(Mnp1)
#    
#Mf2 = np.loadtxt("creations_sch1/creation_2.txt")
#Mnp2 = np.matrix(Mf2)
#g2 = nx.from_numpy_matrix(Mnp2)
#    
#listag1 = propiedades_paper_escuelita(g1)
#listag2 = propiedades_paper_escuelita(g2)
#
#print(listag1)
#print(listag2)  
  
for ind in range(1,1001):
    Mf = np.loadtxt("creations_sch2/"+"creation_"+str(ind)+".txt")
    Mnp = np.matrix(Mf)
    g = nx.from_numpy_matrix(Mnp)
    graph_list.append(g)

#listag1 = propiedades_paper_escuelita(graph_list[0])
#listag2 = propiedades_paper_escuelita(graph_list[1])

#print(round(time.time()-start_time))
 
countrepit = 0
norma = float(1)/1000
gavera = nx.Graph()
counter = 0
#Build the graph with synthetic graphs
for elem in graph_list:
     edge_list = list(elem.edges())
     for (u,v) in edge_list:
       if gavera.has_edge(u,v):
         gavera[u][v]["weight"] += norma
         countrepit +=1
    
       else:
         gavera.add_edge(u, v, weight=norma)
     
#     print("file "+str(counter)+" completed")
     counter = counter + 1       
#    
#build the graph with the original graph 
escuela = "esc2.txt"     
origin_Mf = np.loadtxt("esc2.txt")
origin_Mnp = np.matrix(origin_Mf)
origin_g = nx.from_numpy_matrix(origin_Mnp)
count_onelink = 0
linkone_created = 0

origin_list = origin_g.edges()
for (u,v) in origin_list:
    if gavera.has_edge(u,v):
         gavera[u][v]["weight"] = 1
         count_onelink +=1    
    else:
         gavera.add_edge(u, v, weight=1)
         linkone_created +=1
#gavera.edges.data()
#FALTA IMPRIMIR EL GRAFO EN UN ARCHIVO DE TEXTO Y APLICAR IM_ERGM_PRUEBAUNO
final_factor = 0.5        
for (u,v) in gavera.edges():
    gavera[u][v]["weight"] *= final_factor


####AQUI VA EL CODIGO PARA UTILIZAR EL 10, 20 POR CIENTO EN VEZ DE USAR TODOS LOS MISSING LINKS########## 

dicciona = dict( ((u,v), round(gavera[u][v]["weight"], 4) ) for (u,v) in gavera.edges() ) 

for (u,v) in origin_g.edges():
    dicciona.pop((u,v))



proba_ord = sorted(dicciona.items(), key = lambda kv:kv[1], reverse = True  )


# LA PARTE DE ORDENAMIENTO EN EL DICCIONARIO NO ESTA FUNCIONANDO#
for (u,v) in origin_g.edges():
    origin_g[u][v]["weight"] = 0.5

        
nuevos =round( 0.1 * len(origin_g.edges()))
nuevos = int(nuevos)
for k in range(nuevos):
    origin_g.add_edge(proba_ord[k][0][0], proba_ord[k][0][1], weight = proba_ord[k][1] )
    
####AQUI VA EL CODIGO PARA UTILIZAR EL 10, 20 POR CIENTO EN VEZ DE USAR TODOS LOS MISSING LINKS########## 
    

####AQUI VA EL CODIGO PARA USAR TODOS LOS MISSING LINKS #####################################    
ergm_proba = []
for u,v in origin_g.edges():
    ergm_proba.append(origin_g[u][v]["weight"])



### POR EFICIENCIA CREARE UN GRAFO SIN LOS LINKS PESADOS########
g_light = nx.Graph() 
for (u,v) in origin_g.edges():
    g_light.add_edge(u,v)

####AQUI VA EL CODIGO PARA USAR TODOS LOS MISSING LINKS #####################################
 
### POR EFICIENCIA CREARE UN GRAFO SIN LOS LINKS PESADOS######## 
for var_seed in range(3, 8):    
    start_time = time.time()
    celf_seed = var_seed   
    steps = 1000  
    gains = []
    for node in range(len(g_light.nodes())):
        spread = model_IC(g_light, [node], ergm_proba, steps)
        heapq.heappush(gains, (-spread, node))

    spread, node = heapq.heappop(gains)
    solution = [node]
    spread = -spread
    spreads = [spread]
    
    # record 
    lookups = [len(g_light.nodes())]
    elapsed = [round(time.time() - start_time, 3)]
    
    for v in range(celf_seed-1):
        node_lookup = 0
        matched = False
        
        while not matched:
            node_lookup += 1
            
            v, current_node = heapq.heappop(gains)
            spread_gain = model_IC(g_light, solution + [current_node], ergm_proba, steps) - spread
            
            heapq.heappush(gains, (-spread_gain, current_node))
            matched = gains[0][1] == current_node   
        spread_gain, node = heapq.heappop(gains)
        spread -= spread_gain
        solution.append(node)
        spreads.append(spread)
        lookups.append(node_lookup)
        
        elapse = round(time.time() - start_time, 3)
        elapsed.append(elapse)
        
        
    TOTAL_TIME = round(time.time() - start_time, 3)
    
    results = []
    results.append(escuela)
    results.append(len(origin_g.edges()))
    results.append(celf_seed)
    results.append(steps)
    results.append(TOTAL_TIME)
    results.append(solution)
    results.append(spread)
    results.append(elapsed)
    # f = open('influencers_sch1/', 'a')
    with open('influencers_sch2/results_sch2.txt', 'a') as f: 
       json.dump(results, f)
                 
    print(round(time.time() - start_time, 3))


   
