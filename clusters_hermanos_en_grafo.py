#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 16:25:47 2017

@author: raulduarte
"""
from __future__ import print_function
from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as r
class Student():
    def __init__(self, id):
        self.id = id
        self.i = r.random()
        self.a = self.i 
        self.alpha = 0.8
    
    def __str__(self):
        return(str(self.id))
        
    def step(self):
#loop through the neighbors and aggregate their preferences 
        neighbors= g[self]
        w=1/float((len(neighbors)+1))
        s=w*self.a
        for node in neighbors:
             s+=w*node.a
# update my beliefs = initial belief plus sum of all influences 
        self.a=(1-self.alpha)*self.i + self.alpha*s
##### funcion para sumar una lista de matrices ###############################
def tuplasfromlista(lista):
# puedo sacar tuplas de elementos de una lista     
    tuplas = []
    for i in range(len(lista)):
        for j in range(len(lista)):
            if (j > i):
                s = (lista[i],lista[j])
                tuplas.append(s)
            else:
                pass
    return tuplas
def sumalistamatrices(n, lista_m):
   s = (n,n)
   suma = np.zeros(s)
   for i in range(len(lista_m)):
      suma += lista_m[i]
   return suma
##### funcion para sumar una lista de matrices ###############################
#### función para darle colores a los enlaces de acuerdo a su peso##########
def darcoloressegunpesoenlace(G):
  for i,j in G.edges():
    if (G[i][j]["weight"] == 1):#amigos
        G[i][j]["color"] = "blue" 
    elif (G[i][j]["weight"] == 2):#enemigos
        G[i][j]["color"] = "red" 
    elif (G[i][j]["weight"] == 3):#primos
       G[i][j]["color"] = "cyan"  
    elif (G[i][j]["weight"] == 4):#ojo hermanos inicial pero no lo uso!!(resta)
        G[i][j]["color"] = "brown"  
    elif (G[i][j]["weight"] == 7):#nodos padres
        G[i][j]["color"] = "yellow" 
#    elif (G[i][j]["weight"] == 14): #hermanos incorrecto debe ser peso=4
#        G[i][j]["color"] = "green"
    elif (G[i][j]["weight"] == 27):#primos
        G[i][j]["color"] = "black"
    elif (G[i][j]["weight"] == 40):#enemigos
        G[i][j]["color"] = "grey"
#### función para darle colores a los enlaces de acuerdo a su peso##########
#########función para sacar lista de nodos con enlaces pesados#########
def nodosconunciertopeso(G,w):
  nodosw = []
  for i,j in G.edges():
    if (G[i][j]["weight"] == w):
        nodosw.append(i)
        nodosw.append(j)
  listnodosw = list(set(nodosw))
  return listnodosw
#########función para sacar lista de nodos con enlaces pesados#########
####funcion para retornar una lista con los sets de #########
def componentes_grafo(G):
#returna una lista con los sets de nodos de cada componente en el grafo
    componentes = []
    for i in G:
       compo = nx.node_connected_component(G,i)
       if compo in componentes:
         pass
       else:
          componentes.append(compo)
    lista = []
    for i in range(len(componentes)):
        lista.append(list(componentes[i]))
    return lista 
####funcion para retornar una lista con los sets de nodos de cada componete grafo#########
###funcion para descomponer la adyacencia final en componentes###############
#cambia matriz inicial y devuleve la nueva, suce =np.array()
# ejemplo = np.array([2,3,1])#arreglo numpy para aplicar a.all() 
# si se quiere un sólo elemento np.array([4])
def transforma(m,suce):
    m2 = np.matrix(m)
    for i in range(len(m2)):
       for j in range(len(m2)): 
           if (m2[i,j] != suce).all() :
               m2[i,j] = 0
    return m2
###funcion para descomponer la adyacencia final en componentes###############

######funcion para imprimir los enlaces de un grafo y sus atributos##########
def ipesos_enlaces(grafo):
   for n1,n2,attr in grafo.edges(data=True):
    print(n1,n2,attr) 
######funcion para imprimir los enlaces de un grafo y sus atributos##########
######función para mostrar un grafo en tiempo real##########################
def show_graph(G):
   pos = nx.spring_layout(G)
   edges = G.edges()
   colors = [G[u][v]["color"] for u,v in edges]
   nx.draw(G, pos, edge_color=colors, with_labels=True,node_size=400)
   plt.show()
######función para mostrar un grafo en tiempo real##########################

##############Archivo de texto a una matriz numpy####
#Mf = np.loadtxt("matEsc1.txt")
Mf = np.loadtxt("dos_prueba.txt")
Mnp = np.matrix(Mf)
print("matriz dada inicialmente como archivo de texto\n", Mnp)
##############Archivo de texto a una matriz numpy####
###########sustitucion de 5 y 6 por cero#############
for i in range(len(Mnp)):
    for j in range(len(Mnp)):
        if (Mnp[i,j] == 5) or (Mnp[i,j] == 6):
            Mnp[i,j] = 0
print("la matriz sin los 5 y 6 es\n", Mnp)
#print(Mnp.transpose())
###########sustitucion de 5 y 6 por cero#############
##########Revisión de si la matriz es Simetrica######
if (Mnp == Mnp.transpose()).all() :
   print("la matriz es simetrica")
else:
    print("la matriz no es simetrica")
##########Revisión de si la matriz es Simetrica######
##########MATRIZ y GRAFO DE AMISTAD, hacer 3 y 4= 1 #########
#ojo esta matriz es para trabajar aparte del G = Mnp
mfr = np.matrix(Mnp)#ambos comandos funcionan np.matrix()
for i in range(len(mfr)):
    for j in range(len(mfr)):
        if (mfr[i,j] == 3) or (mfr[i,j] == 4):
            mfr[i,j] = 1
        elif (mfr[i,j] == 2):
            mfr[i,j] = 0      
print("matriz de amistad\n", mfr)
Gfr = nx.from_numpy_matrix(mfr)
##########MATRIZ y GRAFO DE AMISTAD, hacer 3 y 4= 1 #########
####convertir matriz a grafo y numero de nodos iniciales####
G = nx.from_numpy_matrix(Mnp)
first_g = len(G) #numero de nodos iniciales
#first_edges = G.edges()#numero de enlaces iniciales 
####conteo de los 4 por cada nodo en la matriz inicial##########
bro_list = []
for i in range(first_g):
    bro = 0
    for j in range(first_g):
        if (Mnp[i,j] == 4):
            bro += 1
    bro_list.append(bro)
print("lista de 4 en matriz inicial", bro_list)
####conteo de los 4 por cada nodo en la matriz inicial##########
######lectura de numero de hermanos de archivo####
#archivo = open("1_anio_temozon", "r")
archivo = open("dos_agrega.txt", "r")
lineas = []
hermanos = []
for line in archivo.readlines():
    lineas.append(line)
for i in range(len(lineas)):
    vector = lineas[i]
    vector = vector.split()
    hermanos.append(int(vector[0]))
#    print(5*float(vector[0]))
print("lista de hermanos en archivo de texto", hermanos)
######lectura de numero de hermanos de archivo####
#####copia lista hermanos para la media##########
sin_uno = hermanos[:]
sin_uno = [x for x in sin_uno if x != -1]
suma = 0
for i in (sin_uno):
    suma += i
media = suma/len(sin_uno)
media = int(round(media))
#####copia lista hermanos para la media##########
###sustitucion de los -1 por la media###########
bro_media = hermanos[:]
for i in range(len(bro_media)):
    if (bro_media[i] == -1):
        bro_media[i] = media  
print("lista de archivo de texto con la media es ", bro_media)
###sustitucion de los -1 por la media###########
###lista resta hermanos encuesta - num de 4 matriz####
resta_hermanos = []
for i in range(len(hermanos)):
    resta = bro_media[i]-bro_list[i]
    resta_hermanos.append(resta)
print("lista final de hermanos para agregar es ", resta_hermanos)
###lista resta hermanos encuesta - num de 4 matriz####
####### CREAR SETS DE NODOS HERMANOS 
##### adicion de hermanos al grafo original ###############
prime_bro = componentes_grafo(G)
conta = len(G)
for i in range(conta):
    for j in range(resta_hermanos[i]):#agregar j veces un hermano
        G.add_edge(i, conta)
        G[i][conta] ["weight"] = 4
        conta += 1
##### adicion de hermanos al grafo original ###############
##### descomposicion del grafo en solo nodos con weight= 4#####################
nuevam = nx.adjacency_matrix(G)
numero = np.array([4])
mcuatros = transforma(nuevam.todense(),numero)
grafo = nx.from_numpy_matrix(mcuatros)
##### descomposicion del grafo en solo nodos con weight= 4#####################
#####separar el grafo en sus componente w=4 #####################################
componentesw4 =componentes_grafo(grafo)
print("la funcion de componentes devuelve \n", componentesw4)
########sacar enlaces posibles de cada set de hermanos ######################
listatuplas = []
for i in range(len(componentesw4)):
    enlaces = tuplasfromlista(componentesw4[i])
    for elem in enlaces:
        listatuplas.append(elem)
print("la lista de todos los enlaces entre hermanos es\n", listatuplas)
print("enlaces entre hermanos son n(n-1)/2 : \n", len(listatuplas))
########sacar enlaces posibles de cada set de hermanos ######################
#######revisar y agregar que tuplas posibles de hermanos no estan en grafo ############

for u,v in listatuplas:
    grafo.add_edge(u,v)
    grafo[u][v]["weight"] = 4
    G.add_edge(u,v)
    G[u][v]["weight"] = 4     
#######revisar y agregar que tuplas posibles de hermanos no estan en grafo ############
#####separar el grafo en sus componente w=4 #####################################
nodos4 = nodosconunciertopeso(grafo, 4)
print(nodos4)
########ALGORITMO PARA AGREGAR PADRES A NO HERMANOS###############################
num_plus = 2
n = len(G)
for i in range(n):
    if i in nodos4:
        pass
    else:
        for j in range(num_plus):
            G.add_edge(i,n)
            G[i][n]["weight"] = 7
            n += 1      
########ALGORITMO PARA AGREGAR PADRES A NO HERMANOS###############################
##### algoritmo para padres de los hermanos######################################
n = len(G)
num_plus = 2
for i in range(len(componentesw4)):
   for j in range(len(componentesw4[i])):
          G.add_edge(componentesw4[i][j],n)
          G[componentesw4[i][j]][n]["weight"] = 7
          G.add_edge(componentesw4[i][j], n+1)
          G[componentesw4[i][j]][n+1] ["weight"] = 7
   n += 2        
##### algoritmo para padres de los hermanos######################################
########Algoritmo para separar por familias###################################

########Algoritmo para separar por familias###################################

####poner colores a enlaces #########

#######ALGORITMO PARA DEJAR FAMILIAS EN CLUSTERS despues de añadir hermanos ##########
###convertir grafo final en una matriz de adyacencia con function transforma###
adj_matrix = nx.adjacency_matrix(G)
adj_matrix = adj_matrix.todense()
G_final = nx.from_numpy_matrix(adj_matrix)#GRAFO FINAL
darcoloressegunpesoenlace(G_final) 
print("matriz adyacencia final\n", adj_matrix)#convertir grafo final a matriz
###convertir grafo final en una matriz de adyacencia con function transforma###
######cambiar matriz adyacencia final por una de familias######################
lista = np.array([4, 7, 27, 40])#tenia que ser un arreglo de numpy para aplicar a.all()    
print("original\n", adj_matrix)#ojo en la lista anterior puedo agregar 27 y 40
grafo_familias = transforma(adj_matrix, lista)
print("nueva\n", grafo_familias)
######cambiar matriz adyacencia final por una de familias######################
######dibujar nuevo grafo con las familias separadas###########################
G_familias = nx.from_numpy_matrix(grafo_familias)
darcoloressegunpesoenlace(G_familias)
#######poner colores a enlaces nuevo grafo #########
plt.figure(4)
show_graph(G_familias)# función para graficar el grafo on the fly 
########ALGORITMO PARA DEJAR FAMILIAS EN CLUSTERS despues de añadir hermanos ##########
##### UNA VEZ QUE TENGA EL GRAFO FINAL #
#### CREO QUE PUEDO PROCEDER A APLICAR LA DESCOMPOSICION 
##### CON MI FUNCION TRANSFORMA Y LUEGO CON MI FUNCION 
###### SUMA PUEDO CREAR EL GRAFO COMPUESTO FINAL ######
total_adj = nx.adj_matrix(G)
total_adj = total_adj.todense()
plt.figure(1)
show_graph(G_final)

##########MATRIZ y GRAFO DE AMISTAD, hacer 3 y 4= 1 #########
#ojo esta matriz es para trabajar aparte del G = Mnp
mfr = np.matrix(total_adj)#ambos comandos funcionan np.matrix()
for i in range(len(mfr)):
    for j in range(len(mfr)):
        if (mfr[i,j] == 3) or (mfr[i,j] == 4):
            mfr[i,j] = 1
        elif (mfr[i,j] == 2):
            mfr[i,j] = 0      
#print("matriz de amistad\n", mfr)
Gfr = nx.from_numpy_matrix(mfr)
##########MATRIZ y GRAFO DE AMISTAD, hacer 3 y 4= 1 #########
lista_num = np.array([2,3,4])
lista_mat = []
lista_mat.append(Gfr)
for i in range(len(lista_num)):
    matri = transforma(total_adj,lista_num[i])
    lista_mat.append(matri)
print(lista_mat)
matri_sum_weight = sumalistamatrices(len(total_adj),lista_mat)
print(matri_sum_weight)

###Aqui intentare crear una copia de este grafo con objetos PERSONAS ##########
g = nx.Graph()
for i in range(len(G_final)):
    p = Person(i)
    g.add_node(p)

lista = []
for x in g.nodes():
    lista.append(x)

for u,v in G_final.edges():
    x = [c for c in g.nodes() if c.id == u] 
    y = [z for z in g.nodes() if z.id == v]
    g.add_edge(x[0],y[0])

        
## draw the resulting graph and color the nodes by their value 
#print(g[1])

col=[objeto.a for objeto in g.nodes()]
pos=nx.spring_layout(g)
plt.figure(2)
nx.draw_networkx(g, pos, node_color = col)

plt.figure(3)
for i in range(30):
    for node in g.nodes():
        node.step()
    col = [v.a for v in g.nodes()]
    print(col)    
    plt.plot(col)   
    
ipesos_enlaces(G)
        
