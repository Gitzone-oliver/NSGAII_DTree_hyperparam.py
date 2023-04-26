# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 16:42:36 2018

@author: OLIVER
"""
# %% LIBRARY
import pickle
import math
import random
#import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
# %% FUNCTIONS

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    # list1  = Lista de valores
    # values = Tambien es una lista
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    #Hace vectores S, n, rank en zeros de tamaño de value
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                   S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to carry out the crossover
def crossover(a,b): # a & b are list
    r=random.random()
    if r > 0.5:
        newList=[x + y for x, y in zip(a, b)]
        newList=[x/2 for x in newList]
        return mutation(newList)
    else:
        newList=[x - y for x, y in zip(a, b)]
        newList=[x/2 for x in newList]
        return mutation(newList)

#Function to carry out the mutation operator
def mutation(solution):
    #print('Antes de la mutacion', solution)
    #names=[rasgo for rasgo in param_space.keys()]
    for element, k in zip(solution, range(len(solution))):       
        # Condiciones del rasgo
        items =param_space[names[k]]
        # Se define valores min y max del rasgo
        min_x=items[0]      # Valor minimo
        max_x=items[1]      # Valor maximo del rasgo
        tipo_rasgo=items[2] # int, float, catagorico
      
        # Mutacion de un elemento de 20% de la poblacion
        mutation_prob = random.random()
        if mutation_prob > 1-.10:
           element = min_x+(max_x-min_x)*random.random()  
         
        # Condiciones de tipo de dato
        if 'int' == tipo_rasgo:
            element = int(np.around(element))
            
        # Condiciones de limites del rasgo                            
        while (element < min_x or element > max_x):  
            #print('Limites min: %2.1f, max: %2.1f' %(min_x, max_x))
            #print(element)              
            # Se crea un valor aleatorio para ese rasgo
            if 'int' == tipo_rasgo:
                element = int(min_x+(max_x-min_x)*random.random()) 
            else: # Crea un flotante
                element = min_x+(max_x-min_x)*random.random() 
        solution[k]=element
    #print('Despues de la mutacion', solution)
    return solution

# %% Evaluación con el regresor
def Fn_eval_DecisionTree(params_sol, X_train, y_train):
    params=params_sol.copy()
    X_train_aux = X_train[:]
#    # Hiperparametros de los datos
#    if 'normalize' in params:
#        if params['normalize'] == 1:
#            X_train_aux = normalize(X_train)
#            X_train_aux = pd.DataFrame(X_train_aux)
#        del params['normalize']
##    # Normalize total_bedrooms column
##    x_array = np.array(df['total_bedrooms'])
##    normalized_X = preprocessing.normalize([x_array]) 
##        
#        
#    if 'scale' in params:
#        if params['scale'] == 1:
#            X_train_aux = scaler.fit_transform(X_train)
#            X_train_aux = pd.DataFrame(X_train_aux)
#        del params['scale']

    # Validacion como hiperparametro
    if 'CV_num' in params:
        CV=params['CV_num']
        del params['CV_num']
    else:
        CV=10

    # Hiparametros del algoritmo
    from sklearn.tree import DecisionTreeRegressor
    regg_model = DecisionTreeRegressor(**params, criterion='mse', random_state=1)
    from sklearn.model_selection import KFold
    #tabla_CV=[]
    #ConcordanciaCV=[] 
    ##%% VALIDACION   
    y_valid=[]
    #VALIDACION CRUZADA VIAS 10
    kf = KFold(n_splits=CV)
    yvalidvec=[]
    y_testvec=[]
    for cvx_train, cvx_test in kf.split(X_train_aux, y_train):
        X_entrenamiento = X_train_aux.iloc[cvx_train]
        X_testeo = X_train_aux.iloc[cvx_test]
        y_entrenamiento = y_train.iloc[cvx_train].values.ravel()
        y_testeo = y_train.iloc[cvx_test].values.ravel()
        y_testvec=list(y_testvec)+list(y_testeo)
        # entrenamiento
        
        regg_model.fit(X_entrenamiento, y_entrenamiento)
        y_valid = np.array(regg_model.predict(X_testeo))
        #print(y_valid)
        yvalidvec=list(yvalidvec)+list(y_valid)
    # Valores de cada modelo
    y_testvec=np.array(y_testvec)
    yvalidvec=np.array(yvalidvec)
    return (yvalidvec, y_testvec)

def function1(params_sol, X_train, y_train):
    yvalidvec, y_testvec = Fn_eval_DecisionTree(params_sol,X_train, y_train)
    MAE    = np.mean(np.abs( yvalidvec - y_testvec))
    #MAEsd  = np.std(np.abs( yvalidvec - y_testvec))
    #MPE,  MPEsd  = mean_percentage_error(y_testvec, yvalidvec, 1)
    #MAPE, MAPEsd = mean_absolute_percentage_error(y_testvec, yvalidvec,1)
    return (100/(MAE))
    
def function2(params, X_train, y_train):
    yvalidvec, y_testvec = Fn_eval_DecisionTree(params,X_train, y_train)
    R_and_P = stats.pearsonr(y_testvec, yvalidvec)
    #tabla_CV.append([MAE, MAEsd, R_and_P[0], R_and_P[1], Fn_obj])
    #ConcordanciaCV.append(Tabla_concordancia(y_testvec,yvalidvec))  
    return np.abs(R_and_P[0])

def build_param_regresor(paramcode,names):
    #Entrega un diccionario
    params_for_algoritm = {}
    for i in range(len(names)):
        params_for_algoritm[names[i]] = paramcode[i]
    return (params_for_algoritm)

def plot_fnvsfn(fn1,fn2,num):
    #Lets plot the final front now
    #function1 = [i * -1 for i in fn1]
    #function2 = [j * -1 for j in fn2]
    sequence_of_colors = ["red","black", "green", "magenta", "dodgerblue", 
                          "blue","gray","brown","orange","black"]
    simbol=['o','+','>','v','*','p','x',
            'D','<',',','*','d','_','+',
            '+','+','+',]
    plt.xlabel('Function 1 (MAE)', fontsize=15)
    plt.ylabel('Function 2 (R)', fontsize=15)
    plt.scatter(fn1, fn2,marker=simbol[num],
                c=sequence_of_colors[num],s=100)
    
# %%MAIN PROGRAM STARTS HERE
pop_size = 100
max_gen = 6

f = open('PARA_NSGAII_opt.pckl', 'rb')
[X_train, y_train, X_test, y_test] = pickle.load(f)
f.close()
#from os import system
#system('load_perinato_database.py')
#X_train=loaddata.X_train
#y_train=loaddata.y_train
# Initialization
        
param_space = {
    #'CV_num':       [9,  11, 'int'],    
    'max_depth':    [2,  12, 'int'],
    'min_samples_split': [2, 6, 'int'],
    'min_samples_leaf': [1, 15, 'int'],
    'max_features': [2, 10, 'int']}

# Population generation (pop_size) Matrix_size = pop_size x traits
vec_solution=[]
for i in range(0,pop_size):
    solution=[]# Crea el vector solución
    for rasgo in param_space.keys():
        items=param_space[rasgo]
        # Se define valores min y max del rasgo
        min_x=items[0]
        max_x=items[1]
        # Se crea un valor aleatorio para ese rasgo
        if 'int' == items[2]:
            num_trait=int(min_x+(max_x-min_x)*random.random()) 
        else: # Crea un flotante
            num_trait=min_x+(max_x-min_x)*random.random() 
        solution.append(num_trait)
    vec_solution.append(solution)  
print('Primera poblacion cero (creada de la nada)')    
print(vec_solution)     

#BEGIN HYPERPARAMETER OPTIMIZATION
gen_no=0
names=[rasgo for rasgo in param_space.keys()]
while(gen_no<max_gen):
    params_sol=[ build_param_regresor(vec_solution[i], names)
    for i in range(0,pop_size)]
    function1_values = [function1(params_sol[i], X_train, y_train)
    for i in range(0,pop_size)]
    function2_values = [function2(params_sol[i], X_train, y_train)
    for i in range(0,pop_size)]
    non_dominated_sorted_solution = fast_non_dominated_sort(
            function1_values[:],function2_values[:])
    print("The best front for Generation number ",gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(vec_solution[valuez],end=", ")
    print("\n")
   
    #for j in range(len(non_dominated_sorted_solution)):
    for valuez in non_dominated_sorted_solution[0]:
        plot_fnvsfn(function1_values[valuez],
                    function2_values[valuez],gen_no)
    
            
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],
                                        function2_values[:],
                                        non_dominated_sorted_solution[i][:]))
    solution2 = vec_solution[:]
    #Generating offsprings
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(vec_solution[a1],vec_solution[b1]))
        
    params_sol=[ build_param_regresor(solution2[i], names) for i in range(0,2*pop_size)]    
    function1_values2 = [function1(params_sol[i], X_train, y_train) for i in range(0,2*pop_size)]
    function2_values2 = [function2(params_sol[i], X_train, y_train) for i in range(0,2*pop_size)]
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],
                                                             function2_values2[:])
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],
                                                           function2_values2[:],
                                                           non_dominated_sorted_solution2[i][:]))
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],
                                                     non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
        #front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    vec_solution = [solution2[i] for i in new_solution]
    gen_no = gen_no + 1

plt.show()
print("ALGORITMO - DecisionTree\n")
print("Evaluacion de los puntos del ultimo frente de pareto\n")
for valuez in non_dominated_sorted_solution[0]:
        print(vec_solution[valuez],end=", ")
        print('MAE y R --> ')
        print(1/(function1_values[valuez]*(1/100)), ',',function2_values[valuez])
        print('\n')
##Lets plot the final front now
#function1 = [i * -1 for i in function1_values]
#function2 = [j * -1 for j in function2_values]
#plt.xlabel('Function 1', fontsize=15)
#plt.ylabel('Function 2', fontsize=15)
#plt.scatter(function1, function2)
#plt.show()
