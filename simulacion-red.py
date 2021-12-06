from math import log10
import types
from typing import List, Tuple
import numpy as np
from numpy.random import default_rng
# Graficas:
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
CONSIDERACIONES
	* Empezar por nodo y grado mas alejado. (Primero Grado 7!!!)
	* Para cada grado a analizar:
		+ Verificar cuantos nodos tienen un buffer con espacio.
	* En un grado:
		+ Identificar a los nodos que pertenecen a este grado.
		+ Identificar a los nodos que tienen algo que Tx.
			- Si hay algo que Tx:
				Se genera el contador de backoff para los nodos que tienen algo que Tx.
				Se seleccionan a los nodos ganadores:
					Si no hay colision 
					-> Eliminamos el pkt transmitido del nodo ganador
					-> En el grado inferior:
						Si hay espacio, se inserta el pkt recibido.
						Si el pkt llego al sink, tomar estadisticas
					Si hay colision
					-> Eliminamos los pkts que colisionaron de los nodos ganadores
					Se incrementa el tiempo de simulacion
					-> t_sim  += t_slot
PARAMETROS DE DESEMPENO A MEDIR:
	* Paquetes perdidos (POR GRADO!!!)
	* Throughput del sistema  pkt / ciclos
	* Retardo ent-to-end (POR GRADO!!!)
			source-to-sink
LA evaluacion sera con diferentes valores de 
	* N, W, lambda
------------------------------------------------------------------
"""
# INICIALIZACION PARAMETROS:
DIFS	= 10e-3		# 
SIFS	= 5e-3		# 
t_RTS	= 11e-3		# Tiempo de Tx de pkt RTS
t_CTS	= 11e-3		# 
t_ACK	= 11e-3		# Tiempo de Tx de pkt de confirmacion
t_DATA	= 43e-3		# Tiempo de Tx de pkt de datos
Sigma	= 1e-3		# Tiempode miniranura
H		= 7			# Numero de grados en red
K		= 15		# Espacio en cada Buffer
Xi		= 18		# Num. ranuras de tiempo estado Sleeping
N		= [5, 10, 15, 20]				# Num Nodos por Grado
W		= [16, 32, 64, 128, 256]		# Num Max de miniranuras
Lambda	= [0.0005, 0.001, 0.005, 0.03]	# Tasa de generacion de pkts por grado
sink 	= 0			# ID nodo sink
N_case	= N[0]
Lambda_case = Lambda[0]
W_case = W[0]
pkts_perdidos = []
retardos_pkts = []


def simulacion(iter):
	global W_case, N_case, Lambda_case, pkts_perdidos, retardos_pkts
	# INICIALIZACION VARIABLES DE SIMULACION:
	t_sim		= 0						# Este siempre ira avanzando (No se mide de forma continua, solo queremos ir midiendo instantes en que se van generando los eventos)
	t_arribo	= 0						# Nuevo tiempo de arribo, con esto encontramos el siguiente tiempo en que se genera un arribo
	t_slot		= DIFS + 3*SIFS + (Sigma*W_case) + t_RTS + t_CTS + t_ACK + t_DATA
	Tc 			= (2 + Xi) * t_slot		# Tiempo_Tx + Tiempo_Rx + Tiempo_Sleeping


	nodos_perdidos	= np.zeros( (H+1,1), dtype=np.int32 )
	retardo_pkts_grado = np.zeros( (H+1,1), dtype=np.int32 )
	contador_pkts 	= 0
	buffer 		= [[]*N_case]*(H+1)  # Para cada nodo hay una lista de Pkts (se almacena time de creacion)
	# print(np.shape(buffer))		# buffer[GRADO][NODO] = LIST(ta_NODO1,ta_NODO2,...)
	
	while t_sim < 10000*Tc:
		while t_arribo < t_sim : # Ya ocurrio un arribo en la red
			contador_pkts += 1
			print('ta:', t_arribo, ' | tsim:', t_sim)
			# Se_asigna_el_pkt_recibido_a_un_nodo_en_un_grado
			grado_rand, nodo_rand = select_grade_and_node()

			if len(buffer[grado_rand][nodo_rand]) < K:
				buffer[grado_rand][nodo_rand].append({t_arribo,grado_rand})
			else:
				nodos_perdidos[grado_rand] += 1

			# Se define un nuevo tiempo de arribo
			t_arribo = generate_arrival_time(t_sim)

		# test(buffer, nodos_perdidos)
		
		nodos_por_Tx = []
		for grado_i in range(H, 0, -1):
			# Verificar cuantos nodos tienen un buffer con datos.
			nodos_contendientes = get_nodes_with_data(grado_i, buffer)
			# Proceso de contencion (se genera el contador de Backoff para estos nodos contendientes en el mismo  grado)
			# nodo_ganador = contencion_del_canal(nodos_contendientes)
			nodos_contendientes.sort()

			nodo_winner = -1
			for i in range(0, len(nodos_contendientes)-1):
				if nodo_winner != -1:
					break
				if nodos_contendientes[i][0] < nodos_contendientes[i+1][0]:
					nodo_winner = nodos_contendientes[i][1]
				else:
					nodos_perdidos[grado_i] += 1
					buffer[grado_i][nodos_contendientes[i][1]].pop(0)
					buffer[grado_i][nodos_contendientes[i+1][1]].pop(0)
					nodos_contendientes.pop(i)
					nodos_contendientes.pop(i)

			pkt_tx = buffer[grado_i][nodo_winner].pop(0)
			# Recibir pkt en siguiente nodo
			if grado_i == 1:
				nodo_rand = 0
			else:
				_, nodo_rand = select_grade_and_node()
			if len(buffer[grado_i - 1][nodo_rand]) < K:
				buffer[grado_i-1][nodo_rand].append(pkt_tx)
			else:
				nodos_perdidos[grado_i-1] += 1
		
		# SINK
		for pkt in buffer[0][0]:
			retardo_pkt = t_sim - pkt[0]
			retardo_pkts_grado[pkt[1]].append(retardo_pkt)

		t_sim += t_slot
	pkts_perdidos.append(nodos_perdidos)
	ret = np.average(retardo_pkts_grado, axis=1)
	retardos_pkts.append(ret)


rng_generacion_pkt_grado = default_rng()
rng_generacion_pkt_nodo = default_rng()
def select_grade_and_node():
	grade = rng_generacion_pkt_grado.integers(1, H + 1, dtype=np.int32)
	node = rng_generacion_pkt_nodo.integers(0, N_case, dtype=np.int32)
	return grade, node


rng_arrival = default_rng()
VA_uniforme	= rng_arrival.uniform(0., 1., 100_000_000)
VA_i = 0
def generate_arrival_time(current_time):
	global VA_i
	nuevo_t		= - (1 / Lambda_case) * log10(1 - VA_uniforme[VA_i])  # Siguiente intervalo de tiempo en que se va a generar un pkt
	VA_i += 1
	return ( current_time + nuevo_t )


def get_nodes_with_data(grado_i, buffer):
	global W_case
	nodes = []
	for node in range(0, N_case):
		if(len(buffer[grado_i][node]) > 0):
			num_backoff = rng_arrival.integers(0, W_case)
			nodes.append({num_backoff, node})
	return nodes


# def contencion_del_canal(nodos_contendientes):
# 	# 	1) COMO SE ELIGE EL NODO GANADOR?
# 	# 	2) HAY COLISION ENTRE NODOS DEL MISMO GRADO Y/O DE DIFERENTE GRADO?
# 	# 	3) QUE PASA CUANDO COLISIONAN LOS NODOS GANADOREs?
# 	# nodos = []
# 	# for nodo in nodos_contendientes:
# 	# 	backoff_time = rng_arrivalintegers()
# 	# 	nodos.append({backoff_time, nodo})
# 	# nodos.sort()
# 	# nodo_ganador = nodos[0][1]
# 	idx_ganador = rng_arrival.integers(0, len(nodos_contendientes))
# 	return nodos_contendientes[idx_ganador]	if len(nodos_contendientes) > 0 else -1


# def insertar_pkt_recibido(grado):
# 	if grado == sink: # Pkt llego a destino
# 		Tomar estadistica, pkt exitoso
# 	else:
# 		aumentar buffer de nodo


def test(buffer, nodos_perdidos):
	print('buffer:', np.shape(buffer), 'perdidos:', np.shape(nodos_perdidos))


def len_buff(buffer):
	print(type(buffer))
	print(np.shape(buffer))
	count = 0
	for elem in buffer:
		count += 1
	return count


def graficar_con_interpolacion(x, y, x_lbl, y_lbl, my_title):
	fig, ax = plt.subplots()

	y_interp = interp1d(x, y, kind='cubic')
	ax.plot(x, y_interp, '-')
	ax.set_xlabel(x_lbl)
	ax.set_ylabel(y_lbl)
	ax.set_title(my_title)


def generar_graficas():
	pass


if __name__ == '__main__':
	global W_case, N_case, Lambda_case
	for iter, n_case in zip(range(0,len(N)), N):
		N_case = n_case
		W_case = W[0]
		Lambda_case = Lambda[0]
		simulacion(iter)
	generar_graficas()
	pkts_perdidos = []