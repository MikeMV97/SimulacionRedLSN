from math import log10
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

# INICIALIZACION PARAMETROS:
DIFS	= 10e-3		# Tiempo de Espacio entre tramas para poder transmitir
SIFS	= 5e-3		# Tiempo para procesar y responder a una trama
t_RTS	= 11e-3		# Tiempo de Tx de pkt RTS
t_CTS	= 11e-3		# Tiempo de Tx de pkt CTS
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
throughputs_sim = []

def simulacion(iter):
	global W_case, N_case, Lambda_case, pkts_perdidos, retardos_pkts, contador_pkts
	# INICIALIZACION VARIABLES DE SIMULACION:
	t_sim		= 0.001					# Este siempre ira avanzando (No se mide de forma continua, solo queremos ir midiendo instantes en que se van generando los eventos)
	t_arribo	= 0						# Nuevo tiempo de arribo, con esto encontramos el siguiente tiempo en que se genera un arribo
	t_slot		= DIFS + 3*SIFS + (Sigma*W_case) + t_RTS + t_CTS + t_ACK + t_DATA
	Tc 			= (2 + Xi) * t_slot		# Tiempo_Tx + Tiempo_Rx + Tiempo_Sleeping

	nodos_perdidos	= np.zeros( (H+1,1), dtype=np.int32 )
	retardo_pkts_grado = generar_lista_lista()
	contador_pkts 	= 0
	buffer 			= generar_buffer()
	NUM_CICLOS		= 300_000
	while t_sim < NUM_CICLOS*Tc:

		# print('__________GENERACION PKTS_____________')
		while t_arribo < t_sim : # Ya ocurrio un arribo en la red			
			contador_pkts += 1
			# Se_asigna_el_pkt_recibido_a_un_nodo_en_un_grado
			grado_rand, nodo_rand = select_grade_and_node()

			if len(buffer[grado_rand][nodo_rand]) < K:
				packet = generar_pkt(t_arribo, grado_rand)
				buffer[grado_rand][nodo_rand].append(packet)
			else:
				nodos_perdidos[grado_rand] += 1

			# Se define un nuevo tiempo de arribo
			t_arribo = generate_arrival_time((t_arribo + 2*t_sim ) / 3.)  # (t_sim)

		# print('__________ENVIO PKTS_____________')
		nodos_contendientes = []
		for grado_i in range(H, 0, -1):
			# print('\t__________GRADO', grado_i, '_____________')
			# Verificar cuantos nodos tienen un buffer con datos.
			nodos_contendientes = get_nodes_with_data(grado_i, buffer)
			# Proceso de contencion (se genera el contador de Backoff para estos nodos contendientes en el mismo  grado)
			# nodo_ganador = contencion_del_canal(nodos_contendientes)
			nodos_contendientes.sort()
			
			nodo_winner = -1
			if len(nodos_contendientes) == 1:
				nodo_winner = nodos_contendientes[0][1]
			elif len(nodos_contendientes) > 1:
				i = 0
				while i < len(nodos_contendientes) - 1:
					try:
						if nodos_contendientes[i][0] < nodos_contendientes[i+1][0]:
							nodo_winner = nodos_contendientes[i][1]
							break
						else: # Son iguales los backoff
							nodos_perdidos[grado_i] += 2
							eliminado = buffer[grado_i][nodos_contendientes[i][1]].pop(0)
							if len(buffer[grado_i][nodos_contendientes[i+1][1]]) > 0:
								eliminado = buffer[grado_i][nodos_contendientes[i+1][1]].pop(0)
								i += 1
					except:
						print('i:', i, 'Contendientes:', nodos_contendientes, 'len(n_cont):', len(nodos_contendientes))
						print('BUFFER Nodo 1:', buffer[grado_i][nodos_contendientes[i][1]])
						print('BUFFER Nodo 2:', buffer[grado_i][nodos_contendientes[i+1][1]])
					i += 1

			if nodo_winner != -1 and len(buffer[grado_i][nodo_winner]) > 0:
				pkt_tx = buffer[grado_i][nodo_winner].pop(0)
				# Recibir pkt en siguiente nodo
				nodo_rx = 0
				if grado_i > 1:
					_, nodo_rx = select_grade_and_node()

				if len(buffer[grado_i - 1][nodo_rx]) < K:
					buffer[grado_i-1][nodo_rx].append(pkt_tx)
				else:
					nodos_perdidos[grado_i-1] += 1
		# print('\t___Leer SINK: ___')
		# SINK
		if len(buffer[0]) > 0:
			for pkt in buffer[0][0]:
				retardo_pkt = t_sim - pkt[0]
				retardo_pkts_grado[pkt[1]].append(retardo_pkt)
			while len(buffer[0][0]) > 0:
				buffer[0][0].pop(0)

		t_sim += Tc
	pkts_perdidos.append(nodos_perdidos)
	throughputs_sim.append(contador_pkts / NUM_CICLOS)
	retardos_pkts.append(get_promedio_retardos(retardo_pkts_grado))


def generar_pkt(tiempo_creacion, grado):
	pkt = []
	pkt.append(tiempo_creacion)
	pkt.append(grado)
	return pkt

def generar_nodo():
	nod = []
	return nod

def generar_grad(num):
	grado = []
	for i in range(num):
		n = generar_nodo()
		grado.append(n)
	return grado

def generar_buffer():
	buf = []
	for i in range(H+1):
		grado = generar_grad(N_case)
		buf.append(grado)
	return buf

def generar_lista_lista():
	lista_de_listas = []
	for g in range(H+1):
		nd = generar_nodo()
		lista_de_listas.append(nd)
	return lista_de_listas


rng_generacion_pkt_grado = default_rng()
rng_generacion_pkt_nodo = default_rng()
def select_grade_and_node():
	grade = rng_generacion_pkt_grado.integers(1, H + 1, dtype=np.int32)
	node  = rng_generacion_pkt_nodo.integers(0, N_case, dtype=np.int32)
	return grade, node


rng_arrival = default_rng()
VA_uniforme	= rng_arrival.uniform(0., 0.1, 1_000_000)
VA_i = 0
def generate_arrival_time(current_time):
	global VA_i
	nuevo_t	= - (1 / Lambda_case) * log10(1 - VA_uniforme[VA_i])  # Siguiente intervalo de tiempo en que se va a generar un pkt
	VA_i 	= (VA_i+1) % 1_000_000
	return ( current_time + nuevo_t )


def get_nodes_with_data(grado_i, buffer):
	global W_case
	nodes = []
	for node in range(0, N_case):
		if len(buffer[grado_i]) == 0:
			continue
		if len(buffer[grado_i][node]) > 0:
			num_backoff = rng_arrival.integers(0, W_case)
			nodes.append([num_backoff, node])
	return nodes


def get_promedio_retardos(retardo_pkts_grado):
	rets = []
	for ret in retardo_pkts_grado:
		if len(ret):
			rets.append(np.average(ret))
		else:
			rets.append(0.)
	return rets


def grafica_throughput(variable):
	fig = plt.figure()
	fig.suptitle('Throughput')
	if variable == 'N':
		x = N
	if variable == 'lambda':
		x = Lambda
	if variable == 'omega':
		x = W
	plt.plot(x, throughputs_sim)
	plt.ylabel('pkt / ciclo')
	if variable == 'N':
		plt.title('N variable')
		plt.xlabel('N (Nodos por grado)')
	if variable == 'lambda':
		plt.title('λ variable')
		plt.xlabel('λ Pkt / seg')
	if variable == 'omega':
		plt.title('ω variable')
		plt.xlabel('Número de miniranuras ω')
	plt.grid(True)
	plt.show()

def grafica_pkts_perdidos(x, legends, variable):
	fig = plt.figure()
	fig.suptitle('Paquetes perdidos')
	for pkts_lost in pkts_perdidos:
		plt.plot(x, pkts_lost)
	plt.legend(legends)
	plt.xlabel('Grado')
	plt.ylabel('Pkts')
	if variable == 'N':
		plt.title('Variando N, λ = '+str(Lambda_case)+', ω = '+str(W_case))
	if variable == 'lambda':
		plt.title('N = '+str(N_case)+', Variando λ, ω = '+str(W_case))
	if variable == 'omega':
		plt.title('N = '+str(N_case)+', λ = '+str(Lambda_case)+', Variando ω')
	plt.grid(True)
	plt.show()

def grafica_retardos(x, legends, variable):
	fig = plt.figure()
	fig.suptitle('Retardo source-to-sink')
	for ret_pkts in retardos_pkts:
		plt.plot(x, ret_pkts)
	plt.legend(legends)
	plt.xlabel('Grado')
	plt.ylabel('Segundos')
	if variable == 'N':
		plt.title('Variando N, λ = '+str(Lambda_case)+', ω = '+str(W_case))
	if variable == 'lambda':
		plt.title('N = '+str(N_case)+', Variando λ, ω = '+str(W_case))
	if variable == 'omega':
		plt.title('N = '+str(N_case)+', λ = '+str(Lambda_case)+', Variando ω')
	plt.grid(True)
	plt.show()

def generar_graficas(variable):
	if variable == 'N':
		legends = ['N = ' + str(nn) for nn in N]
	if variable == 'lambda':
		legends = ['λ = ' + str(ll) for ll in Lambda]
	if variable == 'omega':
		legends = ['ω = ' + str(ww) for ww in W]
	x = range( H + 1 )
	grafica_pkts_perdidos(x, legends, variable)
	grafica_retardos(x, legends, variable)
	grafica_throughput(variable)

if __name__ == '__main__':
	print('________________N variable___________________')
	for iter, n_case in zip(range(0,len(N)), N):
		N_case = n_case
		W_case = W[0]
		Lambda_case = Lambda[0]*N_case*H
		simulacion(iter)
	generar_graficas('N')
	print('________________Lambda variable___________________')
	pkts_perdidos = []
	retardos_pkts = []
	throughputs_sim = []
	for iter, lamb_case in zip(range(0,len(Lambda)), Lambda):
		N_case = N[1]
		W_case = W[1]
		Lambda_case = lamb_case*N_case*H
		simulacion(iter)
	generar_graficas('lambda')
	print('________________Omega variable___________________')
	pkts_perdidos = []
	retardos_pkts = []
	throughputs_sim = []
	for iter, omega_case in zip(range(0,len(W)), W):
		N_case = N[1]
		W_case = omega_case
		Lambda_case = Lambda[2]*N_case*H
		simulacion(iter)
	generar_graficas('omega')