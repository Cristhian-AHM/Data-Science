import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#Extraer los datos y darles formato
data = fetch_movielens(min_rating=4.0)

#Imprimimos lo datos de entrenamiento y de prueba
print(repr(data['train']))
print(repr(data['test']))

#Creacion del modelo - Loss function mide la diferencia entre la prediccion del modelo y la salida deseada
model = LightFM(loss='warp')

#Entrenar el modelo
model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
	
	#Numero de usuarios y pelicula en los datos de entrenamientos
	n_users, n_items = data['train'].shape
	
	#Generando recomendaciones para cada usuario
	for user_id in user_ids:
		
		#Peliculas que conocemos su calificacion
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
		
		#Peliculas que el modelo predice que les gustara
		scores = model.predict(user_id, np.arange(n_items))
		#Ordenar los resultados del menor al mayor
		top_items = data['item_labels'][np.argsort(-scores)]
		
		#Imprimir los resultados
		print("User %s" % user_id)
		print("      Peliculas que le gustaron:")
		
		for x in known_positives[:3]:
			print("          %s" % x)
			
		print("     Recomendadas:")
		
		for x in top_items[:3]:
			print("      %s" % x)
		

sample_recommendation(model, data, [3,25,58])
