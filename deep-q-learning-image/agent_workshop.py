#!/usr/bin/env python3
"""
🎮 WORKSHOP : Deep Q-Learning avec Vision par Ordinateur
Apprentissage par Renforcement basé sur les Images

Ce workshop vous apprendra à créer un agent qui "voit" le jeu à travers des images,
exactement comme le ferait un humain !
"""

import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import random
from game import SnakeGameAI2
import matplotlib.pyplot as plt
import cv2

print("🎯 Workshop : CNN + Reinforcement Learning")
print("=" * 50)
print(f"TensorFlow version: {tf.__version__}")
print("Apprentissage par vision artificielle !")
print("=" * 50)

class SimpleDQNAgent:
    """
    Agent DQN simplifié pour l'apprentissage par renforcement basé sur les images
    """
    
    def __init__(self, 
                 state_shape=(84, 84, 4),  # 84x84 pixels, 4 frames empilées
                 action_size=3,
                 learning_rate=0.0005,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_episodes=600,
                 memory_size=1000,
                 batch_size=32,
                 target_update_freq=100,
                 min_replay_size=300):
        
        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        
        # Construction des réseaux
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Optimiseur
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Mémoire pour l'experience replay
        self.memory = deque(maxlen=memory_size)
        
        # Variables d'entraînement
        self.episode_count = 0
        self.step_count = 0
        self.epsilon = epsilon_start
        self.losses = []
        self.scores = []
        self.avg_scores = []
        
    def _build_network(self):
        """
        EXERCICE 1 - ARCHITECTURE CNN POUR LA VISION (★★★ AVANCÉ)
        
        Votre mission : Créer un réseau de neurones convolutionnel (CNN) qui peut "voir" 
        et comprendre les images du jeu Snake.
        
        POURQUOI UN CNN ?
        - Les réseaux denses ne comprennent pas la structure spatiale des images
        - Les CNN détectent les motifs locaux (formes, bords, objets)
        - Chaque couche détecte des motifs plus complexes
        
        ARCHITECTURE RECOMMANDÉE :
        1. Normalisation : Diviser les pixels par 255 (0-1 au lieu de 0-255)
        2. Couches convolutionnelles pour extraire les features
        3. Couches denses pour la décision finale
        
        COUCHES CONVOLUTIONNELLES :
        - Conv2D(16, 8, strides=4) : 16 filtres, taille 8x8, stride 4
        - Conv2D(32, 4, strides=2) : 32 filtres, taille 4x4, stride 2  
        - Conv2D(32, 3, strides=1) : 32 filtres, taille 3x3, stride 1
        
        INTUITION :
        - Première couche : détecte les bords et formes simples
        - Deuxième couche : combine les bords en motifs plus complexes
        - Troisième couche : reconnaît les objets (serpent, nourriture)
        """
        
        # TODO: Créez votre architecture CNN
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            
            # ÉTAPE 1: Normalisation des pixels (0-255 → 0-1)
            # TODO: Ajoutez une couche Lambda pour diviser par 255.0
            # Indice: tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0)
            
            # ÉTAPE 2: Couches convolutionnelles
            # TODO: Première couche Conv2D - 16 filtres, kernel 8x8, stride 4, activation 'relu'
            # TODO: Deuxième couche Conv2D - 32 filtres, kernel 4x4, stride 2, activation 'relu'  
            # TODO: Troisième couche Conv2D - 32 filtres, kernel 3x3, stride 1, activation 'relu'
            
            # ÉTAPE 3: Aplatir pour les couches denses
            # TODO: Ajoutez tf.keras.layers.Flatten()
            
            # ÉTAPE 4: Couches denses pour la décision
            # TODO: Couche Dense de 256 neurones avec activation 'relu'
            # TODO: Couche de sortie - self.action_size neurones, activation 'linear'
        ])
        
        # TODO: Supprimez cette ligne quand vous avez implémenté le modèle
        raise NotImplementedError("Implémentez votre architecture CNN dans _build_network()")
        
        return model
    
    def update_target_network(self):
        """Copie les poids du réseau principal vers le réseau cible"""
        self.target_network.set_weights(self.main_network.get_weights())
    
    def get_epsilon(self):
        """Calcule la valeur actuelle d'epsilon pour l'exploration"""
        if self.episode_count < self.epsilon_decay_episodes:
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                     (self.episode_count / self.epsilon_decay_episodes)
        else:
            epsilon = self.epsilon_end
        return epsilon
    
    def preprocess_state(self, state):
        """
        EXERCICE 2 - PRÉPROCESSING DES IMAGES (★★☆ INTERMÉDIAIRE)
        
        Votre mission : Préparer les images pour l'entrée du réseau CNN.
        
        POURQUOI LE PRÉPROCESSING ?
        - Les images brutes sont trop grandes et variables
        - Il faut normaliser les données pour l'entraînement
        - Éliminer les détails non essentiels
        
        ÉTAPES RECOMMANDÉES :
        1. Vérifier la forme de l'état d'entrée
        2. S'assurer que les valeurs sont dans la bonne plage
        3. Convertir en float32 si nécessaire
        
        FORMES ATTENDUES :
        - Entrée : (84, 84, 4) ou (height, width, channels)
        - Sortie : (84, 84, 4) normalisée
        """
        
        # TODO: Implémentez le préprocessing
        
        # Vérifications de base
        if state is None:
            raise ValueError("État d'entrée ne peut pas être None")
        
        # TODO: Vérifiez que state a la bonne forme (84, 84, 4)
        # Indice: state.shape should equal self.state_shape
        
        # TODO: Convertissez en float32 et normalisez si nécessaire
        # Indice: state = state.astype(np.float32)
        # Indice: Si les valeurs sont entre 0-255, divisez par 255.0
        
        # TODO: Supprimez cette ligne quand vous avez implémenté
        pass
        
        return state
    
    def act(self, state):
        """
        EXERCICE 3 - STRATÉGIE EPSILON-GREEDY POUR IMAGES (★★☆ INTERMÉDIAIRE)
        
        Votre mission : Adapter la stratégie epsilon-greedy pour les images.
        
        DIFFÉRENCES AVEC LE PROJET PRÉCÉDENT :
        - L'état est maintenant une image 4D au lieu d'un vecteur 1D
        - Il faut préprocesser l'image avant de la donner au modèle
        - Le modèle CNN prédit directement les Q-values
        
        ÉTAPES :
        1. Calculer epsilon actuel
        2. Si exploration : action aléatoire
        3. Si exploitation : utiliser le CNN pour prédire
        """
        
        # TODO: Calculez epsilon
        self.epsilon = self.get_epsilon()
        
        # TODO: Implémentez la stratégie epsilon-greedy
        
        if np.random.random() < self.epsilon:
            # TODO: Exploration - retournez une action aléatoire
            # Indice: np.random.randint(0, self.action_size)
            pass
        else:
            # TODO: Exploitation - utilisez le réseau pour prédire
            
            # Étape 1: Préprocesser l'état
            # processed_state = self.preprocess_state(state)
            
            # Étape 2: Ajouter dimension batch (modèle attend (batch, height, width, channels))
            # state_tensor = tf.constant(np.expand_dims(processed_state, axis=0))
            
            # Étape 3: Prédire les Q-values
            # q_values = self.main_network(state_tensor, training=False)
            
            # Étape 4: Choisir l'action avec la plus haute Q-value
            # return int(tf.argmax(q_values[0]).numpy())
            
            pass
    
    def remember(self, state, action, reward, next_state, done):
        """Stocker l'expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """
        EXERCICE 4 - ENTRAÎNEMENT CNN AVEC EXPERIENCE REPLAY (★★★ AVANCÉ)
        
        Votre mission : Adapter l'entraînement pour les images et les CNNs.
        
        DÉFIS SPÉCIFIQUES AUX IMAGES :
        - Images plus grandes → plus de mémoire
        - CNNs plus complexes → entraînement plus lent
        - Batches d'images → gestion des dimensions
        
        ARCHITECTURE D'ENTRAÎNEMENT :
        1. Échantillonner un batch d'expériences
        2. Séparer états, actions, récompenses, etc.
        3. Calculer les Q-values actuelles et cibles
        4. Optimiser avec gradient descent
        """
        
        if len(self.memory) < self.min_replay_size:
            return None
        
        # Échantillonnage du batch
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # TODO: Préparez les données du batch
        # Séparez les éléments du batch
        # states = np.array([e[0] for e in batch], dtype=np.float32)
        # actions = np.array([e[1] for e in batch], dtype=np.int32)
        # rewards = np.array([e[2] for e in batch], dtype=np.float32)
        # next_states = np.array([e[3] for e in batch], dtype=np.float32)
        # dones = np.array([e[4] for e in batch], dtype=np.bool_)
        
        # TODO: Convertissez en tenseurs TensorFlow
        # states_tensor = tf.constant(states)
        # next_states_tensor = tf.constant(next_states)
        
        # TODO: Implémentez l'entraînement avec GradientTape
        with tf.GradientTape() as tape:
            # TODO: Q-values actuelles du réseau principal
            # current_q_values = self.main_network(states_tensor, training=True)
            
            # TODO: Sélectionnez les Q-values pour les actions prises
            # batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            # action_indices = tf.stack([batch_indices, actions], axis=1)
            # current_q_values = tf.gather_nd(current_q_values, action_indices)
            
            # TODO: Q-values suivantes du réseau cible
            # next_q_values = self.target_network(next_states_tensor, training=False)
            # max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            
            # TODO: Q-values cibles (équation de Bellman)
            # target_q_values = rewards + (self.gamma * max_next_q_values * (1.0 - tf.cast(dones, tf.float32)))
            
            # TODO: Calculez la perte
            # loss = tf.keras.losses.MeanSquaredError()(target_q_values, current_q_values)
            pass
        
        # TODO: Appliquez les gradients
        # gradients = tape.gradient(loss, self.main_network.trainable_variables)
        # self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))
        
        # Mise à jour du réseau cible
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        self.step_count += 1
        # TODO: Enregistrez la perte
        # self.losses.append(float(loss))
        # return float(loss)
        
        return None
    
    def save(self, filepath='./model/'):
        """Sauvegarder le modèle"""
        os.makedirs(filepath, exist_ok=True)
        self.main_network.save_weights(os.path.join(filepath, 'cnn_model.weights.h5'))
        
        # Sauvegarder l'état d'entraînement
        np.savez(os.path.join(filepath, 'cnn_state.npz'),
                 episode_count=self.episode_count,
                 step_count=self.step_count,
                 epsilon=self.epsilon,
                 losses=self.losses,
                 scores=self.scores,
                 avg_scores=self.avg_scores)
        
        print(f"💾 Modèle CNN sauvegardé dans {filepath}")


def train_cnn_agent(episodes=1000):
    """
    Fonction d'entraînement principal pour le workshop
    """
    print("🚀 Démarrage de l'entraînement CNN...")
    
    # Initialisation
    game = SnakeGameAI2(w=336, h=336, image_based=True)  # Taille adaptée pour CNN
    agent = SimpleDQNAgent(state_shape=(84, 84, 4))
    
    scores = []
    avg_scores = []
    total_score = 0
    record = 0
    
    for episode in range(episodes):
        # Réinitialiser le jeu
        state, _, _, _ = game.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            # Choisir une action
            action = agent.act(state)
            
            # Exécuter l'action
            next_state, reward, done, score = game.play_step([1 if i == action else 0 for i in range(3)])
            
            # Stocker l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Passer au state suivant
            state = next_state
            total_reward += reward
            
            # Entraîner l'agent
            if len(agent.memory) > agent.min_replay_size:
                loss = agent.train()
        
        # Statistiques
        agent.episode_count += 1
        scores.append(score)
        total_score += score
        avg_score = total_score / (episode + 1)
        avg_scores.append(avg_score)
        
        if score > record:
            record = score
            agent.save()
        
        # Affichage des progrès
        if episode % 10 == 0:
            print(f"Épisode {episode}, Score: {score}, Moyenne: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Record: {record}")
    
    print("🎉 Entraînement terminé !")
    return agent, scores, avg_scores


if __name__ == '__main__':
    """
    Point d'entrée du workshop
    """
    print("🎮 Lancement du Workshop CNN + Reinforcement Learning")
    print("\nPour commencer :")
    print("1. Complétez l'Exercice 1 : _build_network()")
    print("2. Complétez l'Exercice 2 : preprocess_state()")  
    print("3. Complétez l'Exercice 3 : act()")
    print("4. Complétez l'Exercice 4 : train()")
    print("\nPuis lancez l'entraînement avec train_cnn_agent()")
    
    # Décommentez quand les exercices sont terminés :
    # agent, scores, avg_scores = train_cnn_agent(episodes=100) 