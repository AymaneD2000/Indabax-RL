#!/usr/bin/env python3
"""
SOLUTIONS COMPLÈTES - Agent CNN pour Deep Q-Learning basé sur les Images
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

class SimpleDQNAgent:
    """Agent DQN avec CNN pour l'apprentissage par renforcement basé sur les images"""
    
    def __init__(self, 
                 state_shape=(84, 84, 4),
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
        SOLUTION EXERCICE 1 - ARCHITECTURE CNN COMPLÈTE
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            
            # Normalisation des pixels (0-255 → 0-1)
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            
            # Couches convolutionnelles pour extraction de features
            tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, strides=1, activation='relu'),
            
            # Aplatir pour les couches denses
            tf.keras.layers.Flatten(),
            
            # Couches denses pour la décision
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
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
        SOLUTION EXERCICE 2 - PRÉPROCESSING DES IMAGES
        """
        # Vérifications de base
        if state is None:
            raise ValueError("État d'entrée ne peut pas être None")
        
        # Vérifier la forme
        if state.shape != self.state_shape:
            raise ValueError(f"Forme attendue {self.state_shape}, reçue {state.shape}")
        
        # Convertir en float32 et normaliser si nécessaire
        state = state.astype(np.float32)
        
        # Si les valeurs sont entre 0-255, les normaliser
        if np.max(state) > 1.0:
            state = state / 255.0
        
        return state
    
    def act(self, state):
        """
        SOLUTION EXERCICE 3 - STRATÉGIE EPSILON-GREEDY POUR IMAGES
        """
        # Calculer epsilon
        self.epsilon = self.get_epsilon()
        
        if np.random.random() < self.epsilon:
            # Exploration - action aléatoire
            return np.random.randint(0, self.action_size)
        else:
            # Exploitation - utiliser le réseau pour prédire
            
            # Préprocesser l'état
            processed_state = self.preprocess_state(state)
            
            # Ajouter dimension batch
            state_tensor = tf.constant(np.expand_dims(processed_state, axis=0))
            
            # Prédire les Q-values
            q_values = self.main_network(state_tensor, training=False)
            
            # Choisir l'action avec la plus haute Q-value
            return int(tf.argmax(q_values[0]).numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """Stocker l'expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """
        SOLUTION EXERCICE 4 - ENTRAÎNEMENT CNN AVEC EXPERIENCE REPLAY
        """
        if len(self.memory) < self.min_replay_size:
            return None
        
        # Échantillonnage du batch
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # Préparer les données du batch
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.bool_)
        
        # Convertir en tenseurs TensorFlow
        states_tensor = tf.constant(states)
        next_states_tensor = tf.constant(next_states)
        
        # Entraînement avec GradientTape
        with tf.GradientTape() as tape:
            # Q-values actuelles du réseau principal
            current_q_values = self.main_network(states_tensor, training=True)
            
            # Sélectionner les Q-values pour les actions prises
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            current_q_values = tf.gather_nd(current_q_values, action_indices)
            
            # Q-values suivantes du réseau cible
            next_q_values = self.target_network(next_states_tensor, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            
            # Q-values cibles (équation de Bellman)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1.0 - tf.cast(dones, tf.float32)))
            
            # Calculer la perte
            loss = tf.keras.losses.MeanSquaredError()(target_q_values, current_q_values)
        
        # Appliquer les gradients
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))
        
        # Mise à jour du réseau cible
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        self.step_count += 1
        self.losses.append(float(loss))
        
        return float(loss)
    
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
    """Fonction d'entraînement complète avec solutions"""
    print("🚀 Démarrage de l'entraînement CNN avec solutions...")
    
    # Initialisation
    game = SnakeGameAI2(w=336, h=336, image_based=True)
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


def demo_cnn_architecture():
    """Démonstration de l'architecture CNN"""
    print("🏗️ Démonstration de l'architecture CNN")
    
    agent = SimpleDQNAgent()
    
    # Afficher le résumé du modèle
    agent.main_network.summary()
    
    # Test avec une image factice
    dummy_state = np.random.randint(0, 255, (84, 84, 4), dtype=np.uint8)
    processed_state = agent.preprocess_state(dummy_state)
    
    print(f"\nTest avec image factice :")
    print(f"État original shape: {dummy_state.shape}")
    print(f"État traité shape: {processed_state.shape}")
    print(f"Valeurs min/max: {processed_state.min():.3f} / {processed_state.max():.3f}")
    
    # Prédiction
    action = agent.act(dummy_state)
    print(f"Action prédite: {action}")


if __name__ == '__main__':
    print("🎮 Solutions CNN + Reinforcement Learning")
    print("\nOptions disponibles :")
    print("1. demo_cnn_architecture() - Voir l'architecture")
    print("2. train_cnn_agent(episodes=100) - Entraîner l'agent")
    
    # Démonstration de l'architecture
    demo_cnn_architecture()
    
    # Entraînement (décommentez pour lancer)
    # agent, scores, avg_scores = train_cnn_agent(episodes=100) 