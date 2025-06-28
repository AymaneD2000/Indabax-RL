#!/usr/bin/env python3
"""
CPU-Safe Snake DQN Training
Maximum compatibility version that avoids all GPU/MPS issues
"""

import os
import sys
import numpy as np
import tensorflow as tf
from collections import deque
import random
from game import SnakeGameAI2
import matplotlib.pyplot as plt

print("🖥️  CPU-Safe Snake DQN Training")
print("=" * 40)
print(f"TensorFlow version: {tf.__version__}")
print("Running on CPU only for maximum compatibility")
print("=" * 40)

class SimpleDQNAgent:
    """Simplified DQN Agent optimized for CPU training and stability"""
    
    def __init__(self, 
                 state_shape=(336, 336, 4),
                 action_size=3,
                 learning_rate=0.0005,  # Adjusted learning rate for better performance
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_episodes=600,  # Faster decay for shorter training
                 memory_size=1000,
                 batch_size=32,  # Smaller batch for CPU
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
        
        # Build networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Memory
        self.memory = deque(maxlen=memory_size)
        
        # Training variables
        self.episode_count = 0
        self.step_count = 0
        self.epsilon = epsilon_start
        self.losses = []
        self.scores = []
        self.avg_scores = []
        
    def _build_network(self):
        """Build a simplified CNN network"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_shape),
            
            # Normalization
            tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
            
            # Simplified CNN
            tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu'),
            tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, strides=1, activation='relu'),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        
        return model
    
    def update_target_network(self):
        """Copy weights from main to target network"""
        self.target_network.set_weights(self.main_network.get_weights())
    
    def get_epsilon(self):
        """Calculate current epsilon"""
        if self.episode_count < self.epsilon_decay_episodes:
            epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * \
                     (self.episode_count / self.epsilon_decay_episodes)
        else:
            epsilon = self.epsilon_end
        return epsilon
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        self.epsilon = self.get_epsilon()
        
        if np.random.random() < self.epsilon:
            print('taking random action')
            return np.random.randint(0, self.action_size)
        print('taking brain action')
        # Ensure correct data types
        state_tensor = tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)
        q_values = self.main_network(state_tensor, training=False)
        return int(tf.argmax(q_values[0]).numpy())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the network"""
        if len(self.memory) < self.min_replay_size:
            return None
        
        # Sample batch
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        # Prepare data with explicit data types
        states = np.array([e[0] for e in batch], dtype=np.float32)
        actions = np.array([e[1] for e in batch], dtype=np.int32)
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.array([e[3] for e in batch], dtype=np.float32)
        dones = np.array([e[4] for e in batch], dtype=np.bool_)
        
        # Convert to tensors
        states_tensor = tf.constant(states)
        next_states_tensor = tf.constant(next_states)
        
        with tf.GradientTape() as tape:
            # Current Q values
            current_q_values = self.main_network(states_tensor, training=True)
            
            # Select Q values for taken actions
            batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
            action_indices = tf.stack([batch_indices, actions], axis=1)
            current_q_values = tf.gather_nd(current_q_values, action_indices)
            
            # Next Q values from target network
            next_q_values = self.target_network(next_states_tensor, training=False)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)
            
            # Target Q values
            target_q_values = rewards + (self.gamma * max_next_q_values * (1.0 - tf.cast(dones, tf.float32)))
            
            # Loss
            loss = tf.keras.losses.MeanSquaredError()(target_q_values, current_q_values)
        
        # Apply gradients
        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))
        
        # Update target network
        if self.step_count % self.target_update_freq == 0:
            self.update_target_network()
        
        self.step_count += 1
        self.losses.append(float(loss))
        
        return float(loss)
    
    def save(self, filepath='./model/'):
        """Save the model"""
        os.makedirs(filepath, exist_ok=True)
        self.main_network.save_weights(os.path.join(filepath, 'cpu_safe_model.weights.h5'))
        
        # Save training state
        np.savez(os.path.join(filepath, 'cpu_safe_state.npz'),
                 episode_count=self.episode_count,
                 step_count=self.step_count,
                 epsilon=self.epsilon,
                 losses=self.losses,
                 scores=self.scores,
                 avg_scores=self.avg_scores)
        
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath='./model/'):
        """Load the model"""
        try:
            self.main_network.load_weights(os.path.join(filepath, 'cpu_safe_model.h5'))
            self.update_target_network()
            
            data = np.load(os.path.join(filepath, 'cpu_safe_state.npz'))
            self.episode_count = int(data['episode_count'])
            self.step_count = int(data['step_count'])
            self.epsilon = float(data['epsilon'])
            self.losses = data['losses'].tolist()
            self.scores = data['scores'].tolist()
            self.avg_scores = data['avg_scores'].tolist()
            
            print(f"📂 Model loaded from {filepath}")
            print(f"Resuming from episode {self.episode_count}")
            
        except FileNotFoundError:
            print("No saved model found. Starting fresh training.")

def train_cpu_safe(episodes=5000):
    """CPU-safe training function"""
    
    print(f"🚀 Starting CPU-safe training for {episodes} episodes")
    print("This may be slower but should be very stable!")
    
    # Initialize
    env = SnakeGameAI2(image_based=True)
    agent = SimpleDQNAgent()
    
    # Try to load existing model
    agent.load()
    
    scores_window = deque(maxlen=100)
    start_episode = agent.episode_count
    
    try:
        for episode in range(start_episode, episodes):
            # Reset environment
            state, _, done, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while not done and steps < 500:  # Max steps per episode
                # Choose action
                action = agent.act(state)
                
                # Take action
                next_state, reward, done, score = env.play_step(np.eye(3)[action])
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train
                loss = agent.train()
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Record scores
            scores_window.append(score)
            agent.scores.append(score)
            agent.episode_count = episode + 1
            avg_score = np.mean(scores_window)
            agent.avg_scores.append(avg_score)
            
            # Print progress
            if episode % 50 == 0:
                print(f"Episode {episode:4d} | Score: {score:2d} | "
                      f"Avg: {avg_score:5.2f} | Epsilon: {agent.epsilon:.3f} | "
                      f"Memory: {len(agent.memory):5d}")
            
            # Save periodically
            if episode % 500 == 0 and episode > 0:
                agent.save()
            
            # Early stopping
            if avg_score > 20.0:
                print(f"\n🎉 Solved in {episode} episodes! Average score: {avg_score:.2f}")
                break
        
        # Final save
        agent.save()
        
        # Plot results
        if agent.scores:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(agent.scores, alpha=0.7, label='Episode Score')
            plt.plot(agent.avg_scores, label='Average Score')
            plt.title('Training Scores')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            if agent.losses:
                plt.plot(agent.losses)
                plt.title('Training Loss')
                plt.xlabel('Training Step')
                plt.ylabel('Loss')
            
            plt.tight_layout()
            plt.savefig('./model/cpu_safe_training.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        print("🎉 Training completed!")
        return agent
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted. Progress saved.")
        agent.save()
        return agent

if __name__ == "__main__":
    print("Starting CPU-safe training...")
    print("This version avoids all GPU/MPS compatibility issues!")
    print("Expected time: 2-4 hours for good results")
    
    trained_agent = train_cpu_safe(episodes=5000) 