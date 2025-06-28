# 📋 Guide de l'Animateur - Workshop CNN + Apprentissage par Renforcement

## 🎯 Vue d'ensemble du Workshop Avancé

**Durée estimée :** 4-6 heures  
**Niveau :** Intermédiaire à Avancé en IA/ML  
**Prérequis :** Bases CNN, TensorFlow, notions d'apprentissage par renforcement

## ⚡ Différences avec le Workshop Basique

### **Complexité Accrue**
| Aspect | Workshop Basique | Workshop CNN |
|--------|------------------|--------------|
| **Public cible** | Débutants ML | Intermédiaires CNN |
| **Concepts** | RL fondamental | RL + Vision |
| **Temps d'exécution** | 5-10 min | 30-60 min |
| **Débogage** | Simple | Complexe (dimensions, mémoire) |

## 📊 Structure Recommandée

### **Phase 1 : Introduction Avancée (45-60 min)**

#### **1. Rappel Rapide RL (10 min)**
- Q-Learning, epsilon-greedy, expérience replay
- *"Vous connaissez déjà ces concepts"*

#### **2. Introduction CNN pour RL (20 min)**
- **Pourquoi les images ?** : Vision humaine vs algorithmes
- **Défis spécifiques** : Dimensions, mémoire, temps de calcul
- **Avantages** : Généralisation, réalisme

#### **3. Architecture CNN (15 min)**
- **Convolution** : Détection de motifs locaux
- **Stride et padding** : Réduction de taille
- **Feature maps** : Représentations abstraites
- **Pipeline complète** : Image → Features → Décision

### **Phase 2 : Démonstration Technique (20 min)**

#### **Analyse du Projet**
- Comparaison avec le projet basique
- Traitement d'images optimisé
- Frame stacking et preprocessing

### **Phase 3 : Travaux Pratiques (3h30-4h)**

## 🔄 Ordre d'Exécution Recommandé

### **EXERCICE 1 : Architecture CNN (60-75 min) - AVANCÉ**

#### **Points d'Attention Critiques :**
- **Dimensions** : Input (84,84,4) → Output (3)
- **Normalisation** : Lambda layer obligatoire
- **Stride vs Pooling** : Stride pour réduction de taille
- **Activation finale** : Linear pour Q-values

#### **Erreurs Fréquentes :**
```python
# ❌ ERREUR : Oublier la normalisation
tf.keras.layers.Conv2D(16, 8, strides=4)

# ✅ CORRECT : Avec normalisation
tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32) / 255.0),
tf.keras.layers.Conv2D(16, 8, strides=4, activation='relu')
```

#### **Aide Progressive :**
1. "Commencez par Input et Lambda pour normalisation"
2. "Conv2D prend (filtres, kernel_size, strides, activation)"
3. "N'oubliez pas Flatten() avant Dense"

---

### **EXERCICE 2 : Preprocessing (30-40 min) - INTERMÉDIAIRE**

#### **Points Clés :**
- **Validation des formes** : Critique pour éviter les crashes
- **Types de données** : float32 obligatoire pour TensorFlow
- **Gestion des plages** : 0-255 vs 0-1

#### **Code Type à Expliquer :**
```python
if state.shape != self.state_shape:
    raise ValueError(f"Attendu {self.state_shape}, reçu {state.shape}")

state = state.astype(np.float32)
if np.max(state) > 1.0:
    state = state / 255.0
```

---

### **EXERCICE 3 : Epsilon-Greedy Images (45-60 min) - INTERMÉDIAIRE**

#### **Nouveautés vs Version Basique :**
- **Preprocessing intégré** dans la pipeline
- **Gestion des dimensions** : expand_dims pour batch
- **TensorFlow tensor** au lieu de numpy

#### **Séquence Critique :**
```python
processed_state = self.preprocess_state(state)  # ✅
state_tensor = tf.constant(np.expand_dims(processed_state, axis=0))  # ✅
q_values = self.main_network(state_tensor, training=False)  # ✅
return int(tf.argmax(q_values[0]).numpy())  # ✅
```

---

### **EXERCICE 4 : Entraînement CNN (75-90 min) - TRÈS AVANCÉ**

#### **Défis Majeurs :**
- **Gestion mémoire** : Arrays d'images volumineuses
- **Dimensions complexes** : 4D tensors
- **Performance** : Beaucoup plus lent que dense

#### **Points Critiques à Surveiller :**
```python
# Gestion des types de données
states = np.array([e[0] for e in batch], dtype=np.float32)  # ⚠️ Obligatoire
actions = np.array([e[1] for e in batch], dtype=np.int32)   # ⚠️ Obligatoire

# Sélection des Q-values pour actions prises
batch_indices = tf.range(tf.shape(actions)[0], dtype=tf.int32)
action_indices = tf.stack([batch_indices, actions], axis=1)
current_q_values = tf.gather_nd(current_q_values, action_indices)
```

## 🆘 Gestion des Problèmes Spécifiques

### **Problèmes de Performance**

#### **"L'entraînement est trop lent"**
```python
# Solutions immédiates :
batch_size = 16  # Au lieu de 32
memory_size = 500  # Au lieu de 1000
SPEED = 400  # Augmenter la vitesse du jeu
```

#### **"Mémoire insuffisante"**
```python
# Réductions nécessaires :
state_shape = (42, 42, 4)  # Au lieu de (84, 84, 4)
batch_size = 8  # Très petit batch
```

### **Problèmes de Debugging**

#### **"Formes incompatibles"**
```python
# Debug systématique :
print(f"State shape: {state.shape}")
print(f"Expected: {agent.state_shape}")
print(f"State dtype: {state.dtype}")
print(f"State min/max: {state.min()}/{state.max()}")
```

#### **"Pas d'amélioration"**
- **Epsilon trop élevé** : Vérifier `get_epsilon()`
- **Learning rate** : Essayer 0.001 ou 0.0001
- **Target update** : Vérifier la fréquence

### **Erreurs TensorFlow Communes**

#### **Type Mismatch**
```python
# ❌ ERREUR
states = np.array([e[0] for e in batch])  # dtype par défaut

# ✅ SOLUTION  
states = np.array([e[0] for e in batch], dtype=np.float32)
```

#### **Dimension Errors**
```python
# Debug dimensions :
print(f"States tensor shape: {states_tensor.shape}")  # Doit être (batch, 84, 84, 4)
print(f"Q-values shape: {current_q_values.shape}")    # Doit être (batch, 3)
```

## 🎯 Adaptations selon Niveau

### **Groupe Intermédiaire**
- **Plus de temps** sur concepts CNN (45 min théorie)
- **Exercice 1 guidé** étape par étape
- **Focus sur debugging** plutôt que optimisations

### **Groupe Avancé**
- **Théorie accélérée** (30 min)
- **Défis bonus** : Double DQN, Dueling DQN
- **Expérimentations** : Hyperparamètres, architectures

## 📊 Monitoring de l'Entraînement

### **Métriques Critiques à Surveiller**

#### **Pendant l'Entraînement :**
```python
# Affichage recommandé :
print(f"Épisode {episode}")
print(f"Score: {score}, Moyenne: {avg_score:.2f}")
print(f"Epsilon: {agent.epsilon:.3f}")
print(f"Memory size: {len(agent.memory)}")
print(f"Loss: {loss:.4f}" if loss else "Training not started")
```

#### **Signaux de Santé :**
- **Epsilon** : Diminue régulièrement de 1.0 vers 0.01
- **Perte** : Décroissance générale (peut fluctuer)
- **Score moyen** : Augmentation lente mais constante
- **Memory** : Atteint min_replay_size rapidement

## ⏱️ Gestion du Temps

### **Timeline Réaliste :**
- **Exercice 1** : 60-75 min (le plus complexe)
- **Exercice 2** : 30-40 min (plus simple)
- **Exercice 3** : 45-60 min (intégration)
- **Exercice 4** : 75-90 min (le plus difficile)
- **Test/Demo** : 30-45 min

### **Si Retard :**
1. **Simplifier Ex1** : Donner l'architecture, expliquer seulement
2. **Accélérer Ex2** : Code ensemble au tableau
3. **Focus Ex3-4** : Cœur du workshop

## 🎮 Phase de Test et Démonstration

### **Résultats Attendus :**
- **0-50 épisodes** : Scores très bas (0-2), mouvement aléatoire
- **50-100 épisodes** : Premiers apprentissages (2-5)
- **100-200 épisodes** : Évite les murs consistentement (5-8)
- **200+ épisodes** : Stratégies émergentes (8-15)

### **Comportements à Pointer :**
- **Exploration massive** au début
- **Graduelle exploitation** avec amélioration
- **Émergence de stratégies** : spirales, contournement

## 💡 Questions Avancées des Participants

### **"Pourquoi 84x84x4 ?"**
- **84x84** : Standard Atari DQN (équilibre résolution/performance)
- **4 frames** : Perception du mouvement et de la direction

### **"Pourquoi grayscale ?"**
- **Réduction dimensionnelle** : 3 canaux → 1 canal
- **Information suffisante** : Formes plus importantes que couleurs
- **Performance** : 3x moins de paramètres

### **"Différence avec classification d'images ?"**
- **Objectif** : Actions optimales vs catégorisation
- **Feedback** : Récompenses vs labels
- **Temporalité** : Séquences vs images isolées

## 🏆 Extensions et Ouvertures

### **Améliorations Techniques :**
- **Double DQN** : Réduction du biais Q-value
- **Dueling DQN** : Séparation valeur/avantage  
- **Prioritized Experience Replay** : Échantillonnage intelligent

### **Autres Applications :**
- **Jeux Atari** : Adaptation directe
- **Robotique** : Vision pour navigation
- **Véhicules autonomes** : Perception visuelle

---

**Bonne Animation de Workshop Avancé ! 🚀🧠**

*N'oubliez pas : CNN + RL = révolution de l'IA moderne !* 