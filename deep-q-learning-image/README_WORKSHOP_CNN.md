# 🧠 Workshop Avancé : CNN + Apprentissage par Renforcement

Bienvenue dans ce workshop avancé ! Vous allez apprendre à créer un agent qui "voit" le jeu à travers des images, exactement comme le ferait un humain, en utilisant les **Réseaux de Neurones Convolutionnels (CNN)**.

## 🎯 Objectifs Pédagogiques

À la fin de ce workshop, vous maîtriserez :
- **Architecture CNN** pour la vision par ordinateur
- **Traitement d'images** pour l'IA
- **Deep Q-Learning** avec des entrées visuelles
- **Experience Replay** pour l'apprentissage par renforcement
- **Target Networks** pour stabiliser l'entraînement

## 🔥 Nouveautés par rapport au Workshop Précédent

### **Évolution Majeure : De Vecteurs à Images**
| Aspect | Workshop Basique | Workshop CNN |
|--------|------------------|--------------|
| **Entrée** | 11 nombres (dangers, directions) | Images 84x84x4 |
| **Réseau** | Dense (256 neurones) | CNN (Conv2D + Dense) |
| **Preprocessing** | Aucun | Redimensionnement, Grayscale |
| **Complexité** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Temps d'entraînement** | ~5 minutes | ~30-60 minutes |

## 🏗️ Architecture du Projet

```
├── game.py              # ✅ Jeu Snake optimisé pour CNN (complet)
├── agent_workshop.py    # 🔧 Agent CNN à compléter (4 EXERCICES)
├── solutions/           # 💡 Solutions complètes
└── README_WORKSHOP_CNN.md
```

## 📚 Les 4 Exercices Avancés

### 🥇 **EXERCICE 1 : Architecture CNN** (★★★ Avancé)
**Fichier :** `agent_workshop.py` → fonction `_build_network()`

**Mission :** Créer un CNN qui transforme des images en décisions.

**Concepts clés :**
- **Convolution** : Détection de motifs locaux
- **Pooling implicite** : Réduction de taille avec stride
- **Feature maps** : Représentations abstraites des images

**Architecture à implémenter :**
```
Input (84, 84, 4) → Normalisation → Conv2D → Conv2D → Conv2D → Dense → Output (3)
```

---

### 🥈 **EXERCICE 2 : Préprocessing d'Images** (★★☆ Intermédiaire)
**Fichier :** `agent_workshop.py` → fonction `preprocess_state()`

**Mission :** Préparer les images pour l'entrée du CNN.

**Concepts clés :**
- **Normalisation** : 0-255 → 0-1
- **Validation des formes** : (84, 84, 4)
- **Types de données** : float32 pour TensorFlow

---

### 🥉 **EXERCICE 3 : Epsilon-Greedy pour Images** (★★☆ Intermédiaire)
**Fichier :** `agent_workshop.py` → fonction `act()`

**Mission :** Adapter la stratégie d'exploration pour les images.

**Nouveautés :**
- **Dimensions** : Gestion des images 4D
- **Preprocessing** : Intégration dans la pipeline
- **Prédiction** : CNN au lieu de réseau dense

---

### 🏆 **EXERCICE 4 : Entraînement CNN** (★★★ Avancé)
**Fichier :** `agent_workshop.py` → fonction `train()`

**Mission :** Optimiser un CNN avec l'expérience replay.

**Défis spécifiques :**
- **Mémoire** : Images vs vecteurs
- **Batch processing** : Gestion des dimensions
- **Target Networks** : Stabilisation de l'entraînement

## 🚀 Pré-requis Techniques

### **Bibliothèques Nécessaires**
```bash
pip install tensorflow opencv-python numpy pygame matplotlib
```

### **Connaissances Recommandées**
- **Bases CNN** : Convolution, pooling, feature maps
- **TensorFlow/Keras** : Couches, optimiseurs, gradient tape
- **Traitement d'images** : Redimensionnement, normalisation
- **Apprentissage par renforcement** : Q-Learning, epsilon-greedy

## 🎮 Différences avec le Jeu Basique

### **Optimisations Avancées**

#### **Couleurs Optimisées pour CNN**
- **Arrière-plan** : Noir (0) → Valeur grayscale minimale
- **Nourriture** : Rouge (255,0,0) → ~76 en grayscale
- **Corps serpent** : Gris (180,180,180) → ~180 en grayscale  
- **Tête serpent** : Blanc (255,255,255) → 255 en grayscale

#### **Frame Stacking**
- **4 images consécutives** empilées
- **Perception du mouvement** comme un humain
- **Historique temporel** pour meilleures décisions

#### **Preprocessing Avancé**
- **Capture d'écran** sans texte
- **Redimensionnement** 336x336 → 84x84
- **Conversion grayscale** optimisée
- **Normalisation** 0-255 → 0-1

## 💡 Conseils pour Réussir

### **🔍 Débogage CNN**
```python
# Vérifier les formes d'entrée
print(f"State shape: {state.shape}")  # Doit être (84, 84, 4)

# Vérifier les prédictions
q_values = model.predict(state_batch)
print(f"Q-values: {q_values}")  # 3 valeurs par action
```

### **📊 Surveillance de l'entraînement**
- **Perte** : Doit diminuer progressivement
- **Epsilon** : Décroît de 1.0 à 0.01
- **Score moyen** : Amélioration lente mais constante
- **Temps** : Patience ! CNN = plus lent que dense

### **⚡ Optimisations Performance**
- **Batch size** : 32 (équilibre mémoire/vitesse)
- **Target update** : Tous les 100 steps
- **Memory size** : 1000 expériences max
- **Frame rate** : 200 FPS pour entraînement rapide

## 🧪 Expérimentations Possibles

### **Variations d'Architecture**
- **Plus de couches** : Conv2D supplémentaires
- **Filtres différents** : 64, 128 au lieu de 32
- **Kernel sizes** : 5x5, 7x7 au lieu de 3x3
- **Dropout** : Régularisation pour éviter l'overfitting

### **Hyperparamètres**
- **Learning rate** : 0.001, 0.0001
- **Gamma** : 0.95, 0.99
- **Epsilon decay** : Plus rapide ou plus lent
- **Batch size** : 16, 64, 128

## 🔬 Analyse des Résultats

### **Métriques à Observer**
- **Score maximum** : Record absolu atteint
- **Score moyen** : Tendance générale
- **Perte** : Convergence de l'apprentissage
- **Epsilon** : Évolution exploration→exploitation

### **Comportements Émergents**
- **Premiers épisodes** : Mouvement aléatoire
- **50-100 épisodes** : Évite les murs
- **200-300 épisodes** : Cherche la nourriture
- **500+ épisodes** : Stratégies optimales

## ⚠️ Résolution de Problèmes

### **Erreurs Communes**
```python
# Forme incorrecte
ValueError: Input shape mismatch → Vérifiez (84, 84, 4)

# Mémoire insuffisante
OOM error → Réduisez batch_size ou memory_size

# Pas d'amélioration
→ Vérifiez epsilon decay et learning rate
```

### **Signaux d'Alerte**
- **Perte qui explose** : Learning rate trop élevé
- **Score qui stagne** : Epsilon trop bas trop tôt
- **Entraînement trop lent** : CPU au lieu de GPU

## 🏁 Résultats Attendus

### **Après 500 Épisodes**
- **Score moyen** : 5-10 points
- **Score maximum** : 15-25 points
- **Comportement** : Navigation fluide, recherche active de nourriture

### **Comparaison avec Humain**
- **Humain novice** : ~10-15 points
- **Agent entraîné** : ~15-25 points
- **Avantage agent** : Réflexes parfaits, pas de fatigue

## 🔮 Extensions Possibles

### **Algorithmes Avancés**
- **Double DQN** : Réduction du biais d'estimation
- **Dueling DQN** : Séparation valeur/avantage
- **Rainbow DQN** : Combinaison de toutes les améliorations

### **Autres Jeux**
- **Atari** : Breakout, Pong, Space Invaders
- **Jeux personnalisés** : Adaptation de l'architecture
- **Environnements 3D** : Minecraft, Unity

---

**🎉 Bon Workshop et Bonne Découverte de la Vision Artificielle !**

*Rappelez-vous : Les CNNs révolutionnent l'IA en permettant aux machines de "voir" le monde comme nous !* 🤖👁️ 