# Importation des bibliothèques nécessaires
import pygame  # Pygame est utilisé pour gérer l'interface graphique du jeu.
import random  # Random est utilisé pour générer des positions aléatoires pour la nourriture.
from enum import Enum  # Enum permet de créer des énumérations, ici pour les directions.
from collections import namedtuple  # Namedtuple est utilisé pour créer des points avec des coordonnées x et y.
import numpy as np  # NumPy est utilisé pour manipuler les actions du serpent.
import cv2  # OpenCV pour le traitement d'images

# Initialisation de Pygame et de la police d'écriture
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Définition de la classe Direction pour les quatre directions possibles du serpent
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Définition d'une structure pour représenter un point (x, y)
Point = namedtuple('Point', 'x, y')

# Définition des couleurs RGB optimisées pour l'apprentissage CNN
WHITE = (255, 255, 255)  # Couleur blanche, utilisée pour le texte
BLACK = (0, 0, 0)        # Arrière-plan - Valeur grayscale: 0

# Couleurs optimisées pour la nourriture (valeurs distinctes en grayscale)
FOOD_COLOR = (255, 0, 0)      # Rouge vif - Valeur grayscale: ~76
FOOD_BORDER = (180, 0, 0)     # Rouge foncé pour les bordures

# Couleurs optimisées pour le serpent (contrastes maximums en grayscale)
SNAKE_HEAD = (255, 255, 255)        # Blanc pur - Valeur grayscale: 255 (maximum)
SNAKE_HEAD_BORDER = (200, 200, 200) # Gris clair pour la bordure
SNAKE_BODY = (180, 180, 180)        # Gris clair - Valeur grayscale: ~180
SNAKE_BODY_CORE = (120, 120, 120)   # Gris moyen - Valeur grayscale: ~120

# Alternative: Couleurs vertes naturelles (commentées - à utiliser si préféré)
# SNAKE_HEAD = (50, 255, 50)         # Vert vif - Valeur grayscale: ~200
# SNAKE_HEAD_BORDER = (30, 200, 30)  # Vert foncé
# SNAKE_BODY = (100, 200, 100)       # Vert moyen - Valeur grayscale: ~160
# SNAKE_BODY_CORE = (60, 140, 60)    # Vert foncé - Valeur grayscale: ~120

# Taille des blocs qui composent le serpent et la nourriture
BLOCK_SIZE = 20
# Vitesse de rafraîchissement du jeu (nombre d'images par seconde)
SPEED = 200

# Définition de la classe SnakeGameAI qui gère le jeu du serpent
class SnakeGameAI2:
    
    def __init__(self, w=620, h=480, image_based=True):
        """
        Initialise les paramètres du jeu.

        :param w: Largeur de la fenêtre du jeu.
        :param h: Hauteur de la fenêtre du jeu.
        :param image_based: Si True, active le mode basé sur les images pour l'IA.
        """
        self.w = w
        self.h = h
        self.reward = 0
        self.image_based = image_based
        
        # Configuration de l'affichage
        self.display = pygame.display.set_mode((self.w, self.h))  # Création de la fenêtre du jeu
        pygame.display.set_caption('Snake AI - Optimized Colors for CNN Learning')  # Titre de la fenêtre
        self.clock = pygame.time.Clock()  # Horloge pour gérer le taux de rafraîchissement
        
        # Configuration pour la capture d'images
        if self.image_based:
            self.frame_stack_size = 4  # Nombre d'images consécutives à empiler
            self.processed_frame_size = (84, 84)  # Taille des images traitées
            self.frame_buffer = []  # Buffer pour les images consécutives
            
        self.reset()  # Initialisation du jeu

    def reset(self):
        """
        Réinitialise le jeu à son état initial.
        """
        self.direction = Direction.RIGHT  # Direction initiale du serpent (droite)
        
        # Ensure snake head starts on grid-aligned position
        center_x = (self.w // BLOCK_SIZE // 2) * BLOCK_SIZE
        center_y = (self.h // BLOCK_SIZE // 2) * BLOCK_SIZE
        self.head = Point(center_x, center_y)  # Position initiale de la tête du serpent (centre de la fenêtre, grid-aligned)
        
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]  # Corps initial du serpent (trois blocs)

        self.score = 0  # Score initial
        self.reward = 0  # Récompense initiale
        self.food = None  # Position de la nourriture (sera définie par _place_food)
        self._place_food()  # Place la nourriture à une position aléatoire
        self.frame_iteration = 0  # Compteur de frames (utilisé pour limiter le nombre d'actions possibles)
        
        # Réinitialisation du buffer d'images pour l'IA basée sur les images
        if self.image_based:
            self.frame_buffer = []
            # Remplissage initial du buffer avec la première image
            initial_frame = self._get_processed_frame()
            for _ in range(self.frame_stack_size):
                self.frame_buffer.append(initial_frame)
        state = self.get_state_image()
        reward = 0
        done = False
        return state, reward, done, self.score

    def _place_food(self):
        """
        Place la nourriture à un endroit aléatoire sur la grille.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # Coordonnée x aléatoire
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # Coordonnée y aléatoire
        self.food = Point(x, y)  # Position de la nourriture
        if self.food in self.snake:  # Si la nourriture est placée sur le serpent, on la replace
            print("Food is on the snake, replacing...")
            self._place_food()
    
    def _get_raw_frame(self):
        """
        Capture l'image actuelle du jeu sans le texte du score.
        
        Returns:
            Image numpy array (H, W, 3) en RGB
        """
        # Créer une surface temporaire sans le texte
        temp_surface = pygame.Surface((self.w, self.h))
        temp_surface.fill(BLACK)
        
        # Dessiner le corps du serpent (sauf la tête)
        for i, pt in enumerate(self.snake):
            if i == 0:  # Tête du serpent - couleur spéciale
                pygame.draw.rect(temp_surface, SNAKE_HEAD, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(temp_surface, SNAKE_HEAD_BORDER, pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
            else:  # Corps du serpent
                pygame.draw.rect(temp_surface, SNAKE_BODY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(temp_surface, SNAKE_BODY_CORE, pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6))
        
        # Dessiner la nourriture avec un design amélioré
        pygame.draw.rect(temp_surface, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(temp_surface, FOOD_BORDER, pygame.Rect(self.food.x + 2, self.food.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
        
        # Convertir la surface pygame en array numpy
        frame = pygame.surfarray.array3d(temp_surface)
        frame = np.transpose(frame, (1, 0, 2))  # Correction de l'orientation
        
        return frame
    
    def _get_processed_frame(self):
        """
        Traite l'image brute pour l'IA : redimensionnement et conversion en niveaux de gris.
        
        Returns:
            Image traitée (84, 84) en niveaux de gris avec valeurs optimisées
        """
        # Capturer l'image brute
        raw_frame = self._get_raw_frame()
        
        # Conversion en niveaux de gris optimisée pour les nouvelles couleurs
        # Poids ajustés pour maximiser les contrastes entre les éléments
        gray_frame = np.dot(raw_frame[...,:3], [0.299, 0.587, 0.114])
        
        # Redimensionnement à 84x84 (taille standard pour les DQN)
        processed_frame = cv2.resize(gray_frame, self.processed_frame_size, interpolation=cv2.INTER_AREA)
        
        # Normalisation et amélioration du contraste
        processed_frame = processed_frame.astype(np.uint8)
        
        # Optionnel: Amélioration du contraste (CLAHE)
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # processed_frame = clahe.apply(processed_frame)
        
        return processed_frame
    
    def get_state_image(self):
        """
        Retourne l'état actuel du jeu sous forme d'image empilée.
        
        Returns:
            Stack de frames (84, 84, 4) pour l'entrée du CNN
        """
        if not self.image_based:
            raise ValueError("Mode image non activé. Utilisez image_based=True lors de l'initialisation.")
        
        # Ajouter la nouvelle frame au buffer
        current_frame = self._get_processed_frame()
        self.frame_buffer.append(current_frame)
        
        # Maintenir la taille du buffer
        if len(self.frame_buffer) > self.frame_stack_size:
            self.frame_buffer.pop(0)
        
        # Empiler les frames le long de la dimension des canaux
        stacked_frames = np.stack(self.frame_buffer, axis=-1)
        
        return stacked_frames
    
    def get_color_analysis(self):
        """
        Analyse les valeurs de grayscale pour vérifier l'optimisation des couleurs.
        
        Returns:
            Dict avec les valeurs de grayscale moyennes pour chaque élément
        """
        if not self.image_based:
            return None
            
        frame = self._get_processed_frame()
        
        # Analyse des régions pour vérifier les contrastes
        analysis = {
            "background_avg": np.mean(frame[frame < 50]),  # Pixels sombres (arrière-plan)
            "food_avg": np.mean(frame[(frame > 60) & (frame < 100)]),  # Pixels moyens (nourriture)
            "snake_body_avg": np.mean(frame[(frame > 100) & (frame < 200)]),  # Pixels clairs (corps)
            "snake_head_avg": np.mean(frame[frame > 200]),  # Pixels très clairs (tête)
            "contrast_ratio": np.std(frame),  # Écart-type comme mesure de contraste
            "unique_values": len(np.unique(frame))  # Nombre de valeurs uniques
        }
        
        return analysis
    
    def play_step(self, action):
        """
        Avance le jeu d'une étape en fonction de l'action choisie par l'agent.

        :param action: Action choisie par l'agent (avancer, tourner à droite, tourner à gauche).
        :return: Tuple contenant la récompense, l'état de fin de partie (True/False), et le score actuel.
        """
        self.frame_iteration += 1  # Incrémente le compteur de frames
        
        # 1. Collecte des événements (comme la fermeture de la fenêtre)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Debug: Print positions before movement
        old_head = self.head
        distance_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # 2. Déplacement du serpent
        self._move(action)  # Mise à jour de la position de la tête du serpent
        self.snake.insert(0, self.head)  # Ajout de la nouvelle position de la tête au début de la liste représentant le corps du serpent

        # Debug: Print positions after movement
        distance_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Debug when very close to food
        if distance_after <= BLOCK_SIZE * 2:  # Within 2 blocks
            print(f"🔍 CLOSE TO FOOD:")
            print(f"   Old head: ({old_head.x}, {old_head.y})")
            print(f"   New head: ({self.head.x}, {self.head.y})")
            print(f"   Food pos: ({self.food.x}, {self.food.y})")
            print(f"   Distance before: {distance_before}, after: {distance_after}")
            print(f"   Head == Food? {self.head == self.food}")
            print(f"   Head.x == Food.x? {self.head.x == self.food.x}")
            print(f"   Head.y == Food.y? {self.head.y == self.food.y}")

        # 3. Vérification des collisions et calcul des récompenses
        reward = self._calculate_reward()
        game_over = False
        
        # Vérification de collision ou timeout
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True  # Fin du jeu en cas de collision ou de dépassement du nombre maximum de frames
            reward = -100  # Récompense négative importante pour une collision
            state = self.get_state_image()
            return state, reward, game_over, self.score

        # 4. Vérification si le serpent a mangé la nourriture
        if self.head == self.food:
            print("🍎 Food is eaten!")
            print(f"   Head position: ({self.head.x}, {self.head.y})")
            print(f"   Food position: ({self.food.x}, {self.food.y})")
            self.score += 1  # Incrémente le score
            reward = 150  # Récompense positive pour avoir mangé la nourriture
            self._place_food()  # Place une nouvelle nourriture
            print(f"   New food position: ({self.food.x}, {self.food.y})")
        else:
            self.snake.pop()  # Supprime le dernier élément du corps du serpent (le serpent avance)

        # 5. Mise à jour de l'interface graphique et de l'horloge
        self._update_ui()
        self.clock.tick(SPEED)  # Contrôle de la vitesse du jeu
        
        state = self.get_state_image()
        return state, reward, game_over, self.score
    
    def _calculate_reward(self):
        """
        Calcule une récompense plus sophistiquée basée sur la distance à la nourriture.
        
        Returns:
            Récompense calculée
        """
        # Récompense de base pour rester en vie
        reward = 0.0
        
        # Distance à la nourriture (récompense basée sur la proximité)
        head_x, head_y = self.head.x, self.head.y
        food_x, food_y = self.food.x, self.food.y
        
        # Distance Manhattan normalisée
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        max_distance = self.w + self.h
        normalized_distance = distance / max_distance
        
        # Récompense inversement proportionnelle à la distance
        reward += (normalized_distance - 2) 
        
        return reward

    def is_collision(self, pt=None):
        """
        Vérifie si le serpent entre en collision avec les murs ou avec lui-même.

        :param pt: Point à vérifier (par défaut la tête du serpent).
        :return: True si le serpent est en collision, sinon False.
        """
        if pt is None:
            pt = self.head  # Si aucun point n'est spécifié, on vérifie la tête du serpent
        # Vérifie si la tête touche les limites de la fenêtre
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Vérifie si la tête touche une partie du corps du serpent
        if pt in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        """
        Met à jour l'interface graphique avec les nouvelles positions du serpent et de la nourriture.
        """
        self.display.fill(BLACK)  # Remplit l'écran de noir

        # Dessine chaque segment du serpent avec les nouvelles couleurs
        for i, pt in enumerate(self.snake):
            if i == 0:  # Tête du serpent - design spécial
                pygame.draw.rect(self.display, SNAKE_HEAD, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, SNAKE_HEAD_BORDER, pygame.Rect(pt.x + 2, pt.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
            else:  # Corps du serpent
                pygame.draw.rect(self.display, SNAKE_BODY, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, SNAKE_BODY_CORE, pygame.Rect(pt.x + 3, pt.y + 3, BLOCK_SIZE - 6, BLOCK_SIZE - 6))

        # Dessine la nourriture avec le nouveau design
        pygame.draw.rect(self.display, FOOD_COLOR, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, FOOD_BORDER, pygame.Rect(self.food.x + 2, self.food.y + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))

        # Affiche le score en haut à gauche de l'écran
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()  # Rafraîchit l'écran

    def _move(self, action):
        """
        Déplace le serpent dans la direction choisie par l'agent.

        :param action: Tableau indiquant l'action choisie [avancer, tourner à droite, tourner à gauche].
        """
        # Liste des directions dans le sens des aiguilles d'une montre
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)  # Indice de la direction actuelle

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Pas de changement de direction
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Tourne à droite
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Tourne à gauche
            new_dir = clock_wise[next_idx]

        self.direction = new_dir  # Met à jour la direction du serpent

        # Met à jour la position de la tête du serpent en fonction de la direction choisie
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)  # Met à jour la position de la tête
