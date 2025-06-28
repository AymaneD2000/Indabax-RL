# IOAI 2025 Participants List Generator

Ce script génère automatiquement une liste PDF complète de tous les participants de l'IOAI 2025 à partir des fichiers CSV d'attestation.

## 📋 Fonctionnalités

- **Consolidation des données** : Lit automatiquement tous les fichiers CSV des différentes phases
- **Suppression des doublons** : Élimine les participants dupliqués basés sur leur token unique
- **Format professionnel** : Génère un PDF avec un design professionnel et une mise en page claire
- **Confidentialité** : Exclut les numéros de téléphone pour respecter la confidentialité
- **Rapport de synthèse** : Génère des statistiques sur les participants par ville et établissement

## 📁 Fichiers traités

Le script lit automatiquement les fichiers suivants :
- `Atttestation  phase3.csv`
- `Atttestation  phase4.csv`
- `Phase2_Admitted_Not_In_Phase3.csv`

## 🚀 Installation et utilisation

### Prérequis
```bash
pip install pandas reportlab
```

### Exécution
```bash
python participants_list_generator.py
```

## 📊 Contenu du PDF généré

Le PDF contient les informations suivantes pour chaque participant :
- **Numéro** : Index dans la liste
- **Token** : Identifiant unique du participant
- **Nom Complet** : Nom et prénom du participant
- **Ville** : Ville de résidence
- **Établissement** : Dernier établissement scolaire
- **Phase** : Phase de participation (Phase 2, Phase 3, ou Phase 4)

**Note** : Les participants des Phase 3 et Phase 4 ont reçu des certificats, contrairement à ceux de la Phase 2.

## 📈 Statistiques générées

Le script génère automatiquement :
- Nombre total de participants uniques
- Nombre de doublons supprimés
- Distribution par phase de participation (avec indication des certificats)
- Top 5 des villes avec le plus de participants
- Top 5 des établissements avec le plus de participants

## 📄 Fichiers de sortie

- **PDF** : `IOAI_2025_Participants_List_YYYYMMDD_HHMMSS.pdf`
- **Log** : `participants_list_generation_YYYYMMDD_HHMMSS.log`

## 📊 Dernière exécution

Lors de la dernière exécution :
- **103 entrées** trouvées au total
- **11 doublons** supprimés
- **92 participants uniques** dans la liste finale
- PDF généré avec succès

### Répartition par ville (Top 5)
1. Bamako : 79 participants
2. bamako : 2 participants
3. Moribabougou : 2 participants
4. Kati : 2 participants
5. BAMAKO : 2 participants

### Répartition par établissement (Top 5)
1. Lycée technique de Bamako : 8 participants
2. Lycée Technique de Bamako : 5 participants
3. Lycée Planète Enfants : 4 participants
4. Lycée El Hadj Karim Traoré : 3 participants
5. Lycée planète enfants : 2 participants

## 🔒 Confidentialité

Ce script respecte la confidentialité des participants en :
- Excluant automatiquement les numéros de téléphone
- Ne conservant que les informations publiques essentielles
- Générant des logs sécurisés sans données sensibles

## ⚠️ Notes importantes

- Le script nettoie automatiquement les données (espaces, caractères spéciaux)
- Les participants sont triés par ordre alphabétique du nom complet
- Seuls les participants avec un token et un nom valides sont inclus
- Les doublons sont identifiés uniquement par le token
- **Optimisation PDF** : Les noms d'établissements longs sont automatiquement abrégés pour un meilleur affichage
- **Mise en page adaptative** : Colonnes redimensionnées et police ajustée pour optimiser l'espace

## 🔧 Fonctionnalités d'optimisation

### Abréviation automatique des établissements
Le script applique des abréviations courantes pour les noms d'établissements :
- `Lycée` → `Lyc.`
- `École` → `Éc.`
- `Institut` → `Inst.`
- `Université` → `Univ.`
- `Technique` → `Tech.`
- `Privé/Privée` → `Priv.`
- `Bamako` → `Bko`
- Et plus encore...

### Troncature intelligente
- Limite à 45 caractères maximum
- Ajoute "..." si nécessaire
- Préserve la lisibilité 