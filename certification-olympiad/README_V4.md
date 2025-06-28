# 🎓 Générateur de Certificats IOAI 2025 Phase 2 - Version 4

## 🎯 Nouveautés Version 4

Cette version apporte les **améliorations de formatage professionnel** demandées :

### ✨ Fonctionnalités Avancées

- **🎯 Centrage automatique des noms** : Les noms sont parfaitement centrés dans leur zone d'affichage
- **📍 Alignement précis des tokens** : Les tokens sont alignés avec le texte "Identifiant (Token):"
- **🔴 Couleur rouge uniforme** : Noms et tokens s'affichent en rouge (couleur originale du template)
- **📏 Préservation des propriétés** : Police, taille et style du texte original maintenus

### 🔧 Améliorations Techniques

1. **Analyse avancée du PDF** : Utilise `get_text("dict")` pour une analyse précise des spans
2. **Calculs de positionnement** : 
   - Noms : Centrage dynamique basé sur la largeur estimée du texte
   - Tokens : Position fixe x=288.2 pour alignement parfait
3. **Gestion des couleurs** : Conversion automatique des couleurs en format RGB (0.7, 0.0, 0.0)
4. **Polices de substitution** : Fallback intelligent (ArchivoBlack → helvetica-bold, Anaktoria → helvetica)

## 📋 Utilisation

### 🚀 Génération Rapide (Test)

```bash
# Test avec 3 certificats
python3 test_v4.py
```

### 🎊 Génération Complète

```bash
# Génération de tous les certificats
python3 generate_all_v4.py
```

## 📊 Résultats Attendus

### 📍 Positionnement
- **Noms** : Centrés dans la zone (94.8, 268.0) → (752.2, 303.7)
- **Tokens** : Alignés à x=288.2 avec "Identifiant (Token):"

### 🎨 Formatage
- **Couleur** : Rouge (0.7, 0.0, 0.0) pour noms ET tokens
- **Police noms** : helvetica-bold, 32.8pt (substitution d'ArchivoBlack-Regular)
- **Police tokens** : helvetica, 24pt (substitution d'Anaktoria)

### 📁 Structure de Sortie

```
final_certificates_v4/
├── IOAI_2025_Certificate_Nom_Participant_token.pdf
├── IOAI_2025_Certificate_Generation_Report_V2_YYYYMMDD_HHMMSS.csv
└── ...
```

## 🔍 Logs Détaillés

Les logs montrent pour chaque certificat :

```
Remplacement trouvé: '[NOM ET PRÉNOM DU PARTICIPANT]' -> 'Nom Participant'
  Position originale: (94.8, 268.0, 752.2, 303.7)
  Police originale: ArchivoBlack-Regular
  Taille originale: 32.808834075927734
  Couleur originale: 11741488
  NOM - Position centrée: x=XXX.X, y=294.8
  NOM - Largeur estimée du texte: XXX.X
  NOM - Point central: 423.5
  Police de substitution: ArchivoBlack-Regular -> helvetica-bold
  ✓ Texte inséré avec succès
    Police utilisée: helvetica-bold
    Taille: 32.808834075927734
    Couleur finale: (0.7, 0.0, 0.0)
    Position finale: (XXX.X, 294.8)

Remplacement trouvé: 'Numéro de token' -> 'olp-xxxxx-xxxxx'
  Position originale: (288.2, 312.9, 439.8, 343.5)
  Police originale: Anaktoria
  Taille originale: 24.001543045043945
  Couleur originale: 11741488
  TOKEN - Position alignée: x=288.2, y=335.8
  Police de substitution: Anaktoria -> helvetica
  ✓ Texte inséré avec succès
    Police utilisée: helvetica
    Taille: 24.001543045043945
    Couleur finale: (0.7, 0.0, 0.0)
    Position finale: (288.2, 335.8)
```

## 🛠️ Dépendances

```
pandas>=1.3.0
PyMuPDF>=1.23.0
pathlib2>=2.3.0
```

## 📝 Fichiers Principaux

- `ioai_certificate_generator_v4.py` : Générateur principal avec centrage et alignement
- `test_v4.py` : Script de test (3 certificats)
- `generate_all_v4.py` : Script de génération complète (65 certificats)
- `README_V4.md` : Cette documentation

## ✅ Validation

Pour vérifier que les certificats sont corrects :

1. **Centrage des noms** : Les noms doivent être visuellement centrés dans leur zone
2. **Alignement des tokens** : Les tokens doivent commencer exactement sous le "N" de "Numéro"
3. **Couleur rouge** : Noms et tokens en rouge, pas en noir
4. **Lisibilité** : Texte net, sans artefacts ou chevauchements

## 🎉 Résumé des Améliorations

| Aspect | Version 3 | Version 4 |
|--------|-----------|-----------|
| **Position noms** | Aligné à gauche | ✅ **Centré dynamiquement** |
| **Position tokens** | Position originale | ✅ **Aligné avec "Identifiant"** |
| **Couleur noms** | Couleur originale | ✅ **Rouge forcé** |
| **Couleur tokens** | Couleur originale | ✅ **Rouge forcé** |
| **Précision** | Approximative | ✅ **Calculs précis** |

La Version 4 répond parfaitement aux exigences de **centrage des noms** et d'**alignement des tokens en rouge** ! 🎯 