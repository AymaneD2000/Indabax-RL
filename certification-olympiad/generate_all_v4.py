#!/usr/bin/env python3
"""
Script final pour générer TOUS les certificats IOAI 2025 Phase 2
Version 4 avec centrage des noms et alignement des tokens en rouge
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.append(str(Path(__file__).parent))

from ioai_certificate_generator_v4 import IOAICertificateGeneratorV2

def setup_logging():
    """Configure logging pour la génération complète"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"generation_complete_v4_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def main():
    """Génération complète de tous les certificats"""
    
    print("=" * 60)
    print("🎓 GÉNÉRATION COMPLÈTE DES CERTIFICATS IOAI 2025 PHASE 3")
    print("📍 Version 4 - Noms centrés & Tokens alignés en rouge")
    print("=" * 60)
    print()
    
    log_file = setup_logging()
    
    # Configuration
    csv_file = "Atttestation  phase4.csv"
    template_pdf = "INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025 - PHASE 4.pdf"
    output_dir = "final_certificates_v4_phase4"
    
    try:
        # Créer le générateur
        print("🔧 Initialisation du générateur...")
        generator = IOAICertificateGeneratorV2(
            csv_file=csv_file,
            template_pdf=template_pdf,
            output_dir=output_dir
        )
        
        print(f"✅ Générateur initialisé avec succès")
        print(f"   📄 CSV: {csv_file}")
        print(f"   📋 Template: {template_pdf}")
        print(f"   📁 Output: {output_dir}")
        print(f"   👥 Participants: {len(generator.participants_df)}")
        print(f"   📝 Log: {log_file}")
        print()
        
        # Génération de tous les certificats
        print("🚀 DÉBUT DE LA GÉNÉRATION...")
        print("   • Noms seront centrés dans leur zone")
        print("   • Tokens seront alignés avec 'Identifiant (Token):'")
        print("   • Couleur rouge pour noms et tokens")
        print()
        
        start_time = datetime.now()
        
        # Générer tous les certificats
        generated_certificates = generator.generate_all_certificates()
        
        # Générer le rapport de synthèse
        report_path = generator.generate_summary_report()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print()
        print("=" * 60)
        print("🎉 GÉNÉRATION TERMINÉE !")
        print("=" * 60)
        print(f"✅ Certificats générés: {len(generated_certificates)}")
        print(f"📊 Rapport de synthèse: {Path(report_path).name}")
        print(f"⏱️  Durée totale: {duration}")
        print(f"📁 Dossier de sortie: {output_dir}")
        print(f"📝 Log détaillé: {log_file}")
        print()
        
        if len(generated_certificates) == len(generator.participants_df):
            print("🎯 SUCCÈS COMPLET - Tous les certificats ont été générés !")
        else:
            missing = len(generator.participants_df) - len(generated_certificates)
            print(f"⚠️  ATTENTION - {missing} certificats manquants")
        
        print()
        print("🔍 VÉRIFICATIONS À EFFECTUER :")
        print("   • Noms bien centrés dans leur zone")
        print("   • Tokens alignés avec le texte 'Identifiant (Token):'")
        print("   • Couleur rouge pour noms et tokens")
        print("   • Polices et tailles préservées")
        print()
        
        return len(generated_certificates) == len(generator.participants_df)
        
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        logging.error(f"Erreur critique lors de la génération: {e}")
        return False

if __name__ == "__main__":
    success = main()
    print("=" * 60)
    if success:
        print("🎊 MISSION ACCOMPLIE - Génération réussie !")
    else:
        print("💥 ÉCHEC - Vérifiez les logs pour plus de détails")
    print("=" * 60)
    sys.exit(0 if success else 1) 