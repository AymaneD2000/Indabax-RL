#!/usr/bin/env python3
"""
Script de test pour IOAICertificateGeneratorV4
Test du centrage des noms et de l'alignement des tokens en rouge
"""

import logging
import sys
from pathlib import Path

# Ajouter le répertoire courant au path
sys.path.append(str(Path(__file__).parent))

from ioai_certificate_generator_v4 import IOAICertificateGeneratorV2

def setup_logging():
    """Configure logging pour les tests"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_v4.log'),
            logging.StreamHandler()
        ]
    )

def test_few_certificates():
    """Test avec quelques certificats pour vérifier le centrage et l'alignement"""
    
    print("=== TEST IOAI CERTIFICATE GENERATOR V4 ===")
    print("Test du centrage des noms et alignement des tokens en rouge")
    print()
    
    # Configuration
    csv_file = "Phase2_Admitted_Not_In_Phase3.csv"
    template_pdf = "INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025 - PHASE 2.pdf"
    output_dir = "test_certificates_v4"
    
    try:
        # Créer le générateur
        generator = IOAICertificateGeneratorV2(
            csv_file=csv_file,
            template_pdf=template_pdf,
            output_dir=output_dir
        )
        
        print(f"✓ Générateur initialisé")
        print(f"  - CSV: {csv_file}")
        print(f"  - Template: {template_pdf}")
        print(f"  - Output: {output_dir}")
        print(f"  - Participants trouvés: {len(generator.participants_df)}")
        print()
        
        # Tester avec les 3 premiers participants
        test_participants = generator.participants_df.head(3)
        
        print("=== GÉNÉRATION DE CERTIFICATS TEST ===")
        generated_certs = []
        
        for index, participant in test_participants.iterrows():
            try:
                print(f"Génération pour: {participant['Nom Complet']} (Token: {participant['Token']})")
                cert_path = generator.generate_certificate(participant.to_dict())
                generated_certs.append(cert_path)
                print(f"  ✓ Certificat généré: {Path(cert_path).name}")
                
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
        
        print()
        print("=== RÉSULTATS DU TEST ===")
        print(f"Certificats générés avec succès: {len(generated_certs)}")
        print(f"Dossier de sortie: {output_dir}")
        print()
        
        if generated_certs:
            print("✅ TEST RÉUSSI - Vérifiez les certificats pour:")
            print("   • Noms centrés dans leur zone")
            print("   • Tokens alignés avec 'Identifiant (Token):'")
            print("   • Couleur rouge pour noms et tokens")
            print()
            print("Certificats générés:")
            for cert in generated_certs:
                print(f"   - {Path(cert).name}")
        else:
            print("❌ TEST ÉCHOUÉ - Aucun certificat généré")
            
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE: {e}")
        return False
    
    return len(generated_certs) > 0

if __name__ == "__main__":
    setup_logging()
    success = test_few_certificates()
    sys.exit(0 if success else 1) 