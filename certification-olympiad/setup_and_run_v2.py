#!/usr/bin/env python3
"""
Quick Setup and Run Script for IOAI 2025 Certificate Generator V2 (Enhanced)
Enhanced version that preserves original text formatting
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def run_enhanced_certificate_generator():
    """Run the enhanced certificate generator with default settings"""
    
    # Check if files exist
    csv_file = "Phase2_Admitted_Not_In_Phase3.csv"
    pdf_template = "INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025 - PHASE 2.pdf"
    
    if not Path(csv_file).exists():
        print(f"❌ CSV file not found: {csv_file}")
        print("Please ensure the CSV file is in the same directory as this script.")
        return False
    
    if not Path(pdf_template).exists():
        print(f"❌ PDF template not found: {pdf_template}")
        print("Please ensure the PDF template is in the same directory as this script.")
        return False
    
    print("🚀 Starting ENHANCED certificate generation...")
    print("🎨 This version preserves original font, size, and style!")
    try:
        # Run the enhanced certificate generator
        cmd = [
            sys.executable, 
            "ioai_certificate_generator_v2.py",
            "--csv", csv_file,
            "--template", pdf_template,
            "--output", "enhanced_certificates",
            "--verbose"
        ]
        
        subprocess.run(cmd, check=True)
        print("🎉 Enhanced certificate generation completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running enhanced certificate generator: {e}")
        return False

def main():
    """Main function"""
    print("=" * 70)
    print("IOAI 2025 ENHANCED CERTIFICATE GENERATOR - QUICK SETUP")
    print("🎨 PRESERVES ORIGINAL FORMATTING (Font, Size, Style)")
    print("=" * 70)
    
    # Step 1: Install requirements
    if not install_requirements():
        return 1
    
    print("\n" + "-" * 70)
    
    # Step 2: Run enhanced certificate generator
    if not run_enhanced_certificate_generator():
        return 1
    
    print("\n" + "=" * 70)
    print("🎉 ENHANCED GENERATION COMPLETE!")
    print("📁 Check the 'enhanced_certificates' folder for your certificates")
    print("📊 Check the CSV report for generation summary")
    print("🎨 All certificates preserve original text formatting!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())
