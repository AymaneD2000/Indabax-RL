#!/usr/bin/env python3
"""
IOAI 2025 Certificate Generator
Professional tool for automating IOAI Phase 2 certificate generation

Author: AI Assistant
Date: 2024
Version: 1.0
"""

import pandas as pd
import fitz  # PyMuPDF
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ioai_certificate_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IOAICertificateGenerator:
    """
    Professional class for generating IOAI 2025 Phase 2 certificates
    
    This class handles:
    - CSV data processing
    - PDF template manipulation
    - Batch certificate generation
    - Error handling and logging
    """
    
    def __init__(self, csv_file: str, template_pdf: str, output_dir: str = "certificates"):
        """
        Initialize the certificate generator
        
        Args:
            csv_file (str): Path to the CSV file containing participant data
            template_pdf (str): Path to the PDF template
            output_dir (str): Directory to save generated certificates
        """
        self.csv_file = Path(csv_file)
        self.template_pdf = Path(template_pdf)
        self.output_dir = Path(output_dir)
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load participant data
        self.participants_df = self._load_participant_data()
        
        logger.info(f"Initialized IOAI Certificate Generator")
        logger.info(f"Template: {self.template_pdf}")
        logger.info(f"Data source: {self.csv_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Participants loaded: {len(self.participants_df)}")
    
    def _validate_inputs(self) -> None:
        """Validate input files and directories"""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        if not self.template_pdf.exists():
            raise FileNotFoundError(f"Template PDF not found: {self.template_pdf}")
        
        if not self.template_pdf.suffix.lower() == '.pdf':
            raise ValueError("Template file must be a PDF")
    
    def _load_participant_data(self) -> pd.DataFrame:
        """
        Load and validate participant data from CSV
        
        Returns:
            pd.DataFrame: Cleaned participant data
        """
        try:
            # Load CSV with explicit encoding
            df = pd.read_csv(self.csv_file, encoding='utf-8')
            
            # Validate required columns
            required_columns = ['Token', 'Nom Complet']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Clean data
            df['Nom Complet'] = df['Nom Complet'].str.strip()
            df['Token'] = df['Token'].str.strip()
            
            # Remove rows with missing essential data
            initial_count = len(df)
            df = df.dropna(subset=['Token', 'Nom Complet'])
            
            if len(df) < initial_count:
                logger.warning(f"Removed {initial_count - len(df)} rows with missing data")
            
            # Sort by ranking for better organization
            if 'Ranking' in df.columns:
                df = df.sort_values('Ranking')
            
            logger.info(f"Successfully loaded {len(df)} participant records")
            return df
            
        except Exception as e:
            logger.error(f"Error loading participant data: {e}")
            raise
    
    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize participant name for safe filename usage
        
        Args:
            name (str): Participant name
            
        Returns:
            str: Sanitized filename
        """
        # Remove special characters and spaces
        sanitized = re.sub(r'[^\w\s-]', '', name)
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        
        return sanitized.strip('_')
    
    def _find_and_replace_text(self, page: fitz.Page, old_text: str, new_text: str) -> bool:
        """
        Find and replace text in a PDF page
        
        Args:
            page (fitz.Page): PDF page object
            old_text (str): Text to be replaced
            new_text (str): Replacement text
            
        Returns:
            bool: True if replacement was successful
        """
        try:
            # Search for text instances
            text_instances = page.search_for(old_text)
            
            if not text_instances:
                logger.warning(f"Text '{old_text}' not found on page")
                return False
            
            for inst in text_instances:
                # Create redaction annotation
                page.add_redact_annot(inst, text=new_text)
            
            # Apply redactions
            page.apply_redactions()
            return True
            
        except Exception as e:
            logger.error(f"Error replacing text '{old_text}' with '{new_text}': {e}")
            return False
    
    def generate_certificate(self, participant_data: Dict) -> str:
        """
        Generate a single certificate for a participant
        
        Args:
            participant_data (Dict): Participant information
            
        Returns:
            str: Path to the generated certificate
        """
        try:
            # Extract participant information
            name = participant_data['Nom Complet']
            token = participant_data['Token']
            
            # Open template PDF
            doc = fitz.open(str(self.template_pdf))
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Replace placeholders
                name_replaced = self._find_and_replace_text(
                    page, "[NOM ET PRÉNOM DU PARTICIPANT]", name
                )
                token_replaced = self._find_and_replace_text(
                    page, "[Numéro de token]", token
                )
                
                if not (name_replaced or token_replaced):
                    logger.warning(f"No placeholders found on page {page_num + 1}")
            
            # Generate output filename
            safe_name = self._sanitize_filename(name)
            output_filename = f"IOAI_2025_Certificate_{safe_name}_{token}.pdf"
            output_path = self.output_dir / output_filename
            
            # Save the modified document
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"Generated certificate: {output_filename}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating certificate for {name}: {e}")
            raise
    
    def generate_all_certificates(self) -> List[str]:
        """
        Generate certificates for all participants
        
        Returns:
            List[str]: List of paths to generated certificates
        """
        logger.info("Starting batch certificate generation...")
        
        generated_certificates = []
        errors = []
        
        for index, participant in self.participants_df.iterrows():
            try:
                cert_path = self.generate_certificate(participant.to_dict())
                generated_certificates.append(cert_path)
                
            except Exception as e:
                error_msg = f"Failed to generate certificate for {participant['Nom Complet']}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        # Summary report
        logger.info(f"Certificate generation complete!")
        logger.info(f"Successfully generated: {len(generated_certificates)} certificates")
        logger.info(f"Errors encountered: {len(errors)}")
        
        if errors:
            logger.warning("Errors summary:")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return generated_certificates
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the certificate generation process
        
        Returns:
            str: Path to the summary report
        """
        report_data = []
        
        for index, participant in self.participants_df.iterrows():
            safe_name = self._sanitize_filename(participant['Nom Complet'])
            expected_filename = f"IOAI_2025_Certificate_{safe_name}_{participant['Token']}.pdf"
            expected_path = self.output_dir / expected_filename
            
            report_data.append({
                'Ranking': participant.get('Ranking', 'N/A'),
                'Nom Complet': participant['Nom Complet'],
                'Token': participant['Token'],
                'Certificate Generated': expected_path.exists(),
                'Output Filename': expected_filename
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(report_data)
        
        # Save summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"IOAI_2025_Certificate_Generation_Report_{timestamp}.csv"
        report_path = self.output_dir / report_filename
        
        summary_df.to_csv(report_path, index=False, encoding='utf-8')
        
        logger.info(f"Summary report saved: {report_filename}")
        return str(report_path)


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description="IOAI 2025 Certificate Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ioai_certificate_generator.py --csv Phase2_Admitted_Not_In_Phase3.csv --template template.pdf
  python ioai_certificate_generator.py --csv data.csv --template cert.pdf --output my_certificates
        """
    )
    
    parser.add_argument(
        '--csv',
        required=True,
        help='Path to CSV file containing participant data'
    )
    
    parser.add_argument(
        '--template',
        required=True,
        help='Path to PDF certificate template'
    )
    
    parser.add_argument(
        '--output',
        default='certificates',
        help='Output directory for generated certificates (default: certificates)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize generator
        generator = IOAICertificateGenerator(
            csv_file=args.csv,
            template_pdf=args.template,
            output_dir=args.output
        )
        
        # Generate certificates
        certificates = generator.generate_all_certificates()
        
        # Generate summary report
        report_path = generator.generate_summary_report()
        
        print(f"\n{'='*60}")
        print("IOAI 2025 CERTIFICATE GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"✅ Generated: {len(certificates)} certificates")
        print(f"📁 Output directory: {generator.output_dir}")
        print(f"📊 Summary report: {Path(report_path).name}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error(f"Certificate generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 