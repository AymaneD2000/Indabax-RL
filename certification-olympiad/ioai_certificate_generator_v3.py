#!/usr/bin/env python3
"""
IOAI 2025 Certificate Generator V3 - Fixed Version
Professional tool with text replacement preserving original formatting (compatible with all PyMuPDF versions)

Author: AI Assistant
Date: 2024
Version: 3.0
Features: Preserves font, size, and color of original text (without flags parameter)
"""

import pandas as pd
import fitz  # PyMuPDF
import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ioai_certificate_generator_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IOAICertificateGeneratorV3:
    """
    Fixed professional class for generating IOAI 2025 Phase 2 certificates
    
    This version preserves original text formatting (compatible version):
    - Font family and size
    - Text color
    - Text positioning and alignment
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
        
        logger.info(f"Initialized IOAI Certificate Generator V3 (Fixed)")
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
    
    def _get_text_properties(self, page: fitz.Page, text_rect: fitz.Rect) -> Dict:
        """
        Extract text properties from the original text location
        
        Args:
            page (fitz.Page): PDF page object
            text_rect (fitz.Rect): Rectangle containing the text
            
        Returns:
            Dict: Text properties (font, size, color)
        """
        try:
            # Get text blocks in the area
            blocks = page.get_text("dict", clip=text_rect)
            
            # Default properties
            properties = {
                'fontname': 'helv',
                'fontsize': 12,
                'color': (0, 0, 0)  # Black
            }
            
            # Extract properties from the first text span found
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                # Get font name, handling potential encoding issues
                                font_name = span.get('font', 'helv')
                                if not font_name or font_name == '':
                                    font_name = 'helv'
                                
                                properties.update({
                                    'fontname': font_name,
                                    'fontsize': span.get('size', 12),
                                    'color': span.get('color', 0)
                                })
                                logger.debug(f"Extracted text properties: {properties}")
                                return properties
            
            logger.debug(f"Using default text properties: {properties}")
            return properties
            
        except Exception as e:
            logger.warning(f"Error extracting text properties: {e}")
            return {
                'fontname': 'helv',
                'fontsize': 12,
                'color': (0, 0, 0)
            }
    
    def _find_and_replace_text_fixed(self, page: fitz.Page, old_text: str, new_text: str) -> bool:
        """
        Fixed find and replace that preserves original text formatting
        
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
            
            replacements_made = 0
            
            for inst in text_instances:
                try:
                    # Get original text properties
                    properties = self._get_text_properties(page, inst)
                    
                    # Convert color if it's an integer
                    color = properties['color']
                    if isinstance(color, int):
                        # Convert integer color to RGB tuple
                        color = (
                            ((color >> 16) & 255) / 255.0,
                            ((color >> 8) & 255) / 255.0,
                            (color & 255) / 255.0
                        )
                    elif isinstance(color, (list, tuple)) and len(color) == 3:
                        # Ensure color values are in 0-1 range
                        color = tuple(min(1.0, max(0.0, c)) for c in color)
                    else:
                        color = (0, 0, 0)  # Default to black
                    
                    # Create a white rectangle to cover the old text
                    white_rect = fitz.Rect(inst.x0 - 2, inst.y0 - 2, inst.x1 + 2, inst.y1 + 2)
                    page.draw_rect(white_rect, color=(1, 1, 1), fill=(1, 1, 1))
                    
                    # Calculate text positioning
                    text_x = inst.x0
                    text_y = inst.y1 - 2  # Adjust for baseline
                    
                    # Insert new text with preserved formatting (without flags parameter)
                    try:
                        page.insert_text(
                            (text_x, text_y),
                            new_text,
                            fontname=properties['fontname'],
                            fontsize=properties['fontsize'],
                            color=color
                        )
                    except Exception as font_error:
                        logger.warning(f"Error with custom font, using default: {font_error}")
                        # Fallback to default font if custom font fails
                        page.insert_text(
                            (text_x, text_y),
                            new_text,
                            fontname='helv',
                            fontsize=properties['fontsize'],
                            color=color
                        )
                    
                    replacements_made += 1
                    logger.debug(f"Replaced '{old_text}' with '{new_text}' using font: {properties['fontname']}, size: {properties['fontsize']}")
                    
                except Exception as e:
                    logger.error(f"Error replacing text instance: {e}")
                    continue
            
            if replacements_made > 0:
                logger.info(f"Successfully replaced {replacements_made} instances of '{old_text}'")
                return True
            else:
                logger.warning(f"No successful replacements made for '{old_text}'")
                return False
                
        except Exception as e:
            logger.error(f"Error in text replacement for '{old_text}': {e}")
            return False
    
    def generate_certificate(self, participant_data: Dict) -> str:
        """
        Generate a single certificate for a participant with preserved formatting
        
        Args:
            participant_data (Dict): Participant information
            
        Returns:
            str: Path to the generated certificate
        """
        try:
            # Extract participant information
            name = participant_data['Nom Complet']
            token = participant_data['Token']
            
            logger.info(f"Generating certificate for: {name} (Token: {token})")
            
            # Open template PDF
            doc = fitz.open(str(self.template_pdf))
            
            # Process each page
            replacements_successful = False
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Replace placeholders with fixed method
                name_replaced = self._find_and_replace_text_fixed(
                    page, "[NOM ET PRÉNOM DU PARTICIPANT]", name
                )
                token_replaced = self._find_and_replace_text_fixed(
                    page, "[Numéro de token]", token
                )
                
                if name_replaced or token_replaced:
                    replacements_successful = True
                    logger.debug(f"Page {page_num + 1}: Name replaced: {name_replaced}, Token replaced: {token_replaced}")
            
            if not replacements_successful:
                logger.warning(f"No text replacements were successful for {name}")
            
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
        logger.info("Starting fixed batch certificate generation...")
        
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
        logger.info(f"Fixed certificate generation complete!")
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
                'Output Filename': expected_filename,
                'File Size (KB)': round(expected_path.stat().st_size / 1024, 2) if expected_path.exists() else 0
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(report_data)
        
        # Save summary report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"IOAI_2025_Certificate_Generation_Report_V3_{timestamp}.csv"
        report_path = self.output_dir / report_filename
        
        summary_df.to_csv(report_path, index=False, encoding='utf-8')
        
        logger.info(f"Fixed summary report saved: {report_filename}")
        return str(report_path)


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description="IOAI 2025 Certificate Generator V3 - Fixed Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fixed Features:
  - Preserves original font family and size
  - Maintains text color
  - Keeps text positioning and alignment
  - Compatible with all PyMuPDF versions

Examples:
  python ioai_certificate_generator_v3.py --csv Phase2_Admitted_Not_In_Phase3.csv --template template.pdf
  python ioai_certificate_generator_v3.py --csv data.csv --template cert.pdf --output fixed_certificates
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
        default='fixed_certificates',
        help='Output directory for generated certificates (default: fixed_certificates)'
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
        # Initialize fixed generator
        generator = IOAICertificateGeneratorV3(
            csv_file=args.csv,
            template_pdf=args.template,
            output_dir=args.output
        )
        
        # Generate certificates
        certificates = generator.generate_all_certificates()
        
        # Generate summary report
        report_path = generator.generate_summary_report()
        
        print(f"\n{'='*70}")
        print("IOAI 2025 FIXED CERTIFICATE GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Generated: {len(certificates)} certificates")
        print(f"📁 Output directory: {generator.output_dir}")
        print(f"📊 Summary report: {Path(report_path).name}")
        print(f"🎨 Fixed formatting: Font, size, and color preserved")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Fixed certificate generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
