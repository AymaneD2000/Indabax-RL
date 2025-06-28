#!/usr/bin/env python3
"""
IOAI 2025 Certificate Generator V2 - Enhanced Version
Professional tool with advanced text replacement preserving original formatting

Author: AI Assistant
Date: 2024
Version: 2.0
Features: Preserves font, size, color, and style of original text
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
        logging.FileHandler('ioai_certificate_generator_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IOAICertificateGeneratorV2:
    """
    Enhanced professional class for generating IOAI 2025 Phase 2 certificates
    
    This version preserves original text formatting:
    - Font family and size
    - Text color and style
    - Text positioning and alignment
    - Character spacing
    """
    
    def __init__(self, csv_file: str, template_pdf: str, output_dir: str = "certificates"):
        """
        Initialize the enhanced certificate generator
        
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
        
        logger.info(f"Initialized IOAI Certificate Generator V2 (Enhanced)")
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
            Dict: Text properties (font, size, color, etc.)
        """
        try:
            # Get text blocks in the area
            blocks = page.get_text("dict", clip=text_rect)
            
            # Default properties
            properties = {
                'fontname': 'helv',
                'fontsize': 12,
                'color': (0, 0, 0),  # Black
                'flags': 0  # No special formatting
            }
            
            # Extract properties from the first text span found
            for block in blocks.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                properties.update({
                                    'fontname': span.get('font', 'helv'),
                                    'fontsize': span.get('size', 12),
                                    'color': span.get('color', 0),
                                    'flags': span.get('flags', 0)
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
                'color': (0, 0, 0),
                'flags': 0
            }
    
    def _find_and_replace_text_enhanced(self, page: fitz.Page, old_text: str, new_text: str) -> bool:
        """
        Enhanced find and replace with centering, alignment, and bracket removal for tokens
        
        Args:
            page (fitz.Page): PDF page object
            old_text (str): Text to be replaced
            new_text (str): Replacement text
            
        Returns:
            bool: True if replacement was successful
        """
        try:
            # Get all text blocks with detailed information
            blocks = page.get_text("dict")
            replacements_made = 0
            
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "")
                        if old_text in text:
                            # Extract original properties
                            bbox = span["bbox"]
                            font = span.get("font", "helvetica")
                            size = span.get("size", 12)
                            color = span.get("color", 0)
                            
                            # Log original properties
                            logger.info(f"Remplacement trouvé: '{old_text}' -> '{new_text}'")
                            logger.info(f"  Position originale: {bbox}")
                            logger.info(f"  Police originale: {font}")
                            logger.info(f"  Taille originale: {size}")
                            logger.info(f"  Couleur originale: {color}")
                            
                            # Create white rectangle to cover old text
                            cover_rect = fitz.Rect(bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
                            page.draw_rect(cover_rect, color=(1, 1, 1), fill=(1, 1, 1))
                            
                            # Calculate positioning based on content type
                            if old_text == "[NOM ET PRÉNOM DU PARTICIPANT]":
                                # CENTER the name in the available space
                                available_width = bbox[2] - bbox[0]  # Width of original text area
                                center_x = bbox[0] + (available_width / 2)  # Center point
                                
                                # Estimate text width (approximate)
                                text_width = len(new_text) * size * 0.6  # Rough estimation
                                insert_x = center_x - (text_width / 2)  # Center the text
                                insert_y = bbox[1] + (bbox[3] - bbox[1]) * 0.75  # Vertical positioning
                                
                                # Force RED color for names (11741488 = 0xB30000 in RGB)
                                final_color = (0.7, 0.0, 0.0)  # Red color in RGB (0-1 range)
                                
                                logger.info(f"  NOM - Position centrée: x={insert_x:.1f}, y={insert_y:.1f}")
                                logger.info(f"  NOM - Largeur estimée du texte: {text_width:.1f}")
                                logger.info(f"  NOM - Point central: {center_x:.1f}")
                                
                            elif old_text == "Numéro de token":
                                # ALIGN token with "Identifiant (Token):" position AND add comma
                                # Token should start at x=288.2 (same as original "Numéro de token")
                                insert_x = 288.2  # Exact alignment with "Identifiant (Token):"
                                insert_y = bbox[1] + (bbox[3] - bbox[1]) * 0.75
                                
                                # Add comma to the token
                                new_text_with_comma = new_text + ","
                                
                                # Force RED color for tokens (11741488 = 0xB30000 in RGB)
                                final_color = (0.7, 0.0, 0.0)  # Red color in RGB (0-1 range)
                                
                                logger.info(f"  TOKEN - Position alignée: x={insert_x:.1f}, y={insert_y:.1f}")
                                logger.info(f"  TOKEN - Texte avec virgule: '{new_text_with_comma}'")
                                
                                # Use the text with comma
                                new_text = new_text_with_comma
                                
                                # Also need to remove both opening "[" and closing "]," bracket spans
                                self._remove_opening_bracket_span(page, blocks)
                                self._remove_bracket_span(page, blocks)
                                
                            else:
                                # Default positioning for other text
                                insert_x = bbox[0]
                                insert_y = bbox[1] + (bbox[3] - bbox[1]) * 0.75
                                # Convert original color to RGB if needed
                                if isinstance(color, int):
                                    final_color = (
                                        ((color >> 16) & 255) / 255.0,
                                        ((color >> 8) & 255) / 255.0,
                                        (color & 255) / 255.0
                                    )
                                else:
                                    final_color = (0, 0, 0)  # Default black
                            
                            # Try to use original font, with fallback
                            font_to_use = font
                            if font not in ["helvetica", "helvetica-bold", "times-roman", "times-bold", "courier"]:
                                if "archivo" in font.lower() or "black" in font.lower():
                                    font_to_use = "helvetica-bold"  # Bold fallback for names
                                else:
                                    font_to_use = "helvetica"
                                logger.info(f"  Police de substitution: {font} -> {font_to_use}")
                            
                            # Insert the new text with enhanced properties
                            insertion_point = fitz.Point(insert_x, insert_y)
                            page.insert_text(
                                insertion_point,
                                new_text,
                                fontname=font_to_use,
                                fontsize=size,
                                color=final_color
                            )
                            
                            replacements_made += 1
                            logger.info(f"  ✓ Texte inséré avec succès")
                            logger.info(f"    Police utilisée: {font_to_use}")
                            logger.info(f"    Taille: {size}")
                            logger.info(f"    Couleur finale: {final_color}")
                            logger.info(f"    Position finale: ({insert_x:.1f}, {insert_y:.1f})")
                            
            return replacements_made > 0
            
        except Exception as e:
            logger.error(f"Erreur lors du remplacement de texte: {str(e)}")
            return False
    
    def _remove_opening_bracket_span(self, page: fitz.Page, blocks: dict):
        """
        Remove the opening bracket "[" before token replacement
        """
        try:
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text.endswith(" : ["):
                            # Found the opening bracket span, we need to replace it with " : "
                            bbox = span["bbox"]
                            
                            # Cover the original span with white
                            cover_rect = fitz.Rect(bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
                            page.draw_rect(cover_rect, color=(1, 1, 1), fill=(1, 1, 1))
                            
                            # Re-insert the text without the opening bracket
                            new_text = text.replace(" : [", " : ")
                            font = span.get("font", "helvetica")
                            size = span.get("size", 12)
                            color = span.get("color", 0)
                            
                            # Convert color to RGB
                            if isinstance(color, int):
                                final_color = (
                                    ((color >> 16) & 255) / 255.0,
                                    ((color >> 8) & 255) / 255.0,
                                    (color & 255) / 255.0
                                )
                            else:
                                final_color = (0, 0, 0)
                            
                            # Try to use original font, with fallback
                            font_to_use = font
                            if font not in ["helvetica", "helvetica-bold", "times-roman", "times-bold", "courier"]:
                                font_to_use = "helvetica"
                            
                            # Insert the corrected text
                            insert_y = bbox[1] + (bbox[3] - bbox[1]) * 0.75
                            page.insert_text(
                                fitz.Point(bbox[0], insert_y),
                                new_text,
                                fontname=font_to_use,
                                fontsize=size,
                                color=final_color
                            )
                            
                            logger.info(f"  ✓ Crochet ouvrant supprimé: '{text}' -> '{new_text}' à position {bbox}")
                            return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du crochet ouvrant: {str(e)}")
            return False

    def _remove_bracket_span(self, page: fitz.Page, blocks: dict):
        """
        Remove the closing bracket span "]," after token replacement
        """
        try:
            for block in blocks["blocks"]:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span.get("text", "").strip()
                        if text == "],":
                            # Found the bracket span, cover it with white
                            bbox = span["bbox"]
                            cover_rect = fitz.Rect(bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
                            page.draw_rect(cover_rect, color=(1, 1, 1), fill=(1, 1, 1))
                            logger.info(f"  ✓ Crochets fermants supprimés: '{text}' à position {bbox}")
                            return True
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des crochets: {str(e)}")
            return False
    
    def generate_certificate(self, participant_data: Dict) -> str:
        """
        Generate a single certificate for a participant with enhanced formatting
        
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
                
                # Replace placeholders with enhanced method (convert name to uppercase)
                name_replaced = self._find_and_replace_text_enhanced(
                    page, "[NOM ET PRÉNOM DU PARTICIPANT]", name.upper()
                )
                token_replaced = self._find_and_replace_text_enhanced(
                    page, "Numéro de token", token
                )
                
                if name_replaced or token_replaced:
                    replacements_successful = True
                    logger.debug(f"Page {page_num + 1}: Name replaced: {name_replaced}, Token replaced: {token_replaced}")
                else:
                    logger.warning(f"No placeholders found on page {page_num + 1}")
            
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
        logger.info("Starting enhanced batch certificate generation...")
        
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
        logger.info(f"Enhanced certificate generation complete!")
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
        report_filename = f"IOAI_2025_Certificate_Generation_Report_V2_{timestamp}.csv"
        report_path = self.output_dir / report_filename
        
        summary_df.to_csv(report_path, index=False, encoding='utf-8')
        
        logger.info(f"Enhanced summary report saved: {report_filename}")
        return str(report_path)


def main():
    """
    Main function for command-line usage
    """
    parser = argparse.ArgumentParser(
        description="IOAI 2025 Certificate Generator V2 - Enhanced Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Enhanced Features:
  - Preserves original font family and size
  - Maintains text color and style
  - Keeps text positioning and alignment
  - Professional formatting preservation

Examples:
  python ioai_certificate_generator_v2.py --csv Phase2_Admitted_Not_In_Phase3.csv --template template.pdf
  python ioai_certificate_generator_v2.py --csv data.csv --template cert.pdf --output enhanced_certificates
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
        default='enhanced_certificates',
        help='Output directory for generated certificates (default: enhanced_certificates)'
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
        # Initialize enhanced generator
        generator = IOAICertificateGeneratorV2(
            csv_file=args.csv,
            template_pdf=args.template,
            output_dir=args.output
        )
        
        # Generate certificates
        certificates = generator.generate_all_certificates()
        
        # Generate summary report
        report_path = generator.generate_summary_report()
        
        print(f"\n{'='*70}")
        print("IOAI 2025 ENHANCED CERTIFICATE GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Generated: {len(certificates)} certificates")
        print(f"📁 Output directory: {generator.output_dir}")
        print(f"📊 Summary report: {Path(report_path).name}")
        print(f"🎨 Enhanced formatting: Font, size, and style preserved")
        print(f"{'='*70}")
        
    except Exception as e:
        logger.error(f"Enhanced certificate generation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
