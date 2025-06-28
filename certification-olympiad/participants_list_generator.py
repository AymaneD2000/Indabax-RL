#!/usr/bin/env python3
"""
IOAI 2025 Participants List Generator

This script reads participant data from multiple CSV files, removes duplicates,
and generates a professional PDF document with participant information.

Created: 2025
"""

import pandas as pd
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import logging

class ParticipantsListGenerator:
    def __init__(self):
        self.setup_logging()
        self.participants_data = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_filename = f"participants_list_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def truncate_establishment_name(self, establishment_name, max_length=45):
        """Truncate establishment names that are too long for PDF formatting"""
        if pd.isna(establishment_name) or establishment_name == 'nan':
            return 'N/A'
        
        establishment_name = str(establishment_name).strip()
        
        # Common abbreviations for French educational institutions
        abbreviations = {
            'Lycée': 'Lyc.',
            'École': 'Éc.',
            'Complexe scolaire': 'Comp. scol.',
            'Institut': 'Inst.',
            'Université': 'Univ.',
            'Faculté': 'Fac.',
            'Centre': 'C.',
            'Établissement': 'Étab.',
            'International': 'Int.',
            'Technique': 'Tech.',
            'Professionnel': 'Prof.',
            'Supérieur': 'Sup.',
            'Supérieure': 'Sup.',
            'Privé': 'Priv.',
            'Privée': 'Priv.',
            'Public': 'Pub.',
            'Publique': 'Pub.',
            'Bamako': 'Bko',
            'de Bamako': 'de Bko'
        }
        
        # Apply abbreviations
        abbreviated_name = establishment_name
        for full_word, abbrev in abbreviations.items():
            abbreviated_name = abbreviated_name.replace(full_word, abbrev)
        
        # If still too long, truncate and add ellipsis
        if len(abbreviated_name) > max_length:
            abbreviated_name = abbreviated_name[:max_length-3] + '...'
        
        return abbreviated_name
        
    def read_csv_files(self):
        """Read and combine data from all CSV files"""
        csv_files = [
            ("Atttestation  phase3.csv", "Phase 3"),
            ("Atttestation  phase4.csv", "Phase 4"), 
            ("Phase2_Admitted_Not_In_Phase3.csv", "Phase 2")
        ]
        
        all_data = []
        
        for csv_file, phase_name in csv_files:
            if os.path.exists(csv_file):
                try:
                    self.logger.info(f"Reading {csv_file}")
                    df = pd.read_csv(csv_file)
                    
                    # Standardize column names
                    if 'Nom Complet' in df.columns:
                        # Extract relevant columns
                        relevant_columns = ['Token', 'Nom Complet', 'Ville', 'Dernier établissement scolaire']
                        df_clean = df[relevant_columns].copy()
                        df_clean = df_clean.rename(columns={
                            'Dernier établissement scolaire': 'Etablissement'
                        })
                        # Add phase information
                        df_clean['Phase'] = phase_name
                        all_data.append(df_clean)
                        self.logger.info(f"Added {len(df_clean)} participants from {csv_file} ({phase_name})")
                    
                except Exception as e:
                    self.logger.error(f"Error reading {csv_file}: {str(e)}")
            else:
                self.logger.warning(f"File {csv_file} not found")
        
        if all_data:
            # Combine all dataframes
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates based on Token
            initial_count = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['Token'], keep='first')
            final_count = len(combined_df)
            
            self.logger.info(f"Combined data: {initial_count} total entries, {final_count} unique participants")
            self.logger.info(f"Removed {initial_count - final_count} duplicates")
            
            # Clean data
            combined_df = combined_df.dropna(subset=['Token', 'Nom Complet'])
            combined_df = combined_df.sort_values('Nom Complet')
            
            # Clean up text data
            for col in ['Nom Complet', 'Ville', 'Etablissement', 'Phase']:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].astype(str).str.strip()
            
            # Truncate long establishment names for better PDF formatting
            if 'Etablissement' in combined_df.columns:
                combined_df['Etablissement'] = combined_df['Etablissement'].apply(self.truncate_establishment_name)
            
            self.participants_data = combined_df.to_dict('records')
            self.logger.info(f"Final participant count: {len(self.participants_data)}")
            
        else:
            self.logger.error("No valid data found in CSV files")
            
        return len(self.participants_data)
    
    def generate_pdf(self, output_filename="IOAI_2025_Participants_List.pdf"):
        """Generate a professional PDF with participant information"""
        if not self.participants_data:
            self.logger.error("No participant data available for PDF generation")
            return False
            
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_filename,
                pagesize=A4,
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=TA_CENTER,
                textColor=colors.darkblue
            )
            
            subtitle_style = ParagraphStyle(
                'CustomSubtitle',
                parent=styles['Normal'],
                fontSize=12,
                spaceAfter=20,
                alignment=TA_CENTER,
                textColor=colors.grey
            )
            
            # Build document content
            story = []
            
            # Title
            title = Paragraph("INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025", title_style)
            story.append(title)
            
            subtitle = Paragraph("Liste Complète des Participants", subtitle_style)
            story.append(subtitle)
            
            # Generation info
            generation_info = Paragraph(
                f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}<br/>"
                f"Nombre total de participants: {len(self.participants_data)}",
                styles['Normal']
            )
            story.append(generation_info)
            story.append(Spacer(1, 20))
            
            # Create table data
            table_data = [
                ['#', 'Token', 'Nom Complet', 'Ville', 'Établissement', 'Phase']
            ]
            
            for idx, participant in enumerate(self.participants_data, 1):
                row = [
                    str(idx),
                    participant.get('Token', 'N/A'),
                    participant.get('Nom Complet', 'N/A'),
                    participant.get('Ville', 'N/A'),
                    participant.get('Etablissement', 'N/A'),
                    participant.get('Phase', 'N/A')
                ]
                table_data.append(row)
            
            # Create table with optimized column widths
            table = Table(table_data, colWidths=[0.3*inch, 1.4*inch, 1.8*inch, 0.8*inch, 2.4*inch, 0.7*inch])
            
            # Table style
            table.setStyle(TableStyle([
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                
                # Data rows
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('ALIGN', (0, 1), (0, -1), 'CENTER'),  # Index column
                ('ALIGN', (1, 1), (-1, -1), 'LEFT'),   # Other columns
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('FONTSIZE', (4, 1), (4, -1), 7),  # Smaller font for establishment column
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
                
                # Grid
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                
                # Padding
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(table)
            
            # Footer note
            story.append(Spacer(1, 20))
            footer_note = Paragraph(
                "Note: Cette liste ne contient pas les numéros de téléphone pour des raisons de confidentialité.",
                styles['Italic']
            )
            story.append(footer_note)
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"PDF successfully generated: {output_filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating PDF: {str(e)}")
            return False
    
    def generate_summary_report(self):
        """Generate a summary report of the participants"""
        if not self.participants_data:
            return
        
        self.logger.info("\n" + "="*50)
        self.logger.info("PARTICIPANTS SUMMARY REPORT")
        self.logger.info("="*50)
        
        # Total count
        total_participants = len(self.participants_data)
        self.logger.info(f"Total Participants: {total_participants}")
        
        # Count by city, establishment, and phase
        cities = {}
        establishments = {}
        phases = {}
        
        for participant in self.participants_data:
            city = participant.get('Ville', 'Unknown')
            establishment = participant.get('Etablissement', 'Unknown')
            phase = participant.get('Phase', 'Unknown')
            
            cities[city] = cities.get(city, 0) + 1
            establishments[establishment] = establishments.get(establishment, 0) + 1
            phases[phase] = phases.get(phase, 0) + 1
        
        # Phase distribution
        self.logger.info("\nDistribution by Phase:")
        sorted_phases = sorted(phases.items(), key=lambda x: x[0])
        for phase, count in sorted_phases:
            cert_note = " (Certificat délivré)" if phase in ["Phase 3", "Phase 4"] else " (Pas de certificat)"
            self.logger.info(f"  {phase}: {count} participants{cert_note}")
        
        # Top cities
        self.logger.info("\nTop 5 Cities:")
        sorted_cities = sorted(cities.items(), key=lambda x: x[1], reverse=True)[:5]
        for city, count in sorted_cities:
            self.logger.info(f"  {city}: {count} participants")
        
        # Top establishments
        self.logger.info("\nTop 5 Establishments:")
        sorted_establishments = sorted(establishments.items(), key=lambda x: x[1], reverse=True)[:5]
        for establishment, count in sorted_establishments:
            self.logger.info(f"  {establishment}: {count} participants")
        
        self.logger.info("="*50)

def main():
    """Main function"""
    print("IOAI 2025 Participants List Generator")
    print("="*50)
    
    generator = ParticipantsListGenerator()
    
    # Read CSV files
    participant_count = generator.read_csv_files()
    
    if participant_count > 0:
        # Generate summary report
        generator.generate_summary_report()
        
        # Generate PDF
        output_filename = f"IOAI_2025_Participants_List_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        success = generator.generate_pdf(output_filename)
        
        if success:
            print(f"\n✅ PDF successfully generated: {output_filename}")
            print(f"📊 Total participants: {participant_count}")
        else:
            print("\n❌ Failed to generate PDF")
    else:
        print("\n❌ No participant data found")

if __name__ == "__main__":
    main() 