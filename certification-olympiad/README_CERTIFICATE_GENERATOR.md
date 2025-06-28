# 🏆 IOAI 2025 Certificate Generator

Professional tool for automating IOAI Phase 2 certificate generation with participant data from CSV files.

## ✨ Features

- **Automated PDF Processing**: Replace placeholders in PDF templates with participant data
- **Batch Generation**: Process all participants in one command
- **Professional Logging**: Comprehensive logging and error reporting
- **Data Validation**: Automatic validation of CSV data and PDF templates
- **Safe File Naming**: Automatic sanitization of participant names for filenames
- **Summary Reports**: Generate detailed reports of the certificate generation process
- **Error Handling**: Robust error handling and recovery

## 📋 Requirements

- Python 3.7 or higher
- Internet connection for initial package installation

## 🚀 Quick Start

### Option 1: Simple Setup (Recommended)

1. **Place your files** in the same directory:
   ```
   certification-olympiad/
   ├── Phase2_Admitted_Not_In_Phase3.csv
   ├── INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025 - PHASE 2.pdf
   ├── setup_and_run.py
   ├── requirements.txt
   └── ioai_certificate_generator.py
   ```

2. **Run the quick setup**:
   ```bash
   python setup_and_run.py
   ```

That's it! The script will:
- Install required packages
- Generate all certificates
- Create a summary report

### Option 2: Manual Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the generator**:
   ```bash
   python ioai_certificate_generator.py \
     --csv "Phase2_Admitted_Not_In_Phase3.csv" \
     --template "INTERNATIONAL OLYMPIAD IN ARTIFICIAL INTELLIGENCE (IOAI) 2025 - PHASE 2.pdf" \
     --output "generated_certificates"
   ```

## 📁 File Structure

### Input Files
- **CSV File**: Contains participant data with columns:
  - `Token`: Unique participant token
  - `Nom Complet`: Full name of participant
  - `Ranking`: (Optional) Participant ranking
  
- **PDF Template**: Certificate template with placeholders:
  - `[NOM ET PRÉNOM DU PARTICIPANT]`: Will be replaced with full name
  - `[Numéro de token]`: Will be replaced with token

### Output Structure
```
generated_certificates/
├── IOAI_2025_Certificate_Tabara_keita_olp-micenas-uhs3g12.pdf
├── IOAI_2025_Certificate_Alphady_Sadio_Diarra_olp-megara-hri8v78.pdf
├── ...
├── IOAI_2025_Certificate_Generation_Report_20241201_143022.csv
└── ioai_certificate_generator.log
```

## 🛠️ Advanced Usage

### Command Line Options

```bash
python ioai_certificate_generator.py [OPTIONS]

Options:
  --csv TEXT        Path to CSV file containing participant data [REQUIRED]
  --template TEXT   Path to PDF certificate template [REQUIRED]
  --output TEXT     Output directory for generated certificates [default: certificates]
  --verbose         Enable verbose logging
  --help            Show help message
```

### Examples

```bash
# Basic usage
python ioai_certificate_generator.py --csv participants.csv --template template.pdf

# Custom output directory
python ioai_certificate_generator.py \
  --csv participants.csv \
  --template template.pdf \
  --output my_certificates

# Verbose logging
python ioai_certificate_generator.py \
  --csv participants.csv \
  --template template.pdf \
  --verbose
```

## 📊 Data Format

### CSV File Requirements

Your CSV file must contain these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `Token` | ✅ Yes | Unique participant identifier |
| `Nom Complet` | ✅ Yes | Full name of participant |
| `Ranking` | ❌ No | Participant ranking (for sorting) |

### Example CSV Data
```csv
Ranking,Token,Nom Complet
1,olp-micenas-uhs3g12,Tabara keita
1,olp-megara-hri8v78,Alphady Sadio Diarra
2,olp-micenas-yec2s16,Mohamed Dourbéla
```

## 🔍 Features Details

### Automatic Data Cleaning
- Removes leading/trailing whitespace
- Handles missing data gracefully
- Validates required columns

### Safe File Naming
- Removes special characters from names
- Replaces spaces with underscores
- Prevents file system conflicts

### Professional Logging
- Timestamped log entries
- Separate file logging (`ioai_certificate_generator.log`)
- Console output for real-time monitoring

### Error Recovery
- Continues processing even if individual certificates fail
- Detailed error reporting
- Summary of successful vs failed generations

## 📈 Output Reports

### Certificate Generation Report
The tool generates a comprehensive CSV report containing:
- Participant ranking
- Full name
- Token
- Generation status (Success/Failed)
- Output filename

### Log File
Detailed log file includes:
- Processing timestamps
- Success/error messages
- Performance metrics
- Debugging information

## ⚠️ Troubleshooting

### Common Issues

**"PDF template not found"**
- Ensure the PDF file path is correct
- Check file permissions

**"CSV file not found"**
- Verify the CSV file exists in the specified location
- Check file encoding (UTF-8 recommended)

**"Missing required columns"**
- Ensure CSV contains 'Token' and 'Nom Complet' columns
- Check column names match exactly

**"No placeholders found"**
- Verify PDF template contains the exact text:
  - `[NOM ET PRÉNOM DU PARTICIPANT]`
  - `[Numéro de token]`

### Performance Tips

For large datasets (100+ certificates):
- Use SSD storage for faster PDF processing
- Ensure sufficient disk space (each certificate ~100KB)
- Consider running in batches if memory is limited

## 🔧 Customization

### Adding New Placeholders

To add more replaceable fields:

1. Update the CSV with new columns
2. Modify the `generate_certificate` method:
   ```python
   # Add new replacement
   self._find_and_replace_text(page, "[NEW_PLACEHOLDER]", new_value)
   ```

### Custom File Naming

Modify the `_sanitize_filename` method to change naming conventions:
```python
def _sanitize_filename(self, name: str) -> str:
    # Your custom naming logic here
    return custom_name
```

## 📞 Support

For issues or questions:
1. Check the log file for detailed error messages
2. Verify input file formats
3. Ensure all dependencies are installed
4. Review the troubleshooting section

## 📝 License

This tool is provided as-is for IOAI 2025 certificate generation.

---

**Professional Certificate Generation for IOAI 2025 🏆** 