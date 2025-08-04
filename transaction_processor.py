from datetime import datetime
from models import db, Transaction
from ml.predictor import CategoryPredictor
from parsers.bank_parser import BankStatementParser
import re

class TransactionProcessor:
    def __init__(self):
        self.parser = BankStatementParser()
        self.predictor = CategoryPredictor()

    def process_uploaded_file(self, user_id, file_path, file_type):
        try:
            raw_df = self.parser.parse_file(file_path, file_type)
            print(f"📊 Parsed {len(raw_df)} rows from file")

            transactions = []
            skipped_count = 0
            
            for idx, row in raw_df.iterrows():
                if len(row) < 4:
                    skipped_count += 1
                    continue

                # Get raw values
                date_raw = str(row[0]).strip()
                description_raw = str(row[1]).strip()
                credit_str = str(row[2]).strip() if len(row) > 2 else ''
                debit_str = str(row[3]).strip() if len(row) > 3 else ''

                print(f"🔍 Row {idx}: Date='{date_raw}', Desc='{description_raw[:30]}...', Credit='{credit_str}', Debit='{debit_str}'")

                # Parse amounts - handle both string and float inputs
                credit_amount = self._parse_amount(credit_str)
                debit_amount = self._parse_amount(debit_str)

                # Skip if no valid amount
                if credit_amount is None and debit_amount is None:
                    print(f"⚠️ Skipping row {idx}: No valid amount")
                    skipped_count += 1
                    continue

                # Determine amount and type
                if debit_amount is not None and debit_amount != 0:
                    amount = abs(debit_amount)  # Make sure it's positive
                    txn_type = 'debit'
                elif credit_amount is not None and credit_amount != 0:
                    amount = abs(credit_amount)  # Make sure it's positive
                    txn_type = 'credit'
                else:
                    print(f"⚠️ Skipping row {idx}: Zero amount")
                    skipped_count += 1
                    continue

                # Skip if description is empty or invalid
                if not description_raw or description_raw.lower() in ['nan', 'none', '']:
                    print(f"⚠️ Skipping row {idx}: Empty description")
                    skipped_count += 1
                    continue

                # Parse date with better error handling
                date_obj = self._parse_date(date_raw)
                if date_obj is None:
                    print(f"⚠️ Skipping row {idx}: Could not parse date '{date_raw}'")
                    skipped_count += 1
                    continue

                # Predict category
                print(f"🤖 Predicting category for: '{description_raw}' (${amount}, {txn_type})")
                category_id = self.predictor.predict_for_user(
                    user_id=user_id,
                    description=description_raw,
                    amount=amount,
                    is_income=(txn_type == 'credit')
                )

                if category_id is None:
                    print(f"⚠️ Skipping row {idx}: Could not determine category")
                    skipped_count += 1
                    continue

                # Create transaction object
                transaction = Transaction(
                    user_id=user_id,
                    category_id=category_id,
                    date=date_obj,
                    description=description_raw,
                    amount=amount,
                    type=txn_type
                )
                transactions.append(transaction)
                print(f"✅ Created transaction: {description_raw[:30]}... on {date_obj}")

            if transactions:
                db.session.add_all(transactions)
                db.session.commit()
                print(f"✅ Successfully saved {len(transactions)} transactions")
            
            print(f"📈 Processing summary: {len(transactions)} saved, {skipped_count} skipped")
            return len(transactions)

        except Exception as e:
            db.session.rollback()
            print(f"❌ Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _parse_date(self, date_str):
        """Improved date parsing for Jamaican bank formats"""
        if not date_str or str(date_str).lower() in ['nan', 'none', '']:
            return None
            
        date_str = str(date_str).strip()
        print(f"🗓️ Parsing date: '{date_str}'")

        # Remove extra whitespace and normalize
        date_str = re.sub(r'\s+', ' ', date_str)

        # Try common Jamaican bank formats
        formats = [
            '%d%b',             # 07APR (new format)
            '%d-%b-%y',      # 15-Jan-23
            '%d-%b-%Y',      # 15-Jan-2023  
            '%d %b %Y',      # 15 Jan 2023
            '%d %b %y',      # 15 Jan 23
            '%d/%m/%Y',      # 15/01/2023
            '%d/%m/%y',      # 15/01/23
            '%Y-%m-%d',      # 2023-01-15
            '%m/%d/%Y',      # 01/15/2023
            '%m/%d/%y',      # 01/15/23
            '%b %d, %Y',     # Jan 15, 2023
            '%d-%m-%Y',      # 15-01-2023
            '%d-%m-%y',      # 15-01-23
        ]

        for fmt in formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                if fmt == '%d%b':
                    parsed_date = parsed_date.replace(year=datetime.now().year)
                print(f"✅ Successfully parsed '{date_str}' as {parsed_date} using format '{fmt}'")
                return parsed_date
            except ValueError:
                continue

        # Try with regex for mixed formats
        # Pattern: DD-MMM-YY or DD MMM YYYY etc
        month_map = {
            'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
            'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
            'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
        }
        
        # Try to extract day, month, year with regex
        pattern = r'(\d{1,2})[-\s/]([a-zA-Z]{3}|\d{1,2})[-\s/](\d{2,4})'
        match = re.search(pattern, date_str.lower())
        
        if match:
            day, month, year = match.groups()
            
            # Convert month name to number if needed
            if month.isalpha() and month.lower() in month_map:
                month = month_map[month.lower()]
            
            # Convert 2-digit year to 4-digit
            if len(year) == 2:
                year_int = int(year)
                if year_int > 50:  # Assume 1950-1999
                    year = f"19{year}"
                else:  # Assume 2000-2049
                    year = f"20{year}"
            
            try:
                parsed_date = datetime(int(year), int(month), int(day)).date()
                print(f"✅ Successfully parsed '{date_str}' as {parsed_date} using regex")
                return parsed_date
            except ValueError:
                pass

        print(f"❌ Could not parse date: '{date_str}'")
        return None

    def _parse_amount(self, amount_str):
        """Parse amount handling both Jamaican format and plain numbers"""
        if not amount_str or str(amount_str).strip() in ['', 'nan', 'None', '0.0', '0']:
            return None
            
        try:
            # If it's already a float, use it
            if isinstance(amount_str, (int, float)):
                return float(amount_str) if amount_str != 0 else None
                
            # Clean string format
            clean_str = str(amount_str).strip()
            
            # Handle Jamaican format: J$1,234.56 or negative amounts
            clean_str = clean_str.replace('J$', '').replace(',', '')
            
            # Check for negative
            is_negative = '-' in clean_str or '(' in clean_str
            
            # Remove all non-numeric except decimal point
            clean_str = re.sub(r'[^\d\.]', '', clean_str)
            
            if not clean_str:
                return None
                
            amount = float(clean_str)
            return -amount if is_negative else amount
            
        except (ValueError, TypeError):
            print(f"⚠️ Could not parse amount: '{amount_str}'")
            return None