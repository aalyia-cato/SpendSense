from datetime import datetime
from models import db, Transaction
from ml.predictor import CategoryPredictor
from parsers.bank_parser import BankStatementParser

class TransactionProcessor:
    def __init__(self):
        self.parser = BankStatementParser()
        self.predictor = CategoryPredictor()

    def process_uploaded_file(self, user_id, file_path, file_type):
        try:
            raw_df = self.parser.parse_file(file_path, file_type)

            transactions = []
            for _, row in raw_df.iterrows():
                if len(row) < 4:
                    continue

                date = str(row[0]).strip()
                description = str(row[1]).strip()
                credit_str = str(row[2]).strip() if len(row) > 2 else ''
                debit_str = str(row[3]).strip() if len(row) > 3 else ''

                # Parse amounts
                credit_amount = self.parser.parse_jamaican_amount(credit_str)
                debit_amount = self.parser.parse_jamaican_amount(debit_str)

                # Skip if no valid amount
                if credit_amount is None and debit_amount is None:
                    continue

                # Determine amount and type
                if debit_amount is not None:
                    amount = debit_amount
                    txn_type = 'debit'
                else:
                    amount = credit_amount
                    txn_type = 'credit'

                # Skip zero amounts or empty descriptions
                if amount == 0 or not description:
                    continue

                # Predict category
                category_id = self.predictor.predict_for_user(
                    user_id=user_id,
                    description=description,
                    amount=amount,
                    is_income=(txn_type == 'credit')
                )

                # Parse date
                date_obj = self._parse_date(date)

                # Create transaction object
                transactions.append(Transaction(
                    user_id=user_id,
                    category_id=category_id,
                    date=date_obj,
                    description=description,
                    amount=amount,
                    type=txn_type
                ))

            if transactions:
                db.session.add_all(transactions)
                db.session.commit()

            return len(transactions)

        except Exception as e:
            db.session.rollback()
            print(f"Processing failed: {str(e)}")
            raise

    def _parse_date(self, date_str):
        """Parse dates in various Jamaican bank formats"""
        date_str = str(date_str).strip()

        # Try common Jamaican bank formats
        formats = [
            '%d-%b-%y', '%d %b %Y', '%d/%m/%Y',
            '%d-%b-%Y', '%d %b %y', '%Y-%m-%d',
            '%d/%m/%y', '%b %d, %Y'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue

        # Fallback to today if all parsing fails (with warning)
        print(f"⚠️ Could not parse date: {date_str}, using today's date")
        return datetime.now().date()