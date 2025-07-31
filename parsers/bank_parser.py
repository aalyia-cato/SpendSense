import camelot
import pandas as pd
import fitz
import re
import os
from typing import Optional 

class BankStatementParser:
    def parse_file(self, file_path, file_type):
        """Use your original parsing logic with minor adaptations"""
        if file_type == 'pdf':
            return self._parse_pdf(file_path)
        else:
            return self._parse_csv(file_path)

    def _parse_pdf(self, pdf_path):
        """Your original PDF parsing logic"""
        page_count = fitz.open(pdf_path).page_count
        str_pages = f'2-{page_count - 2}'

        tables = camelot.read_pdf(
            pdf_path,
            flavor='stream',
            table_areas=['30,630,600,60'],
            pages=str_pages
        )

        processed_tables = []
        for table in tables:
            t = table.df
            group_id = t[0].ne('').cumsum()
            df_combined = t.astype(str).groupby(group_id).agg(lambda x: ' '.join(x[x != '']).strip())
            processed_tables.append(df_combined)
        
        return pd.concat(processed_tables, join='outer')

    def _parse_csv(self, csv_path):
        """Simple CSV reader maintaining your format"""
        return pd.read_csv(csv_path)

    @staticmethod
    def parse_jamaican_amount(amount_str: str) -> Optional[float]:
        """Your original amount parser"""
        if not amount_str or not amount_str.strip():
            return None
        clean_amount = re.sub(r'J\$\s*', '', amount_str.strip())
        is_negative = '-' in clean_amount
        clean_amount = re.sub(r'[+\-\s]', '', clean_amount)
        clean_amount = clean_amount.replace(',', '')
        try:
            parsed = float(clean_amount)
            return -parsed if is_negative else parsed
        except ValueError:
            return None