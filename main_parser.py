import camelot
import pandas as pd
import fitz
import csv
import re
from typing import Optional, Dict, Any
import os
import sys 

def parse_jamaican_amount(amount_str: str) -> Optional[float]:
    if not amount_str or not amount_str.strip():
        return None
    clean_amount = re.sub(r'J\$\s*', '', amount_str.strip())
    is_negative = '-' in clean_amount
    clean_amount = re.sub(r'[+\-\s]', '', clean_amount)
    clean_amount = clean_amount.replace(',', '')
    try:
        parsed = float(clean_amount)
        return parsed
    except ValueError:
        return None

def format_currency(amount: float) -> str:
    return f"J${amount:,.2f}"

def clean_bank_statement_csv(input_file: str, output_file: str) -> Dict[str, Any]:
    clean_rows = []
    valid_amount_count = 0
    total_credits = 0.0
    total_debits = 0.0
    skipped_rows = 0
    headers = ['Date', 'Description', 'Amount', 'Type', 'Balance']
    clean_rows.append(headers)

    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        sample = file.read(1024)
        file.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        reader = csv.reader(file, delimiter=delimiter)

        for row_num, row in enumerate(reader, 1):
            if not row or len(row) < 4:
                skipped_rows += 1
                continue
            if (row[0].strip().lower() in ['date', 'transaction', '0'] or 
                not row[0].strip()):
                skipped_rows += 1
                continue

            date = row[0].strip()
            description = row[1].strip() if len(row) > 1 else ''
            credit_str = row[2].strip() if len(row) > 2 else ''
            debit_str = row[3].strip() if len(row) > 3 else ''
            balance_str = row[4].strip() if len(row) > 4 else ''

            amount = 0.0
            transaction_type = ''

            if credit_str:
                credit_amount = parse_jamaican_amount(credit_str)
                if credit_amount is not None:
                    amount = credit_amount
                    transaction_type = 'CREDIT'
                    total_credits += credit_amount
                    valid_amount_count += 1

            if debit_str:
                debit_amount = parse_jamaican_amount(debit_str)
                if debit_amount is not None:
                    amount = debit_amount
                    transaction_type = 'DEBIT'
                    total_debits += debit_amount
                    valid_amount_count += 1

            balance = parse_jamaican_amount(balance_str)
            balance_formatted = format_currency(balance) if balance is not None else ''

            if amount != 0 and description:
                clean_rows.append([
                    date,
                    description,
                    format_currency(amount),
                    transaction_type,
                    balance_formatted
                ])

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(clean_rows)

    net_amount = total_credits - total_debits
    cleaned_rows_count = len(clean_rows) - 1

    return {
        'valid_transactions': valid_amount_count,
        'total_credits': total_credits,
        'total_debits': total_debits,
        'net_amount': net_amount,
        'skipped_rows': skipped_rows,
        'cleaned_rows': cleaned_rows_count,
        'output_file': output_file
    }

def extract_transactions_from_pdf(pdf_path: str, raw_csv_path: str):
    page_count = fitz.open(pdf_path).page_count
    str_pages = f'2-{page_count - 2}'

    transactions = camelot.read_pdf(
        pdf_path,
        flavor='stream',
        table_areas=['30,630,600,60'],
        pages=str_pages
    )

    tables = []
    for table in transactions:
        t = table.df
        group_id = t[0].ne('').cumsum()
        df_combined = t.astype(str).groupby(group_id).agg(lambda x: ' '.join(x[x != '']).strip())
        tables.append(df_combined)

    all_transactions = pd.concat(tables, join='outer')
    all_transactions.to_csv(raw_csv_path, index=False)

def run_parser_and_cleaner(pdf_path: str, output_csv: str):
    temp_raw_csv = "raw_output_temp.csv"
    
    print("🔍 Extracting transactions from PDF...")
    extract_transactions_from_pdf(pdf_path, temp_raw_csv)
    
    print("🧼 Cleaning extracted CSV...")
    stats = clean_bank_statement_csv(temp_raw_csv, output_csv)

    os.remove(temp_raw_csv)  # Clean up temp file
    
    for k, v in stats.items():
        print(f"  {k}: {v}")

# === MAIN ===
if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python full_parser_cleaner.py <input_pdf> [output_csv]")
        print("Example: python full_parser_cleaner.py my_statement.pdf cleaned.csv")
        sys.exit(1)

    pdf_path = sys.argv[1]
    cleaned_output_csv = sys.argv[2] if len(sys.argv) == 3 else "cleaned_output.csv"

    run_parser_and_cleaner(pdf_path, cleaned_output_csv)
